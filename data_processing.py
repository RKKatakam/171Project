import zipfile
import urllib.request
from pathlib import Path
from collections import Counter
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from config import RATING_MIN, RATING_MAX


def download_movielens20m(root: Path):
    """Downloads and extracts the MovieLens 20M dataset if not present."""
    url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    path = root / "ml-20m"
    if path.exists():
        print("MovieLens-20M already present")
        return path

    print("Downloading MovieLens-20M...")
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "ml-20m.zip"

    # Download with progress tracking
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(
                f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                end="",
                flush=True,
            )
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            print(f"\rDownloaded: {mb_downloaded:.1f} MB", end="", flush=True)

    start_time = time.time()
    urllib.request.urlretrieve(url, zip_path, reporthook=show_progress)
    download_time = time.time() - start_time
    print(f"\nDownload completed in {download_time:.1f} seconds")

    print("Extracting files...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    zip_path.unlink()
    print("Extraction completed!")
    return path


def reindex(df, col):
    """Re-indexes a DataFrame column to create integer labels and returns the number of unique items."""
    df.loc[:, col + "_new"], uniq = pd.factorize(df[col], sort=True)
    return len(uniq)


def build_graph20m(ratings, movies, tags_df, sample_users=0, max_tags=800):
    """Builds the heterogeneous graph from the MovieLens dataframes."""
    if sample_users > 0:
        keep_users = np.random.choice(
            ratings["userId"].unique(), size=sample_users, replace=False
        )
        ratings = ratings[ratings["userId"].isin(keep_users)]

    n_users = reindex(ratings, "userId")
    unique_mids = ratings["movieId"].unique()
    movies = movies[movies["movieId"].isin(unique_mids)].copy()
    n_items = reindex(ratings, "movieId")
    reindex(movies, "movieId")

    # Split data into train, validation, and test sets
    ratings["rating_norm"] = (ratings.rating - RATING_MIN) / (RATING_MAX - RATING_MIN)
    ratings = ratings.sort_values("timestamp")
    ratings["rank_latest"] = ratings.groupby("userId_new")["timestamp"].rank(
        "first", ascending=False
    )

    test_df = ratings[ratings.rank_latest == 1]
    val_df = ratings[ratings.rank_latest == 2]
    train_df = ratings[ratings.rank_latest > 2]

    # Create HeteroData object
    data = HeteroData()
    data["user"].num_nodes = n_users
    data["movie"].num_nodes = n_items

    # Add edges and edge labels
    src, dst = (
        torch.tensor(train_df.userId_new.values),
        torch.tensor(train_df.movieId_new.values),
    )
    edge_label = torch.tensor(train_df.rating_norm.values, dtype=torch.float32)
    data["user", "rates", "movie"].edge_index = torch.stack([src, dst])
    data["user", "rates", "movie"].edge_label = edge_label
    data["movie", "rev_rates", "user"].edge_index = torch.stack([dst, src])
    data["movie", "rev_rates", "user"].edge_label = edge_label

    data.val_edges = (
        torch.tensor(val_df.userId_new.values),
        torch.tensor(val_df.movieId_new.values),
        torch.tensor(val_df.rating_norm.values, dtype=torch.float32),
    )
    data.test_edges = (
        torch.tensor(test_df.userId_new.values),
        torch.tensor(test_df.movieId_new.values),
        torch.tensor(test_df.rating_norm.values, dtype=torch.float32),
    )

    # Process movie features (genres and tags)
    movies["genres"] = movies["genres"].str.split("|")
    genres = sorted(movies.explode("genres")["genres"].unique())
    g2i = {g: i for i, g in enumerate(genres)}
    genre_mat = torch.zeros((n_items, len(genres)), dtype=torch.float32)
    exploded_genres = movies.explode("genres")
    exploded_genres["gidx"] = exploded_genres["genres"].map(g2i)
    genre_mat[exploded_genres.movieId_new.values, exploded_genres.gidx.values] = 1.0

    tags_df = tags_df[tags_df.movieId.isin(unique_mids)].copy()
    tags_df["tag"] = tags_df["tag"].str.lower().str.strip()
    tag_counts = Counter(tags_df["tag"])
    vocab = [w for w, _ in tag_counts.most_common(max_tags)]
    tag2i = {t: i for i, t in enumerate(vocab)}
    tag_mat = torch.zeros((n_items, len(vocab)), dtype=torch.float32)
    tags_df = tags_df[tags_df.tag.isin(tag2i)]
    tags_df = tags_df.merge(
        movies[["movieId", "movieId_new"]], on="movieId", how="left"
    )
    tags_df["tidx"] = tags_df.tag.map(tag2i)
    tag_mat[tags_df.movieId_new.values, tags_df.tidx.values] = 1.0

    data["movie"].x = torch.cat([genre_mat, tag_mat], dim=1)

    return data, n_users, n_items, len(genres), len(vocab)
