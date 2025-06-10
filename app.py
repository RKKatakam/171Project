from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from pathlib import Path

# Import from your modules (based on inference.py)
import config
from data_processing import download_movielens20m, build_graph20m
from model import GATRec

app = Flask(__name__)
CORS(
    app,
    origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ],
)

# Global variables for model and data
model = None
graph = None
movies_df = None
device = None


def find_movie_ids(titles, movies_df):
    """
    Finds the internal 'movieId_new' for a list of movie titles.
    Uses case-insensitive search.
    """
    found_ids = []
    not_found_titles = []

    # Create a mapping from lowercase title to movieId_new for efficient lookup
    title_map = movies_df.set_index(movies_df["title"].str.lower())["movieId_new"]

    for title in titles:
        lower_title = title.lower()
        # First, try for an exact match
        if lower_title in title_map.index:
            found_ids.append(title_map[lower_title])
            print(f"✅ Found exact match for: '{title}'")
        else:
            # If no exact match, try a partial match
            matches = movies_df[
                movies_df["title"]
                .str.lower()
                .str.contains(re.escape(lower_title), na=False)
            ]
            if not matches.empty:
                match_id = matches.iloc[0]["movieId_new"]
                match_title = matches.iloc[0]["title"]
                found_ids.append(match_id)
                print(f"✅ Found partial match for '{title}': '{match_title}'")
            else:
                not_found_titles.append(title)
                print(f"❌ Could not find a match for: '{title}'")

    return list(set(found_ids)), not_found_titles


def add_new_user_to_graph(graph, model, liked_movie_ids, device):
    """
    Adds a new user node and edges to the graph and expands the model's embedding layer.
    """
    new_user_id = graph["user"].num_nodes
    graph["user"].num_nodes += 1

    # Add edges for the new user's liked movies (rating 5 -> normalized to 1.0)
    liked_movie_ids = torch.as_tensor(liked_movie_ids, dtype=torch.long)
    n_edges = liked_movie_ids.numel()
    src_nodes = torch.full((n_edges,), new_user_id, dtype=torch.long)
    edge_label = torch.ones(n_edges, dtype=torch.float32)

    # Add forward edges (user -> movie)
    u_m_edge_index = torch.cat(
        [
            graph["user", "rates", "movie"].edge_index,
            torch.stack([src_nodes, liked_movie_ids]),
        ],
        dim=1,
    )
    u_m_edge_label = torch.cat([graph["user", "rates", "movie"].edge_label, edge_label])
    graph["user", "rates", "movie"].edge_index = u_m_edge_index
    graph["user", "rates", "movie"].edge_label = u_m_edge_label

    # Add reverse edges (movie -> user)
    m_u_edge_index = torch.cat(
        [
            graph["movie", "rev_rates", "user"].edge_index,
            torch.stack([liked_movie_ids, src_nodes]),
        ],
        dim=1,
    )
    m_u_edge_label = torch.cat(
        [graph["movie", "rev_rates", "user"].edge_label, edge_label]
    )
    graph["movie", "rev_rates", "user"].edge_index = m_u_edge_index
    graph["movie", "rev_rates", "user"].edge_label = m_u_edge_label

    # Expand the user embedding layer in the model
    with torch.no_grad():
        old_emb = model.user_emb.weight.data
        new_emb = nn.Embedding(new_user_id + 1, config.EMB_ID, device=device)
        new_emb.weight.data[:-1] = old_emb
        new_emb.weight.data[-1].zero_()  # Initialize new user embedding to zeros
        model.user_emb = new_emb

    return new_user_id


@torch.no_grad()
def get_recommendations(model, graph, user_id, movies_to_exclude, k):
    """Generates top-k recommendations for a given user."""
    device = next(model.parameters()).device
    model.eval()

    # Get embeddings for all nodes
    emb, ptr = model(graph.to(device))

    user_emb = emb[ptr[0] + user_id]
    movie_embs = emb[ptr[1] : ptr[1] + graph["movie"].num_nodes]

    # Predict scores (dot product)
    scores = movie_embs @ user_emb + model.bias

    # Mask out movies the user has already rated
    # Convert to tensor with correct dtype and device
    exclude_tensor = torch.tensor(movies_to_exclude, dtype=torch.long, device=device)
    scores[exclude_tensor] = -float("inf")

    # Get top K results
    top_k_scores, top_k_indices = torch.topk(scores, k)

    # Denormalize scores back to the 0-5 rating scale
    top_k_scores_real = (
        top_k_scores.cpu() * (config.RATING_MAX - config.RATING_MIN) + config.RATING_MIN
    ).clamp(0, 5)

    return top_k_indices, top_k_scores_real


def load_model_and_data():
    """Load the trained model and necessary data"""
    global model, graph, movies_df, device

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Data and Rebuild Graph Structure
        print("Loading dataset and building graph structure...")
        root = download_movielens20m(Path("dataset"))
        ratings = pd.read_csv(root / "rating.csv")
        movies = pd.read_csv(root / "movie.csv")
        tags = pd.read_csv(root / "tag.csv")

        graph, n_users, n_items, n_genres, n_tags = build_graph20m(
            ratings, movies, tags
        )

        movies_df = movies.copy()
        movies_df["movieId_new"] = pd.Categorical(movies_df["movieId"]).codes

        # Load the trained model
        model_path = "best_model.pt"
        print(f"Loading pre-trained model from '{model_path}'...")

        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # New format with hyperparameters
                heads = checkpoint.get("heads", 4)
                hidden = checkpoint.get("hidden", 64)
                model = GATRec(
                    n_users, n_items, n_genres, n_tags, heads=heads, hidden=hidden
                ).to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # fallback to old format
                model = GATRec(
                    n_users, n_items, n_genres, n_tags, heads=8, hidden=32
                ).to(device)
                model.load_state_dict(checkpoint)
        except FileNotFoundError:
            raise Exception(
                f"Model file not found at '{model_path}'. Please train the model first."
            )

        print("Model and data loaded successfully!")
        print(f"Loaded {len(movies_df)} movies")
        print(f"Graph structure: {graph}")

    except Exception as e:
        print(f"Error loading model and data: {e}")
        raise


def get_recommendations_for_user(user_ratings, top_k=10):
    """
    Get recommendations based on user ratings
    user_ratings: list of dicts with 'title' and 'rating' keys
    """
    global model, graph, movies_df, device

    # Create a fresh copy of the graph for this user
    import copy

    user_graph = copy.deepcopy(graph)
    user_model = copy.deepcopy(model)

    # Extract movie titles from user ratings (we'll treat all as "liked" movies)
    liked_titles = [rating_data["title"] for rating_data in user_ratings]

    print(f"Looking for movies: {liked_titles}")

    # Find movie IDs
    liked_movie_ids, not_found_titles = find_movie_ids(liked_titles, movies_df)

    if not liked_movie_ids:
        print("No matching movies found in dataset")
        return []

    if not_found_titles:
        print(f"Could not find these movies: {not_found_titles}")

    print(f"Using {len(liked_movie_ids)} movies for recommendations")

    # Add new user to graph
    new_user_id = add_new_user_to_graph(user_graph, user_model, liked_movie_ids, device)

    # Get recommendations
    rec_indices, rec_scores = get_recommendations(
        user_model, user_graph, new_user_id, movies_to_exclude=liked_movie_ids, k=top_k
    )

    # Format recommendations
    recommendations = []
    for idx, score in zip(rec_indices.cpu().numpy(), rec_scores):
        movie_info = movies_df[movies_df["movieId_new"] == idx].iloc[0]
        recommendations.append(
            {
                "movieId": int(movie_info["movieId"]),
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "predicted_rating": float(score),
                "tmdbId": None,  # Not available in this dataset format
            }
        )

    print(f"Top {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations):
        print(f"  {i + 1}. {rec['title']} - {rec['predicted_rating']:.2f}")

    return recommendations


@app.route("/recommend", methods=["POST"])
def recommend():
    """API endpoint to get movie recommendations"""
    try:
        data = request.json
        user_ratings = data.get("ratings", [])

        if not user_ratings:
            return jsonify({"error": "No ratings provided"}), 400

        print(f"Received {len(user_ratings)} user ratings")
        for rating in user_ratings:
            print(f"  {rating['title']}: {rating['rating']}")

        recommendations = get_recommendations_for_user(user_ratings, top_k=10)

        return jsonify({"success": True, "recommendations": recommendations})

    except Exception as e:
        print(f"Error in recommend endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return "OK", 200


if __name__ == "__main__":
    try:
        load_model_and_data()
        print("Starting Flask server...")
        app.run(debug=True, host="0.0.0.0", port=5001)
    except Exception as e:
        print(f"Failed to start server: {e}")
        import traceback

        traceback.print_exc()
