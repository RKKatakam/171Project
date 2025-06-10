import argparse
import re
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn

# Import from our modules
import config
from data_processing import download_movielens20m, build_graph20m
from model import GATRec

def parse_inference_args():
    """Parses command-line arguments for inference."""
    p = argparse.ArgumentParser(description="Get movie recommendations.")
    p.add_argument(
        "--liked_movies",
        type=str,
        nargs='+',
        help="A list of movie titles you liked (e.g., 'Toy Story (1995)')."
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of recommendations to return."
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="best_model.pt",
        help="Path to the saved model state dictionary."
    )
    return p.parse_args()

def find_movie_ids(titles, movies_df):
    """
    Finds the internal 'movieId_new' for a list of movie titles.
    Uses case-insensitive search.
    """
    found_ids = []
    not_found_titles = []
    
    # Create a mapping from lowercase title to movieId_new for efficient lookup
    title_map = movies_df.set_index(movies_df['title'].str.lower())['movieId_new']
    
    for title in titles:
        lower_title = title.lower()
        # First, try for an exact match
        if lower_title in title_map.index:
            found_ids.append(title_map[lower_title])
            print(f"✅ Found exact match for: '{title}'")
        else:
            # If no exact match, try a partial match
            matches = movies_df[movies_df['title'].str.lower().str.contains(re.escape(lower_title), na=False)]
            if not matches.empty:
                match_id = matches.iloc[0]['movieId_new']
                match_title = matches.iloc[0]['title']
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
    u_m_edge_index = torch.cat([graph["user", "rates", "movie"].edge_index, torch.stack([src_nodes, liked_movie_ids])], dim=1)
    u_m_edge_label = torch.cat([graph["user", "rates", "movie"].edge_label, edge_label])
    graph["user", "rates", "movie"].edge_index = u_m_edge_index
    graph["user", "rates", "movie"].edge_label = u_m_edge_label
    
    # Add reverse edges (movie -> user)
    m_u_edge_index = torch.cat([graph["movie", "rev_rates", "user"].edge_index, torch.stack([liked_movie_ids, src_nodes])], dim=1)
    m_u_edge_label = torch.cat([graph["movie", "rev_rates", "user"].edge_label, edge_label])
    graph["movie", "rev_rates", "user"].edge_index = m_u_edge_index
    graph["movie", "rev_rates", "user"].edge_label = m_u_edge_label

    # Expand the user embedding layer in the model
    with torch.no_grad():
        old_emb = model.user_emb.weight.data
        new_emb = nn.Embedding(new_user_id + 1, config.EMB_ID, device=device)
        new_emb.weight.data[:-1] = old_emb
        new_emb.weight.data[-1].zero_() # Initialize new user embedding to zeros
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
    scores[exclude_tensor] = -float('inf')
    
    # Get top K results
    top_k_scores, top_k_indices = torch.topk(scores, k)
    
    # Denormalize scores back to the 0-5 rating scale
    top_k_scores_real = (top_k_scores.cpu() * (config.RATING_MAX - config.RATING_MIN) + config.RATING_MIN).clamp(0, 5)

    return top_k_indices, top_k_scores_real

def main():
   
    args = parse_inference_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Load Data and Rebuild Graph Structure ---
    print("Loading dataset and building graph structure...")
    root = download_movielens20m(Path("dataset"))
    ratings = pd.read_csv(root / "rating.csv")
    movies = pd.read_csv(root / "movie.csv")
    tags = pd.read_csv(root / "tag.csv")

    
    graph, n_users, n_items, n_genres, n_tags = build_graph20m(
        ratings, movies, tags
    )

    
    movies_with_ids = movies.copy()
    movies_with_ids["movieId_new"] = pd.Categorical(movies_with_ids["movieId"]).codes

    
    print(f"Loading pre-trained model from '{args.model_path}'...")

    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with hyperparameters
            heads = checkpoint.get('heads', 4)
            hidden = checkpoint.get('hidden', 64)
            model = GATRec(n_users, n_items, n_genres, n_tags, heads=heads, hidden=hidden).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else:
            # fallback to old format
            model = GATRec(n_users, n_items, n_genres, n_tags, heads=8, hidden=32).to(device)
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model_path}'.")
        print("Please run 'python main.py' first to train and save a model.")
        return
    
    # print the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded with {num_params} trainable parameters.")

    
    liked_movie_ids, _ = find_movie_ids(args.liked_movies, movies_with_ids)
    if not liked_movie_ids:
        print("\nCould not find any of the specified movies. Please check the titles and try again.")
        return
        
    
    print("\nAdding new user profile to the graph...")
    new_user_id = add_new_user_to_graph(graph, model, liked_movie_ids, device)
    
    print(f"Generating top {args.top_k} recommendations...")
    rec_indices, rec_scores = get_recommendations(
        model, graph, new_user_id, movies_to_exclude=liked_movie_ids, k=args.top_k
    )
  
    recommended_titles = movies_with_ids.set_index("movieId_new").loc[rec_indices.cpu().numpy()].title
    
    print("-" * 75)
    print(f"✨ Top {args.top_k} Movie Recommendations For You ✨")
    print("-" * 75)
    for title, score in zip(recommended_titles.values, rec_scores):
        print(f"  - {title:<60} | Predicted Rating: {score.item():.2f} ★")
    print("-" * 75)

if __name__ == "__main__":
    main()