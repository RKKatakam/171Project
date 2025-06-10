from pathlib import Path
import pandas as pd
import torch
from torch_geometric.loader import LinkNeighborLoader

import config
from data_processing import download_movielens20m, build_graph20m
from model import GATRec
import training


def main():
    # --- Setup ---
    args = config.get_args()
    config.set_seed()
    device = torch.device(args.device)

    # --- Data Loading and Processing ---
    root = download_movielens20m(Path("dataset"))
    ratings = pd.read_csv(root / "rating.csv")
    movies = pd.read_csv(root / "movie.csv")
    tags_df = pd.read_csv(root / "tag.csv")

    data, n_users, n_items, n_genres, n_tags = build_graph20m(
        ratings, movies, tags_df, sample_users=args.sample_users, max_tags=args.max_tags
    )

    # --- Model and Optimizer ---
    model = GATRec(n_users, n_items, n_genres, n_tags, args.heads, args.hidden).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    n_edges = data["user", "rates", "movie"].edge_label.numel()
    print(
        f"Training on {n_edges:,} edges ({n_users} users, {n_items} movies) on {device}"
    )

    # --- Training Setup ---
    use_neighbor_loader = device.type == "cuda" and n_edges > 5_000_000
    link_loader = None
    if use_neighbor_loader:
        print("Using LinkNeighborLoader for mini-batch training.")
        link_loader = LinkNeighborLoader(
            data,
            num_neighbors=[10, 10],
            batch_size=args.batch_size_edges,
            shuffle=True,
            edge_label_index=(
                ("user", "rates", "movie"),
                data["user", "rates", "movie"].edge_index,
            ),
            edge_label=data["user", "rates", "movie"].edge_label,
            num_workers=4,
            persistent_workers=True,
        )

    # --- Training Loop ---
    best_val_rmse = float("inf")
    patience_left = args.patience

    for ep in range(1, args.epochs + 1):
        if use_neighbor_loader:
            loss = training.train_epoch_minibatch(model, link_loader, opt)
        else:
            loss = training.train_epoch_fullgraph(
                model, data, opt, args.batch_size_edges, device
            )

        val_rmse = training.evaluate(model, data, data.val_edges, device)
        improved = val_rmse < best_val_rmse - 1e-4

        print(
            f"Epoch {ep:03d} | Train Loss: {loss:.4f} | Val RMSE: {val_rmse:.4f}"
            + (" (new best)" if improved else "")
        )

        if improved:
            best_val_rmse = val_rmse
            patience_left = args.patience
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "heads": 8,
                    "hidden": 32,
                    "n_users": n_users,
                    "n_items": n_items,
                    "n_genres": n_genres,
                    "n_tags": n_tags,
                },
                "best_model.pt",
            )
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping due to no improvement.")
                break

    # --- Final Evaluation ---
    print("\nLoading best model for final testing...")
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    test_rmse = training.evaluate(model, data, data.test_edges, device)
    print(f"✅ Final Test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    main()
