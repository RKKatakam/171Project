import argparse
import torch
import random
import numpy as np


EMB_ID = 64
EMB_FEAT = 32
DROPOUT = 0.30
RATING_MIN, RATING_MAX = 0.0, 5.0
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def set_seed(seed_value=SEED):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--batch_size_edges", type=int, default=64_000)
    p.add_argument(
        "--sample_users", type=int, default=0, help="train on N random users (0 = all)"
    )
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience")
    p.add_argument(
        "--max_tags", type=int, default=800, help="top-k tags kept as features"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args() if "__file__" in globals() else p.parse_args([])
