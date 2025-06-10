import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_epoch_fullgraph(model, data, opt, batch_edges, device):
    model.train()
    data = data.to(device)
    src, dst = data["user", "rates", "movie"].edge_index
    labels = data["user", "rates", "movie"].edge_label
    perm = torch.randperm(labels.size(0))
    tot = 0.0
    for idx in tqdm(perm.split(batch_edges), desc="Batches"):
        # Compute embeddings fresh for each batch
        emb, ptr = model(data)
        loss = F.mse_loss(model.score(emb, ptr, src[idx], dst[idx]), labels[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item() * len(idx)
    return tot / len(perm)


def train_epoch_minibatch(model, loader, opt):
    model.train()
    device = next(model.parameters()).device
    total_loss = total_items = 0

    for batch in tqdm(loader, desc="Mini-Batches"):
        batch = batch.to(device)
        emb, ptr = model(batch)
        src = batch["user", "rates", "movie"].edge_label_index[0]
        dst = batch["user", "rates", "movie"].edge_label_index[1]
        lbl = batch["user", "rates", "movie"].edge_label

        loss = F.mse_loss(model.score(emb, ptr, src, dst), lbl)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item() * len(lbl)
        total_items += len(lbl)

    return total_loss / total_items


@torch.no_grad()
def evaluate(model, data, edge_pack, device):
    """Evaluates the model and returns the RMSE."""
    model.eval()
    emb, ptr = model(data.to(device))

    u, m, y_true = edge_pack
    pred = model.score(emb, ptr, u.to(device), m.to(device))
    return torch.sqrt(F.mse_loss(pred, y_true.to(device))).item()
