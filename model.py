import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from config import EMB_ID, EMB_FEAT, DROPOUT


class GATRec(nn.Module):
    def __init__(self, n_users, n_items, n_genres, n_tags, heads, hidden):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, EMB_ID)
        self.item_emb = nn.Embedding(n_items, EMB_ID)

        self.genre_proj = nn.Sequential(
            nn.Linear(n_genres, EMB_FEAT), nn.ReLU(), nn.Linear(EMB_FEAT, EMB_FEAT)
        )
        self.tag_proj = nn.Sequential(
            nn.Linear(n_tags, EMB_FEAT), nn.ReLU(), nn.Linear(EMB_FEAT, EMB_FEAT)
        )

        # total movie feature dim
        n_in_movie = EMB_ID + EMB_FEAT + EMB_FEAT

        self.gat1 = GATConv(
            (-1, -1), hidden // heads, heads, dropout=DROPOUT, add_self_loops=False
        )
        self.gat2 = GATConv(
            (-1, -1),
            hidden,
            heads=1,
            concat=False,
            dropout=DROPOUT,
            add_self_loops=False,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _ids(nstore, device):
        # works for full-graph and for neighbor mini-batches
        return (
            nstore.n_id
            if "n_id" in nstore
            else torch.arange(nstore.num_nodes, device=device)
        )

    def forward(self, data):
        dev = self.bias.device
        usr_x_orig = data["user"].x if "x" in data["user"] else None
        mov_x_orig = data["movie"].x  # genre+tag matrix

        u_id = self._ids(data["user"], dev)
        m_id = self._ids(data["movie"], dev)

        data["user"].x = self.user_emb(u_id)
        g_feat, t_feat = mov_x_orig.split(
            [self.genre_proj[0].in_features, self.tag_proj[0].in_features], dim=1
        )
        data["movie"].x = torch.cat(
            [
                self.item_emb(m_id),
                self.genre_proj(g_feat.to(dev)),
                self.tag_proj(t_feat.to(dev)),
            ],
            dim=1,
        )

        homo = data.to_homogeneous(node_attrs=["x"]).to(dev)

        n_u = data["user"].x.size(0)
        n_m = data["movie"].x.size(0)
        ptr = torch.tensor([0, n_u, n_u + n_m], device=dev)

        out = F.elu(self.gat1(homo.x, homo.edge_index))
        out = self.gat2(out, homo.edge_index)


        # restore originals so they don’t accumulate across batches
        if usr_x_orig is None:
            del data["user"].x
        else:
            data["user"].x = usr_x_orig
        data["movie"].x = mov_x_orig
        return out, ptr

    def score(self, emb, ptr, u, m):
        u_off = ptr[0]  # 0
        m_off = ptr[1]  # first movie index
        return (emb[u_off + u] * emb[m_off + m]).sum(-1) + self.bias
