"""Shared pieces for heterogeneous MAG240M: graph loading, sampling, gather, model.

Used by train_hetero.py (training) and by the inference harness, so the
sampler and the model are defined and tested exactly once.

Node types: paper, author, institution. Features for all three live in one
table, full_feat.npy (244,160,499 x 768 fp16), in the order paper, author,
institution; csr_hetero/meta.json has the offsets.

Message-passing relations (edges point TOWARD the node being updated, which
is PyG's convention, so each relation is stored as CSC: colptr indexed by
the destination type, row holding source ids):

    paper       <- paper        'cites'           csr/rowptr, csr/col (symmetric)
    paper       <- author       'writes'          p2a_rowptr, p2a_col
    author      <- paper        'rev_writes'      a2p_rowptr, a2p_col
    author      <- institution  'rev_affiliated'  a2i_rowptr, a2i_col
    institution <- author       'affiliated'      i2a_rowptr, i2a_col

Model: R-GraphSAGE. One SAGEConv per relation per layer, summed per
destination type. Classification head on papers only (153 classes).
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

NODE_TYPES = ["paper", "author", "institution"]
EDGE_TYPES = [
    ("paper", "cites", "paper"),
    ("author", "writes", "paper"),
    ("paper", "rev_writes", "author"),
    ("institution", "rev_affiliated", "author"),
    ("author", "affiliated", "institution"),
]
# relation -> (csc colptr file, csc row file); None means the paper CSR dir
_CSC_FILES = {
    ("paper", "cites", "paper"): None,
    ("author", "writes", "paper"): ("p2a_rowptr.npy", "p2a_col.npy"),
    ("paper", "rev_writes", "author"): ("a2p_rowptr.npy", "a2p_col.npy"),
    ("institution", "rev_affiliated", "author"): ("a2i_rowptr.npy", "a2i_col.npy"),
    ("author", "affiliated", "institution"): ("i2a_rowptr.npy", "i2a_col.npy"),
}


class HeteroGraph:
    """CSC per relation (torch int64), node counts, feature offsets."""

    def __init__(self, csr_dir, hetero_dir):
        t0 = time.time()
        with open(os.path.join(hetero_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.num = {"paper": self.meta["num_papers"],
                    "author": self.meta["num_authors"],
                    "institution": self.meta["num_institutions"]}
        self.offset = self.meta["offset"]
        self.colptr, self.row = {}, {}
        for et, files in _CSC_FILES.items():
            if files is None:
                cp, rw = (os.path.join(csr_dir, "rowptr.npy"),
                          os.path.join(csr_dir, "col.npy"))
            else:
                cp, rw = (os.path.join(hetero_dir, files[0]),
                          os.path.join(hetero_dir, files[1]))
            self.colptr[et] = torch.from_numpy(np.load(cp)).long()
            self.row[et] = torch.from_numpy(np.load(rw)).long()
            assert self.colptr[et].numel() == self.num[et[2]] + 1, \
                f"{et}: colptr has {self.colptr[et].numel()} entries, " \
                f"expected {self.num[et[2]] + 1} ({et[2]} + 1)"
        edges = sum(v.numel() for v in self.row.values())
        print(f"[hetero] 5 relations, {edges:,} directed edges resident "
              f"({time.time()-t0:.0f}s)", flush=True)

    def neighbors(self, et, dst_local):
        """Source ids of one destination node under one relation (for checks)."""
        a, b = self.colptr[et][dst_local].item(), self.colptr[et][dst_local + 1].item()
        return self.row[et][a:b]


def load_features(feat_path, in_memory):
    t0 = time.time()
    if in_memory:
        print(f"[feat] reading {feat_path} into RAM ...", flush=True)
        feat = np.load(feat_path)
        print(f"[feat] RESIDENT {feat.shape} {feat.dtype} "
              f"({feat.nbytes/1e9:.0f} GB, {(time.time()-t0)/60:.1f} min)", flush=True)
    else:
        feat = np.load(feat_path, mmap_mode="r")
        print(f"[feat] MEMMAPPED {feat.shape} {feat.dtype}", flush=True)
    return feat


def sample_hetero(g, seeds, fanout):
    """Phase 1: hetero neighbour sample from paper seeds.

    fanout: list per hop, applied to every relation, e.g. [40, 20].
    Returns node_id (dict type -> global-within-type local ids, seeds first
    for 'paper') and edge_index (dict relation -> (2, E) local indices,
    row 0 = source position in node_id[src], row 1 = destination position
    in node_id[dst]).
    """
    import pyg_lib
    out = pyg_lib.sampler.hetero_neighbor_sample(
        g.colptr, g.row, {"paper": seeds},
        {et: list(fanout) for et in EDGE_TYPES},
        csc=True, replace=False, directed=True, disjoint=False,
        return_edge_id=False,
    )
    row_d, col_d, node_id = out[0], out[1], out[2]
    edge_index = {et: torch.stack([row_d[et], col_d[et]], dim=0)
                  for et in EDGE_TYPES if et in row_d and row_d[et].numel() > 0}
    for nt in NODE_TYPES:
        node_id.setdefault(nt, torch.empty(0, dtype=torch.long))
    return node_id, edge_index


def check_directions(g, seeds, node_id, edge_index):
    """The sampled neighbours of seed 0 must really be its neighbours.

    Guards the CSC direction convention: if it were wrong, the model would
    train on edges that do not exist and nothing downstream would notice.
    """
    s = int(seeds[0])
    assert int(node_id["paper"][0]) == s, "seed 0 is not first in node_id['paper']"
    for et in [("paper", "cites", "paper"), ("author", "writes", "paper")]:
        if et not in edge_index:
            continue
        ei = edge_index[et]
        src_pos = ei[0][ei[1] == 0]                 # edges into local paper 0
        got = node_id[et[0]][src_pos].numpy()
        real = set(g.neighbors(et, s).numpy().tolist())
        bad = [x for x in got if x not in real]
        assert not bad, f"direction check FAILED for {et}: {len(bad)} sampled " \
                        f"sources are not neighbours of seed {s}"
        print(f"[check] {et}: {len(got)} sampled sources of seed {s} all real "
              f"(seed has {len(real)})", flush=True)


def gather_features(feat, node_id, g):
    """Phase 2: one fancy-index per node type from the full table."""
    x = {}
    for nt in NODE_TYPES:
        ids = node_id[nt].numpy()
        if ids.size == 0:
            x[nt] = torch.empty(0, feat.shape[1], dtype=torch.float16)
            continue
        rows = ids + g.offset[nt]
        x[nt] = torch.from_numpy(np.ascontiguousarray(feat[rows]))
    return x


class RSAGE(torch.nn.Module):
    """R-GraphSAGE: per-relation SAGEConv, summed per destination type.

    All node types share the 768-dim input, so every conv is
    SAGEConv((in, in), out). Output logits only for papers.
    """

    def __init__(self, in_dim=768, hidden=256, out_dim=153, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        dims = [in_dim] + [hidden] * num_layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            layer = torch.nn.ModuleDict()
            for et in EDGE_TYPES:
                layer["__".join(et)] = SAGEConv((dims[i], dims[i]), dims[i + 1])
            self.convs.append(layer)
        self.head = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        # x: dict type -> (n_type, d) float32 ; edge_index: dict relation -> (2, E)
        for i, layer in enumerate(self.convs):
            out = {nt: None for nt in NODE_TYPES}
            for et, ei in edge_index.items():
                src, _, dst = et
                if x[src].size(0) == 0 or x[dst].size(0) == 0:
                    continue
                h = layer["__".join(et)]((x[src], x[dst]), ei,
                                         size=(x[src].size(0), x[dst].size(0)))
                out[dst] = h if out[dst] is None else out[dst] + h
            new_x = {}
            for nt in NODE_TYPES:
                if out[nt] is None:
                    # type with no incoming sampled edges this batch: keep a
                    # zero tensor of the right width so later layers index fine
                    new_x[nt] = x[nt].new_zeros((x[nt].size(0), self.convs[i]["paper__cites__paper"].out_channels))
                else:
                    new_x[nt] = F.relu(out[nt]) if i < self.num_layers - 1 else out[nt]
            x = new_x
        return self.head(x["paper"])
