"""Build smaller MAG240M graphs for the latency-vs-capacity plot (paper Fig 3 style).

Subset n = the union of METIS k=16 shards 0 .. n-1, or, with --order, the first n shards of the
order in that JSON file (order_shards.py writes order_density_k16.json). Each subset is a real, smaller graph:
  * nodes: only the nodes whose METIS part is one of the subset's shards (papers, authors, institutions)
  * edges: only edges with BOTH ends inside the subset (every one of the 5 relations)
  * ids:   renumbered 0.. per node type, keeping the original order
  * features: the subset's rows of full_feat.npy, in the same order, as its own .npy

Output folder per subset (same layout the existing scripts read, so nothing else changes):
    <out>/s<n>/csr/rowptr.npy, col.npy                paper <- paper (cites)
    <out>/s<n>/csr_hetero/meta.json                   node counts, feature offsets, feat path
    <out>/s<n>/csr_hetero/{p2a,a2p,a2i,i2a}_{rowptr,col}.npy
    <out>/s<n>/full_feat.npy                          (rows x 768, float16)
    <out>/s<n>/metis/part_k1.npy                      all zeros: lets shard_infer.py hold the whole subset on the GPU
    <out>/s<n>/subset_info.json                       counts, edges kept, GB, checks

The feature table is read ONCE, sequentially, in chunks, and every subset is written in that
same pass, so it runs from a memmapped file on disk at sequential-read speed.

    python build_subset.py --csr /unicorn/gnn/csr --hetero-dir /unicorn/gnn/csr_hetero \
        --feat /unicorn/gnn/data/full_feat.npy --metis /unicorn/gnn/metis --k 16 \
        --sizes 1,2,3,4,8 --out /unicorn/gnn/subsets

Density order (4 Nov), folders d<n> in a new place, old subsets untouched:
    python build_subset.py ... --order results_capacity_order/order_density_k16.json --name-prefix d \
        --sizes 1,2,3,4,6,8,12 --out /unicorn/gnn/subsets_density

Memory: one relation at a time; the largest (cites) needs about 4 arrays of its edge count in
int64 plus masks. CPU only; no GPU needed.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

# relation -> (csc colptr file, csc row file, dst type, src type); same table as hetero_common._CSC_FILES
RELATIONS = [
    ("cites",          ("rowptr.npy", "col.npy"),         "paper",       "paper",       "csr"),
    ("writes",         ("p2a_rowptr.npy", "p2a_col.npy"), "paper",       "author",      "csr_hetero"),
    ("rev_writes",     ("a2p_rowptr.npy", "a2p_col.npy"), "author",      "paper",       "csr_hetero"),
    ("rev_affiliated", ("a2i_rowptr.npy", "a2i_col.npy"), "author",      "institution", "csr_hetero"),
    ("affiliated",     ("i2a_rowptr.npy", "i2a_col.npy"), "institution", "author",      "csr_hetero"),
]
TYPES = ["paper", "author", "institution"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csr", required=True)
    ap.add_argument("--hetero-dir", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--metis", required=True, help="directory with part_k{k}.npy")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--sizes", default="1,2,3,4,8", help="number of shards in each subset (shards 0..n-1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-rows", type=int, default=2_600_000, help="feature rows per read (2.6M x 1.5 KB = 4 GB)")
    ap.add_argument("--order", default="", help="JSON with 'order': subset n = its first n shards (default: shards 0..n-1)")
    ap.add_argument("--name-prefix", default="s", help="subset folder name = prefix + n (s1, s2 ... or d1, d2 ...)")
    a = ap.parse_args()
    sizes = sorted(int(s) for s in a.sizes.split(","))
    t_all = time.time()
    if a.order:
        with open(a.order) as f:
            order = [int(p) for p in json.load(f)["order"]]
        assert sorted(order) == list(range(a.k)), f"{a.order}: order is not a permutation of 0..{a.k - 1}"
    else:
        order = list(range(a.k))
    print(f"[order] {order}" + (f" from {a.order}" if a.order else " (default 0..k-1)"), flush=True)
    name = lambda n: f"{a.name_prefix}{n}"  # noqa: E731

    with open(os.path.join(a.hetero_dir, "meta.json")) as f:
        meta = json.load(f)
    num = {"paper": meta["num_papers"], "author": meta["num_authors"], "institution": meta["num_institutions"]}
    off = meta["offset"]
    N = sum(num.values())
    part = np.load(os.path.join(a.metis, f"part_k{a.k}.npy"))
    assert part.size == N, f"part has {part.size} entries, graph has {N}"
    print(f"[load] {N:,} nodes, part_k{a.k} md5 {md5(os.path.join(a.metis, f'part_k{a.k}.npy'))}", flush=True)

    # ---- node maps per subset and type: keep mask and new local id (-1 if dropped)
    subs = {}
    for n in sizes:
        keep_g = np.isin(part, order[:n])
        d = {"keep_g": keep_g, "num": {}, "newid": {}, "keep": {}}
        for t in TYPES:
            k = keep_g[off[t]: off[t] + num[t]]
            nid = np.full(num[t], -1, dtype=np.int64)
            nid[k] = np.arange(int(k.sum()), dtype=np.int64)
            d["keep"][t], d["newid"][t], d["num"][t] = k, nid, int(k.sum())
        subs[n] = d
        out = os.path.join(os.path.abspath(a.out), name(n))
        for sub in ("csr", "csr_hetero", "metis"):
            os.makedirs(os.path.join(out, sub), exist_ok=True)
        print(f"[{name(n)}] shards {order[:n]}: " + ", ".join(f"{t} {d['num'][t]:,}" for t in TYPES)
              + f" = {sum(d['num'].values()):,} nodes", flush=True)

    # ---- edges: one relation at a time, every subset
    edges = {n: {} for n in sizes}
    for rel, (fp, fr), dst_t, src_t, folder in RELATIONS:
        t0 = time.time()
        src_dir = a.csr if folder == "csr" else a.hetero_dir
        colptr = np.load(os.path.join(src_dir, fp))
        row = np.load(os.path.join(src_dir, fr))
        assert colptr.size == num[dst_t] + 1 and colptr[-1] == row.size, f"{rel}: bad CSC shape"
        dst = np.repeat(np.arange(num[dst_t], dtype=np.int64), np.diff(colptr))   # destination of every edge
        for n in sizes:
            d = subs[n]
            m = d["keep"][dst_t][dst] & d["keep"][src_t][row]          # both ends inside the subset
            new_row = d["newid"][src_t][row[m]]
            deg = np.bincount(d["newid"][dst_t][dst[m]], minlength=d["num"][dst_t]).astype(np.int64)
            new_ptr = np.zeros(d["num"][dst_t] + 1, dtype=np.int64)
            np.cumsum(deg, out=new_ptr[1:])
            outdir = os.path.join(a.out, name(n), folder)
            np.save(os.path.join(outdir, fp), new_ptr)
            np.save(os.path.join(outdir, fr), new_row)
            edges[n][rel] = int(new_row.size)
            # check: 3 kept destinations with neighbours; their new neighbour list maps back to the
            # original neighbours that are inside the subset
            rng = np.random.default_rng(1)
            cand = np.flatnonzero(d["keep"][dst_t] & (np.diff(colptr) > 0))
            for v in rng.choice(cand, size=min(3, cand.size), replace=False):
                orig = row[colptr[v]:colptr[v + 1]]
                want = d["newid"][src_t][orig[d["keep"][src_t][orig]]]
                nv = d["newid"][dst_t][v]
                got = new_row[new_ptr[nv]:new_ptr[nv + 1]]
                assert np.array_equal(np.sort(got), np.sort(want)), f"{rel} {name(n)}: neighbour check failed at {v}"
            del m, new_row, deg, new_ptr
        print(f"[{rel}] {row.size:,} edges -> " + ", ".join(f"{name(n)} {edges[n][rel]:,}" for n in sizes)
              + f"  (checks ok, {time.time() - t0:.0f}s)", flush=True)
        del colptr, row, dst

    # ---- features: one sequential pass over the full table, written into every subset
    t0 = time.time()
    feat = np.load(a.feat, mmap_mode="r")
    assert feat.shape[0] == N, f"feature table has {feat.shape[0]} rows, graph has {N}"
    outs, pos = {}, {}
    for n in sizes:
        rows_n = int(subs[n]["keep_g"].sum())
        outs[n] = np.lib.format.open_memmap(os.path.join(a.out, name(n), "full_feat.npy"), mode="w+",
                                            dtype=feat.dtype, shape=(rows_n, feat.shape[1]))
        pos[n] = 0
    for i in range(0, N, a.chunk_rows):
        j = min(i + a.chunk_rows, N)
        block = np.asarray(feat[i:j])                       # sequential read of this chunk
        for n in sizes:
            k = subs[n]["keep_g"][i:j]
            c = int(k.sum())
            outs[n][pos[n]:pos[n] + c] = block[k]
            pos[n] += c
        if (i // a.chunk_rows) % 10 == 0:
            print(f"[feat] {j:,} / {N:,} rows ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(2)
    for n in sizes:
        assert pos[n] == outs[n].shape[0]
        outs[n].flush()
        gids = np.flatnonzero(subs[n]["keep_g"])
        for r in rng.choice(gids.size, size=5, replace=False):        # check: 5 rows equal the original
            assert np.array_equal(outs[n][r], feat[gids[r]]), f"{name(n)}: feature row check failed"
    print(f"[feat] done, row checks ok ({time.time() - t0:.0f}s)", flush=True)

    # ---- meta.json, part_k1.npy, subset_info.json
    for n in sizes:
        d, out = subs[n], os.path.join(os.path.abspath(a.out), name(n))
        m2 = dict(meta)
        m2.update(num_papers=d["num"]["paper"], num_authors=d["num"]["author"], num_institutions=d["num"]["institution"],
                  offset=dict(paper=0, author=d["num"]["paper"], institution=d["num"]["paper"] + d["num"]["author"]),
                  feat=os.path.join(out, "full_feat.npy"))
        with open(os.path.join(out, "csr_hetero", "meta.json"), "w") as f:
            json.dump(m2, f, indent=1)
        np.save(os.path.join(out, "metis", "part_k1.npy"), np.zeros(sum(d["num"].values()), dtype=np.uint8))
        info = dict(shards=order[:n], order_file=a.order or None, k=a.k, nodes=sum(d["num"].values()), **{f"num_{t}": d["num"][t] for t in TYPES},
                    edges=edges[n], feat_gb=outs[n].nbytes / 1e9,
                    md5_meta=md5(os.path.join(out, "csr_hetero", "meta.json")),
                    md5_cites_rowptr=md5(os.path.join(out, "csr", "rowptr.npy")))
        with open(os.path.join(out, "subset_info.json"), "w") as f:
            json.dump(info, f, indent=1)
        print(f"[{name(n)}] {info['nodes']:,} nodes, {sum(edges[n].values()):,} edges, {info['feat_gb']:.1f} GB features; "
              f"cites rowptr md5 {info['md5_cites_rowptr']}", flush=True)
    print(f"[done] {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
