"""Build the full 244M-node MAG240M feature table (papers + authors + institutions).

MAG240M ships real 768-dim features for papers only. The standard OGB recipe
for the other node types is:
  author feature      = mean of the features of the papers the author wrote
  institution feature = mean of the features of its affiliated authors

Output node order (OGB convention): papers [0, P), authors [P, P+A),
institutions [P+A, P+A+I). float16, so 244,160,499 x 768 x 2 B = 375 GB.

    python build_full_feat.py --root /share/gupta/ak2877/gnn/data \
        --out /share/gupta/ak2877/gnn/data/full_feat.npy

Memory: the paper table is loaded into RAM (187 GB) so the per-author gather
is random RAM reads rather than random NFS reads. Author sums are computed in
float32 in chunks of --chunk authors (default 2M -> ~6 GB accumulator).
Writes go to a .npy memmap on the share, sequentially.
"""
import argparse
import json
import os
import time

import numpy as np


def mean_by_group(feat_src, src_ids, dst_ids, n_dst, out, out_off, chunk, label):
    """out[out_off + d] = mean(feat_src[src_ids where dst_ids == d]) for d in [0, n_dst).

    src_ids / dst_ids: edge list (same length). Edges are sorted by dst, then
    processed in chunks of `chunk` dst ids. Groups with no edges get zeros.
    """
    t0 = time.time()
    order = np.argsort(dst_ids, kind="stable")
    src_ids, dst_ids = src_ids[order], dst_ids[order]
    del order
    # boundaries: first edge index of each dst id
    starts = np.searchsorted(dst_ids, np.arange(n_dst + 1))
    dim = feat_src.shape[1]
    done = 0
    for d0 in range(0, n_dst, chunk):
        d1 = min(d0 + chunk, n_dst)
        e0, e1 = starts[d0], starts[d1]
        if e1 > e0:
            rows = feat_src[src_ids[e0:e1]].astype(np.float32)          # gather
            cnt = (starts[d0 + 1:d1 + 1] - starts[d0:d1])
            ne = cnt > 0                                                # non-empty groups
            idx = starts[d0:d1][ne] - e0                                # strictly increasing, < len(rows)
            acc = np.zeros((d1 - d0, dim), dtype=np.float32)
            acc[ne] = np.add.reduceat(rows, idx, axis=0) / cnt[ne, None].astype(np.float32)
            out[out_off + d0:out_off + d1] = acc.astype(np.float16)
            del rows, acc
        else:
            out[out_off + d0:out_off + d1] = 0
        done = d1
        if (d0 // chunk) % 5 == 0:
            print(f"[{label}] {done:,}/{n_dst:,}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{label}] done ({time.time()-t0:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/share/gupta/ak2877/gnn/data")
    p.add_argument("--out", default="/share/gupta/ak2877/gnn/data/full_feat.npy")
    p.add_argument("--chunk", type=int, default=2_000_000)
    p.add_argument("--memmap-papers", action="store_true",
                   help="do not load the 187 GB paper table into RAM (much slower)")
    p.add_argument("--smoke", action="store_true",
                   help="sanity run on a slice: 1M papers, 200k authors, memmapped, "
                        "writes <out> as a small file; takes minutes")
    args = p.parse_args()


    from ogb.lsc import MAG240MDataset
    ds = MAG240MDataset(root=args.root)
    P, A, I = ds.num_papers, ds.num_authors, ds.num_institutions
    if args.smoke:
        P, A = 1_000_000, 200_000
        print("[smoke] slice: 1M papers, 200k authors, all institutions", flush=True)
    N = P + A + I
    print(f"[meta] papers {P:,}  authors {A:,}  institutions {I:,}  total {N:,}", flush=True)

    t0 = time.time()
    paper_path = os.path.join(args.root, "mag240m_kddcup2021", "processed", "paper", "node_feat.npy")
    if args.smoke:
        # only the slice, in RAM: random reads from an NFS memmap are ~30 ms each
        paper_feat = np.ascontiguousarray(np.load(paper_path, mmap_mode="r")[:P])
    elif args.memmap_papers:
        paper_feat = np.load(paper_path, mmap_mode="r")
    else:
        print("[load] reading 187 GB paper table into RAM ...", flush=True)
        paper_feat = np.load(paper_path)
    print(f"[load] paper features {paper_feat.shape} {paper_feat.dtype} ({(time.time()-t0)/60:.1f} min)", flush=True)
    dim = paper_feat.shape[1]

    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float16, shape=(N, dim))
    print(f"[out] {args.out}  shape {out.shape}  {N*dim*2/1e9:.0f} GB", flush=True)

    # 1. papers: straight copy, sequential
    t0 = time.time()
    step = 4_000_000
    for i in range(0, P, step):
        j = min(i + step, P)
        out[i:j] = paper_feat[i:j]
        if (i // step) % 5 == 0:
            print(f"[papers] {min(i+step,P):,}/{P:,}  ({time.time()-t0:.0f}s)", flush=True)
    out.flush()
    print(f"[papers] done ({(time.time()-t0)/60:.1f} min)", flush=True)

    # 2. authors: mean of written papers
    ei = ds.edge_index("author", "writes", "paper")     # (2, E): row 0 author, row 1 paper
    if args.smoke:
        ei = ei[:, (ei[0] < A) & (ei[1] < P)]
    print(f"[edges] author-writes-paper {ei.shape[1]:,}", flush=True)
    mean_by_group(paper_feat, ei[1], ei[0], A, out, P, args.chunk, "authors")
    del ei
    out.flush()

    # 3. institutions: mean of affiliated authors (read author rows back from out)
    ei = ds.edge_index("author", "institution")          # (2, E): row 0 author, row 1 institution
    if args.smoke:
        ei = ei[:, ei[0] < A]
    print(f"[edges] author-institution {ei.shape[1]:,}", flush=True)
    author_feat = out[P:P + A]                             # memmap view, random reads from the share
    mean_by_group(author_feat, ei[1], ei[0], I, out, P + A, args.chunk, "institutions")
    del ei
    out.flush()

    # spot check: a few author rows against a direct recompute
    chk_ei = ds.edge_index("author", "writes", "paper")
    for a in (0, 1, 7):
        m = chk_ei[0] == a
        pp = chk_ei[1][m]
        if args.smoke:
            pp = pp[pp < P]
        if pp.size:
            ref = paper_feat[pp].astype(np.float32).mean(0)
            got = out[P + a].astype(np.float32)
            print(f"[check] author {a}: {pp.size} papers, max abs diff {np.abs(ref-got).max():.4f}", flush=True)
    meta = dict(num_papers=P, num_authors=A, num_institutions=I, num_nodes=N, dim=dim,
                order="papers, authors, institutions", dtype="float16",
                author="mean of written papers", institution="mean of affiliated authors",
                source=paper_path, out=args.out)
    with open(args.out.replace(".npy", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[done]", json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
