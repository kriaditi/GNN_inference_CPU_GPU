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
is random RAM reads rather than random NFS reads. Author means are computed
in float32 in chunks of --chunk authors. Institution sums are accumulated
during the author pass (25k x 768 fp32, 79 MB), so author rows are never
read back. Output is written sequentially with plain file writes and each
flushed chunk is dropped from the page cache, because dirty page cache counts
against the Slurm memory limit (job 277867 died that way with a memmap).
"""
import argparse
import json
import os
import time

import numpy as np


class NpyWriter:
    """Sequential .npy writer that keeps the page cache from piling up."""

    def __init__(self, path, shape, dtype):
        self.f = open(path, "wb")
        np.lib.format.write_array_header_1_0(
            self.f, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                     "fortran_order": False, "shape": shape})
        self.rows = 0
        self.fd = self.f.fileno()

    def write(self, arr):
        self.f.write(np.ascontiguousarray(arr).tobytes())
        self.rows += arr.shape[0]
        self.f.flush()
        try:
            os.fsync(self.fd)
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass

    def close(self):
        self.f.close()


def author_pass(paper_feat, ei_ap, A, chunk, writer, inst_edges, I, label="authors"):
    """Write author means (chunked); accumulate institution sums on the way.

    ei_ap:      (2, E) author-writes-paper, row 0 author, row 1 paper
    inst_edges: (2, E2) author-institution, row 0 author, row 1 institution
    Returns (inst_sum fp32 [I, dim], inst_cnt int64 [I]).
    """
    t0 = time.time()
    dim = paper_feat.shape[1]
    order = np.argsort(ei_ap[0], kind="stable")
    a_sorted, p_sorted = ei_ap[0][order], ei_ap[1][order]
    del order
    starts = np.searchsorted(a_sorted, np.arange(A + 1))
    del a_sorted

    io = np.argsort(inst_edges[0], kind="stable")
    ia, ii = inst_edges[0][io], inst_edges[1][io]
    del io
    istarts = np.searchsorted(ia, np.arange(A + 1))
    del ia
    inst_sum = np.zeros((I, dim), dtype=np.float32)
    inst_cnt = np.zeros(I, dtype=np.int64)

    for d0 in range(0, A, chunk):
        d1 = min(d0 + chunk, A)
        e0, e1 = starts[d0], starts[d1]
        acc = np.zeros((d1 - d0, dim), dtype=np.float32)
        if e1 > e0:
            rows = paper_feat[p_sorted[e0:e1]].astype(np.float32)
            cnt = starts[d0 + 1:d1 + 1] - starts[d0:d1]
            ne = cnt > 0
            idx = starts[d0:d1][ne] - e0
            acc[ne] = np.add.reduceat(rows, idx, axis=0) / cnt[ne, None].astype(np.float32)
            del rows
        # institutions: add each author's mean to every institution it belongs to
        f0, f1 = istarts[d0], istarts[d1]
        if f1 > f0:
            a_local = np.repeat(np.arange(d0, d1), istarts[d0 + 1:d1 + 1] - istarts[d0:d1]) - d0
            np.add.at(inst_sum, ii[f0:f1], acc[a_local])
            np.add.at(inst_cnt, ii[f0:f1], 1)
        writer.write(acc.astype(np.float16))
        del acc
        if (d0 // chunk) % 5 == 0:
            print(f"[{label}] {d1:,}/{A:,}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{label}] done ({(time.time()-t0)/60:.1f} min)", flush=True)
    return inst_sum, inst_cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/share/gupta/ak2877/gnn/data")
    p.add_argument("--out", default="/share/gupta/ak2877/gnn/data/full_feat.npy")
    p.add_argument("--chunk", type=int, default=2_000_000)
    p.add_argument("--memmap-papers", action="store_true",
                   help="do not load the 187 GB paper table into RAM (much slower)")
    p.add_argument("--smoke", action="store_true",
                   help="sanity run on a slice: 1M papers, 200k authors, all institutions")
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
        paper_feat = np.ascontiguousarray(np.load(paper_path, mmap_mode="r")[:P])
    elif args.memmap_papers:
        paper_feat = np.load(paper_path, mmap_mode="r")
    else:
        print("[load] reading 187 GB paper table into RAM ...", flush=True)
        paper_feat = np.load(paper_path)
    print(f"[load] paper features {paper_feat.shape} {paper_feat.dtype} ({(time.time()-t0)/60:.1f} min)", flush=True)
    dim = paper_feat.shape[1]

    w = NpyWriter(args.out, (N, dim), np.float16)
    print(f"[out] {args.out}  shape {(N, dim)}  {N*dim*2/1e9:.0f} GB", flush=True)

    # 1. papers: sequential copy
    t0 = time.time()
    step = 2_000_000
    for i in range(0, P, step):
        j = min(i + step, P)
        w.write(paper_feat[i:j])
        if (i // step) % 10 == 0:
            print(f"[papers] {j:,}/{P:,}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[papers] done ({(time.time()-t0)/60:.1f} min)", flush=True)

    # 2. authors (+ institution accumulation)
    ei = ds.edge_index("author", "writes", "paper")
    if args.smoke:
        ei = ei[:, (ei[0] < A) & (ei[1] < P)]
    print(f"[edges] author-writes-paper {ei.shape[1]:,}", flush=True)
    ei_inst = ds.edge_index("author", "institution")
    if args.smoke:
        ei_inst = ei_inst[:, ei_inst[0] < A]
    print(f"[edges] author-institution {ei_inst.shape[1]:,}", flush=True)
    inst_sum, inst_cnt = author_pass(paper_feat, ei, A, args.chunk, w, ei_inst, I)
    del ei, ei_inst

    # 3. institutions
    cnt = np.maximum(inst_cnt, 1).astype(np.float32)
    inst = (inst_sum / cnt[:, None]).astype(np.float16)
    inst[inst_cnt == 0] = 0
    w.write(inst)
    print(f"[institutions] {I:,} written, {(inst_cnt == 0).sum():,} with no authors", flush=True)
    w.close()
    assert w.rows == N, f"wrote {w.rows} rows, expected {N}"

    # spot check a few author rows against a direct recompute
    out = np.load(args.out, mmap_mode="r")
    chk = ds.edge_index("author", "writes", "paper")
    shown = 0
    for a in range(A):
        pp = chk[1][chk[0] == a]
        if args.smoke:
            pp = pp[pp < P]
        if pp.size:
            ref = paper_feat[pp].astype(np.float32).mean(0)
            got = out[P + a].astype(np.float32)
            print(f"[check] author {a}: {pp.size} papers, max abs diff {np.abs(ref-got).max():.4f}", flush=True)
            shown += 1
        if shown == 3:
            break

    meta = dict(num_papers=P, num_authors=A, num_institutions=I, num_nodes=N, dim=dim,
                order="papers, authors, institutions", dtype="float16",
                author="mean of written papers", institution="mean of affiliated authors",
                source=paper_path, out=args.out)
    with open(args.out.replace(".npy", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[done]", json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
