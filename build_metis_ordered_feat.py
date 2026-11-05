"""Reorder the feature table by METIS part, so each part is one contiguous block (for naive GPU, 4 Nov).

Naive GPU (Michael, 4 Nov) = streaming with the METIS parts as the chunks: every batch, every part goes
to the GPU in turn and the GPU keeps the rows the batch needs. stream_infer.py streams contiguous slices,
so the table is written once in part order (untimed, offline):

  order   = global ids sorted by part (stable: original order inside a part)
  rowmap  = global id -> row in the reordered table   (rowmap[order] = 0..N-1)
  bounds  = first row of each part, plus N              (part p = rows bounds[p] .. bounds[p+1]-1)

Outputs in --out: full_feat_metis_k<k>.npy (same size as the original, 375 GB), rowmap_k<k>.npy,
bounds_k<k>.npy, info_k<k>.json. Check: 100,000 random global ids, reordered[rowmap[g]] == original[g].

    python build_metis_ordered_feat.py --feat /unicorn/gnn/data/full_feat.npy \
        --metis /unicorn/gnn/metis --k 5 --out /unicorn/gnn/metis_ordered
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--metis", required=True, help="directory with part_k{k}.npy")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-rows", type=int, default=2_000_000, help="rows per read (2M x 1.5 KB = 3 GB)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_all = time.time()

    part_path = os.path.join(a.metis, f"part_k{a.k}.npy")
    part = np.load(part_path).astype(np.int64)
    feat = np.load(a.feat, mmap_mode="r")
    N, D = feat.shape
    assert part.size == N, f"part has {part.size} entries, table has {N} rows"
    counts = np.bincount(part, minlength=a.k)
    bounds = np.zeros(a.k + 1, dtype=np.int64)
    np.cumsum(counts, out=bounds[1:])
    order = np.argsort(part, kind="stable")
    rowmap = np.empty(N, dtype=np.int64)
    rowmap[order] = np.arange(N, dtype=np.int64)
    for p in range(a.k):
        print(f"[part {p}] {counts[p]:,} rows, {counts[p] * D * 2 / 1e9:.1f} GB, rows {bounds[p]:,} .. {bounds[p + 1] - 1:,}", flush=True)

    out_path = os.path.join(a.out, f"full_feat_metis_k{a.k}.npy")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=feat.dtype, shape=(N, D))
    t0 = time.time()
    for i in range(0, N, a.block_rows):            # order is sorted inside each part, so reads move forward
        j = min(i + a.block_rows, N)
        out[i:j] = feat[order[i:j]]
        if (i // a.block_rows) % 10 == 0:
            print(f"[write] {j:,} / {N:,} rows ({time.time() - t0:.0f}s)", flush=True)
    out.flush()

    rng = np.random.default_rng(3)
    for g in rng.choice(N, size=min(100_000, N), replace=False):
        assert np.array_equal(out[rowmap[g]], feat[g]), f"row check failed at global id {g}"
    assert np.all(np.diff(part[order]) >= 0), "reordered parts are not contiguous"
    print("[check] 100,000 random rows equal the original; parts contiguous", flush=True)

    np.save(os.path.join(a.out, f"rowmap_k{a.k}.npy"), rowmap)
    np.save(os.path.join(a.out, f"bounds_k{a.k}.npy"), bounds)
    info = dict(k=a.k, rows=int(N), dim=int(D), part_rows=counts.tolist(), bounds=bounds.tolist(),
                part_gb=[round(c * D * 2 / 1e9, 2) for c in counts], md5_part=md5(part_path), feat=os.path.abspath(a.feat))
    with open(os.path.join(a.out, f"info_k{a.k}.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"[done] {time.time() - t_all:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
