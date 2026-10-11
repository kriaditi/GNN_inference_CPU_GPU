"""Dump per-query sampled neighbourhoods: batch_id, query_id, node_id.

Same seed pool, batch size and fanout as the profiling runs, but sampled
with disjoint=True so every sampled node is attributed to the seed (query)
that pulled it in. Needs only the CSR, no features, no GPU.

    python dump_queries.py --batches 100 --out results/queries_b16384_f15x10

Writes
  <out>.csv.gz          batch_id,query_id,node_id   (long form, large)
  <out>_counts.csv      batch_id,query_id,nodes_touched

Note: sampling is random, so the exact node sets differ from any earlier
run; the distribution is the same.
"""
import argparse
import csv
import gzip
import os
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/share/gupta/ak2877/gnn/data")
    p.add_argument("--csr", default="/share/gupta/ak2877/gnn/csr")
    p.add_argument("--seeds", type=int, default=88092)
    p.add_argument("--seed-mode", default="random", choices=["random", "all"])
    p.add_argument("--fanout", default="15,10")
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--batches", type=int, default=100,
                   help="how many batches to dump (full run = 6000)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import pyg_lib
    fanout = [int(v) for v in args.fanout.split(",")]

    t0 = time.time()
    rowptr = torch.from_numpy(np.load(os.path.join(args.csr, "rowptr.npy"))).long()
    col = torch.from_numpy(np.load(os.path.join(args.csr, "col.npy"))).long()
    print(f"[graph] CSR loaded ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(0)
    if args.seed_mode == "all":
        pool = rng.choice(rowptr.numel() - 1, size=args.seeds, replace=False)
    else:
        from ogb.lsc import MAG240MDataset
        split = MAG240MDataset(root=args.root).get_idx_split("test-dev")
        pool = split if args.seeds >= len(split) else \
            rng.choice(split, size=args.seeds, replace=False)
    pool = torch.from_numpy(np.sort(np.asarray(pool))).long()
    batches = [pool[i:i + args.batch_size]
               for i in range(0, len(pool), args.batch_size)]
    print(f"[seeds] {len(pool):,} seeds, {len(batches)} distinct batches, "
          f"dumping {args.batches}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f_long = gzip.open(args.out + ".csv.gz", "wt", newline="")
    w_long = csv.writer(f_long)
    w_long.writerow(["batch_id", "query_id", "node_id"])
    f_cnt = open(args.out + "_counts.csv", "w", newline="")
    w_cnt = csv.writer(f_cnt)
    w_cnt.writerow(["batch_id", "query_id", "nodes_touched"])

    total_rows = 0
    for b in range(args.batches):
        seeds = batches[b % len(batches)]
        out = pyg_lib.sampler.neighbor_sample(
            rowptr, col, seeds, fanout,
            csc=False, replace=False, directed=True,
            disjoint=True, return_edge_id=False,
        )
        # pyg_lib 0.4 with disjoint=True returns out[2] as (N, 2):
        # column 0 = seed index within the batch, column 1 = global paper id
        pairs = out[2].numpy()
        batch_vec = pairs[:, 0]
        node_id = pairs[:, 1]
        query_id = seeds.numpy()[batch_vec]

        rows = np.column_stack([np.full(node_id.size, b, dtype=np.int64),
                                query_id, node_id])
        w_long.writerows(rows.tolist())
        counts = np.bincount(batch_vec, minlength=len(seeds))
        w_cnt.writerows(np.column_stack([np.full(len(seeds), b),
                                         seeds.numpy(), counts]).tolist())
        total_rows += node_id.size
        if b % 10 == 0:
            print(f"[dump] batch {b}: {node_id.size:,} rows "
                  f"(incl. duplicates across queries), "
                  f"{total_rows:,} total, {time.time()-t0:.0f}s", flush=True)

    f_long.close()
    f_cnt.close()
    print(f"[done] {total_rows:,} rows -> {args.out}.csv.gz")


if __name__ == "__main__":
    main()
