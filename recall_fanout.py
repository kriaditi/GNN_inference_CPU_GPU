"""Neighbourhood recall vs fanout, Hermes style.

Ground truth = exhaustive search: every neighbour of the seed, then every
neighbour of those (exact 2-hop set from the CSR, no sampling).
Retrieved    = the GraphSAGE-sampled neighbourhood at a given fanout.
Recall       = |retrieved| / |ground truth|   (retrieved is always a subset)

Reported per hop and for the full 2-hop set, across seeds.

    python recall_fanout.py --seeds 2000 --fanouts 5,5 10,5 15,10 25,15 40,20 100,50 \\
        --out results/recall_testdev_2000

Writes <out>.csv (one row per seed per fanout) and prints a summary table.
CSR only, no features, no GPU.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch


def exact_hops(rowptr, col, s):
    """Exact 1-hop and 2-hop neighbour sets of seed s (seed excluded)."""
    h1 = col[rowptr[s]:rowptr[s + 1]]
    if h1.size == 0:
        return h1, h1
    starts, ends = rowptr[h1], rowptr[h1 + 1]
    # gather all 2nd-hop neighbours of every 1st-hop neighbour
    idx = np.concatenate([np.arange(a, b) for a, b in zip(starts, ends)])
    h2 = np.union1d(h1, col[idx])
    h2 = h2[h2 != s]
    return h1, h2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/share/gupta/ak2877/gnn/data")
    p.add_argument("--csr", default="/share/gupta/ak2877/gnn/csr")
    p.add_argument("--seeds", type=int, default=2000)
    p.add_argument("--fanouts", nargs="+",
                   default=["5,5", "10,5", "15,10", "25,15", "40,20", "100,50"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import pyg_lib
    t0 = time.time()
    rowptr_np = np.load(os.path.join(args.csr, "rowptr.npy"))
    col_np = np.load(os.path.join(args.csr, "col.npy"))
    rowptr, col = torch.from_numpy(rowptr_np).long(), torch.from_numpy(col_np).long()
    print(f"[graph] CSR loaded ({time.time()-t0:.0f}s)", flush=True)

    from ogb.lsc import MAG240MDataset
    split = MAG240MDataset(root=args.root).get_idx_split("test-dev")
    rng = np.random.default_rng(0)
    seeds = np.sort(rng.choice(split, size=min(args.seeds, len(split)), replace=False))
    seeds_t = torch.from_numpy(seeds).long()
    print(f"[seeds] {len(seeds):,} from test-dev", flush=True)

    # ---- ground truth, exhaustive
    t0 = time.time()
    exact = {}
    for i, s in enumerate(seeds):
        h1, h2 = exact_hops(rowptr_np, col_np, int(s))
        exact[int(s)] = (h1.size, h2.size)
        if i % 500 == 0:
            print(f"[exact] {i}/{len(seeds)}  seed {s}: 1-hop {h1.size:,} "
                  f"2-hop {h2.size:,}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[exact] done ({time.time()-t0:.0f}s)", flush=True)

    # ---- sampled, one disjoint call per fanout
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out + ".csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["fanout", "seed", "degree", "exact_1hop", "exact_2hop",
                "sampled_1hop", "sampled_2hop", "recall_1hop", "recall_2hop"])
    summary = []
    for fo in args.fanouts:
        fan = [int(v) for v in fo.split(",")]
        out = pyg_lib.sampler.neighbor_sample(
            rowptr, col, seeds_t, fan, csc=False, replace=False,
            directed=True, disjoint=True, return_edge_id=False)
        pairs = out[2].numpy()                       # (N, 2): seed idx, node id
        n_per_hop = out[4]                           # [seeds, hop1, hop2]
        hop_of = np.repeat(np.arange(len(n_per_hop)), n_per_hop)
        r1, r2 = [], []
        for i, s in enumerate(seeds):
            mine = pairs[:, 0] == i
            nodes = pairs[mine, 1]
            hops = hop_of[mine]
            s1 = np.unique(nodes[hops == 1]).size
            s2 = np.unique(nodes[hops >= 1]).size
            e1, e2 = exact[int(s)]
            rc1 = s1 / e1 if e1 else 1.0
            rc2 = s2 / e2 if e2 else 1.0
            w.writerow([fo, int(s), e1, e1, e2, s1, s2, f"{rc1:.4f}", f"{rc2:.4f}"])
            r1.append(rc1)
            r2.append(rc2)
        r1, r2 = np.array(r1), np.array(r2)
        summary.append((fo, r1.mean(), np.median(r1), r2.mean(), np.median(r2),
                        np.percentile(r2, 10), (r2 >= 0.999).mean()))
        print(f"[fanout {fo:>6}] 1-hop recall mean {r1.mean():.3f}  "
              f"2-hop recall mean {r2.mean():.3f} median {np.median(r2):.3f}", flush=True)
    f.close()

    print("\nfanout   recall1 mean  recall2 mean  recall2 median  recall2 p10  seeds fully covered")
    for fo, m1, md1, m2, md2, p10, full in summary:
        print(f"{fo:>7} {m1:13.3f} {m2:13.3f} {md2:15.3f} {p10:12.3f} {full*100:18.1f}%")
    print(f"\nwrote {args.out}.csv")


if __name__ == "__main__":
    main()
