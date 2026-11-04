"""Sampling-only check for the latency-vs-capacity subsets: nodes sampled per batch, no features, no GPU.

Latency followed nodes sampled per batch in the 3 Nov runs, so before the long runs we check that
nodes per batch rises at every size step. Same seeds and sampler as the timed runs:
  seeds   default_rng(0).choice(papers, min(20M, papers), replace=False), unsorted, 1,024 per batch
          (the pool sage_mag240m.py --seed-mode all and shard_infer.py --unsorted-seeds use)
  sampler hetero_common.sample_hetero on the CPU

Pass the old subsets too (s1 s2 s3 s4 s8): their nodes per batch should come out close to the
logged values (40/20: s1 70,990 / s2 125,519 / s3 110,794 / s4 146,859 / s8 200,628), which
checks this script against the timed runs.

    python sample_check.py --subsets /unicorn/gnn/subsets/s1,...,/unicorn/gnn/subsets_density/d1,... \
        --out results_capacity_order/sample_check.csv
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

import hetero_common as hc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", required=True, help="comma list of subset folders (each with csr/, csr_hetero/, subset_info.json)")
    ap.add_argument("--fanouts", default="40/20,5/5,10/5,15/10,25/15,60/30")
    ap.add_argument("--batches", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=2, help="batches skipped first, as in the timed runs")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=20_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fanouts = [[int(v) for v in f.split("/")] for f in a.fanouts.split(",")]
    rows = []

    for S in [s for s in a.subsets.split(",") if s]:
        name = os.path.basename(os.path.normpath(S))
        with open(os.path.join(S, "subset_info.json")) as f:
            info = json.load(f)
        t0 = time.time()
        g = hc.HeteroGraph(os.path.join(S, "csr"), os.path.join(S, "csr_hetero"))
        P = g.num["paper"]
        pool = np.random.default_rng(0).choice(P, size=min(a.seeds, P), replace=False)
        need = a.warmup + a.batches
        assert pool.size >= need * a.batch_size, f"{name}: only {pool.size} seeds for {need} batches"
        edges = sum(info["edges"].values())
        print(f"\n[{name}] shards {info['shards']}: {info['nodes']:,} nodes, {edges:,} edges "
              f"({edges / info['nodes']:.2f}/node), {info['feat_gb']:.1f} GB (graph loaded in {time.time() - t0:.0f}s)", flush=True)
        for fo in fanouts:
            t1 = time.time()
            counts = []
            for i in range(need):
                b = torch.from_numpy(pool[i * a.batch_size:(i + 1) * a.batch_size]).long()
                node_id, _ = hc.sample_hetero(g, b, fo)
                if i >= a.warmup:
                    counts.append(sum(int(v.numel()) for v in node_id.values()))
            c = np.asarray(counts)
            r = dict(subset=name, shards=" ".join(map(str, info["shards"])), nodes=info["nodes"], edges=edges,
                     edges_per_node=edges / info["nodes"], feat_gb=info["feat_gb"], fanout="/".join(map(str, fo)),
                     batches=c.size, nodes_per_batch_median=float(np.median(c)), nodes_per_batch_mean=float(c.mean()),
                     nodes_per_batch_min=int(c.min()), nodes_per_batch_max=int(c.max()))
            rows.append(r)
            print(f"  {r['fanout']:>5}: median {r['nodes_per_batch_median']:>11,.0f}  mean {r['nodes_per_batch_mean']:>11,.0f}  "
                  f"({time.time() - t1:.0f}s)", flush=True)
        del g

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # does nodes per batch rise with size? checked separately for each family (s..., d...)
    print("\n[rises?] median nodes per batch, in the order given")
    for fam in sorted({r["subset"].rstrip("0123456789") for r in rows}):
        for fo in fanouts:
            f = "/".join(map(str, fo))
            seq = [r for r in rows if r["fanout"] == f and r["subset"].rstrip("0123456789") == fam]
            seq.sort(key=lambda r: r["nodes"])
            vals = [r["nodes_per_batch_median"] for r in seq]
            drops = [f"{seq[i]['subset']}->{seq[i + 1]['subset']}" for i in range(len(vals) - 1) if vals[i + 1] < vals[i]]
            print(f"  {fam} {f:>5}: " + " / ".join(f"{v:,.0f}" for v in vals) + ("   DROPS at " + ", ".join(drops) if drops else "   rises at every step"))
    print(f"\n[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
