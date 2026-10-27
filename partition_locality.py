"""Batch-level locality of METIS partitions for hetero GraphSAGE sampling.

Question: if one partition's feature rows are resident on the GPU, what
fraction of the rows a batch gathers are already there?

For each k, fanout and seed mode this script runs the real hetero sampler
(hetero_common.sample_hetero, the one the harness and training use) on
--batches batches of --batch-size seeds and records, per batch:

  frac_local            rows in the seed partition / rows gathered
  frac_local_inst_repl  same, counting institutions as always resident
                        (the ~2k-institution hot set replicated everywhere)
  frac_top1             rows in the single best partition for this batch
  parts_touched         distinct partitions the batch needs (search cost)
  per-type counts and per-type local fractions

Seed modes:
  grouped   all seeds drawn from one partition (round-robin over partitions);
            the way a partition-aware serving system would batch
  random    seeds drawn from all papers (today's baseline batching)

Output: <out>/locality_k{k}.csv (one row per batch) and a summary table on
stdout. CPU only; needs the CSRs (30 GB) and the part files.

    python partition_locality.py --ks 4,5,8 --fanouts 5/5,15/10,40/20 \
        --batches 200 --out results/partition
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from hetero_common import NODE_TYPES, HeteroGraph, sample_hetero


def run_one(g, part, k, fanout, mode, batches, batch_size, rng, paper_by_part, writer):
    P = g.num["paper"]
    rows = []
    for b in range(batches):
        if mode == "grouped":
            p = b % k
            seeds = rng.choice(paper_by_part[p], size=batch_size, replace=False)
        else:
            p = -1
            seeds = rng.choice(P, size=batch_size, replace=False)
        seeds = torch.from_numpy(np.sort(seeds)).long()
        node_id, _ = sample_hetero(g, seeds, fanout)

        gid = {nt: node_id[nt].numpy() + g.offset[nt] for nt in NODE_TYPES}
        pid = {nt: part[gid[nt]] for nt in NODE_TYPES}
        all_pid = np.concatenate([pid[nt] for nt in NODE_TYPES])
        n = all_pid.size
        counts = np.bincount(all_pid, minlength=k)
        top1 = int(counts.argmax())
        ref = p if mode == "grouped" else top1          # "resident" partition
        n_local = int(counts[ref])
        srt = np.sort(counts)[::-1]
        top2, top3 = int(srt[:2].sum()), int(srt[:3].sum())
        n_inst = pid["institution"].size
        n_local_inst_repl = n_local + n_inst - int((pid["institution"] == ref).sum())

        row = dict(k=k, fanout="x".join(map(str, fanout)), mode=mode, batch=b, seed_part=p,
                   resident_part=ref, nodes=n,
                   n_paper=pid["paper"].size, n_author=pid["author"].size, n_inst=n_inst,
                   frac_local=n_local / n,
                   frac_local_inst_repl=n_local_inst_repl / n,
                   frac_top1=int(counts.max()) / n,
                   frac_top2=top2 / n, frac_top3=top3 / n,
                   frac_local_paper=float((pid["paper"] == ref).mean()),
                   frac_local_author=float((pid["author"] == ref).mean()) if pid["author"].size else float("nan"),
                   frac_local_inst=float((pid["institution"] == ref).mean()) if n_inst else float("nan"),
                   parts_touched=int((counts > 0).sum()),
                   parts_for_90pct=int(np.searchsorted(np.cumsum(np.sort(counts)[::-1]), 0.9 * n) + 1))
        rows.append(row)
        writer.writerow(row)
    med = lambda key: float(np.median([r[key] for r in rows]))  # noqa: E731
    print(f"  k={k:<3} fanout {row['fanout']:<6} {mode:<8} nodes {med('nodes'):>9,.0f}  "
          f"local {med('frac_local')*100:5.1f}%  +inst repl {med('frac_local_inst_repl')*100:5.1f}%  "
          f"top1 {med('frac_top1')*100:5.1f}%  top2 {med('frac_top2')*100:5.1f}%  top3 {med('frac_top3')*100:5.1f}%  paper {med('frac_local_paper')*100:5.1f}%  "
          f"author {med('frac_local_author')*100:5.1f}%  inst {med('frac_local_inst')*100:5.1f}%  "
          f"parts touched {med('parts_touched'):.0f}  for 90% {med('parts_for_90pct'):.0f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csr", default="/share/gupta/ak2877/gnn/csr")
    ap.add_argument("--hetero", default="/share/gupta/ak2877/gnn/csr_hetero")
    ap.add_argument("--metis", default="/share/gupta/ak2877/gnn/metis")
    ap.add_argument("--ks", default="4,5,8")
    ap.add_argument("--fanouts", default="5/5,15/10,40/20")
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--out", default="results/partition")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n_thr = int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads()))
    torch.set_num_threads(n_thr)

    t0 = time.time()
    g = HeteroGraph(a.csr, a.hetero)
    print(f"[load] graph ready ({time.time()-t0:.0f}s)", flush=True)
    ks = [int(v) for v in a.ks.split(",")]
    fanouts = [[int(x) for x in f.split("/")] for f in a.fanouts.split(",")]
    rng = np.random.default_rng(0)
    P = g.num["paper"]

    summary = []
    for k in ks:
        pf = os.path.join(a.metis, f"part_k{k}.npy")
        if not os.path.exists(pf):
            print(f"[skip] {pf} not found", flush=True); continue
        part = np.load(pf).astype(np.int64)
        assert part.size == g.num["paper"] + g.num["author"] + g.num["institution"]
        paper_by_part = [np.flatnonzero(part[:P] == p) for p in range(k)]
        print(f"[k={k}] papers per part: {[v.size for v in paper_by_part]}", flush=True)
        with open(os.path.join(a.out, f"locality_k{k}.csv"), "w", newline="") as f:
            for fanout in fanouts:
                for mode in ("grouped", "random"):
                    rows = _run(g, part, k, fanout, mode, a, rng, paper_by_part, f)
                    summary.append(dict(k=k, fanout="x".join(map(str, fanout)), mode=mode,
                                        **{key: float(np.median([r[key] for r in rows]))
                                           for key in ("nodes", "frac_local", "frac_local_inst_repl",
                                                       "frac_top1", "frac_top2", "frac_top3",
                                                       "parts_touched", "parts_for_90pct")}))
        del part, paper_by_part
    with open(os.path.join(a.out, "locality_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {len(summary)} configs, {(time.time()-t0)/60:.1f} min -> {a.out}/", flush=True)


_writers = {}


def _run(g, part, k, fanout, mode, a, rng, paper_by_part, fh):
    """Create the CSV writer on first use (needs the row keys), then run."""
    class W:
        def __init__(self): self.w = None
        def writerow(self, row):
            if self.w is None:
                self.w = csv.DictWriter(fh, fieldnames=list(row)); self.w.writeheader()
            self.w.writerow(row); fh.flush()
    w = _writers.setdefault(fh.name, W())
    return run_one(g, part, k, fanout, mode, a.batches, a.batch_size, rng, paper_by_part, w)


if __name__ == "__main__":
    main()
