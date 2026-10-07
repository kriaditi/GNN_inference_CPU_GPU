"""Turn <tag>_nodecount.npy and <tag>_batchreuse.csv into the locality table.

    python analyze_locality.py results/trace_b16384_f15x10_<job>

Reports:
  coverage   how many of the 121.75M papers were touched at all
  reuse      total touches / unique nodes touched
  hot set    share of all touches absorbed by the top 1/5/10/15% of
             touched nodes, and the GB a cache of that size would need
  overlap    per-batch fraction of nodes already seen in the previous
             1 / 10 / 100 batches (median over the run)
"""
import sys
import numpy as np
import pandas as pd

FEAT_BYTES = 768 * 2          # 768 float16 per paper

prefix = sys.argv[1]
count = np.load(prefix + "_nodecount.npy")
reuse = pd.read_csv(prefix + "_batchreuse.csv")

touched = count[count > 0]
total = int(touched.sum())
uniq = touched.size
print(f"graph nodes        : {count.size:,}")
print(f"unique touched     : {uniq:,}  ({uniq / count.size * 100:.2f}% of graph)")
print(f"total touches      : {total:,}")
print(f"reuse factor       : x{total / uniq:.2f}  (touches per touched node)")
print(f"touched once only  : {(touched == 1).sum() / uniq * 100:.1f}% of touched nodes")
print()

srt = np.sort(touched)[::-1]
cum = np.cumsum(srt)
print(f"{'top % of touched':>18} {'nodes':>12} {'share of touches':>17} {'cache GB':>9}")
for pct in (1, 5, 10, 15, 25, 50):
    k = max(1, int(uniq * pct / 100))
    print(f"{pct:>17}% {k:>12,} {cum[k - 1] / total * 100:>16.1f}% "
          f"{k * FEAT_BYTES / 1e9:>9.2f}")
print()

n_batches = len(reuse)
print(f"top-k individual nodes (of {n_batches} batches):")
top_ids = np.argpartition(count, -20)[-20:]
top_ids = top_ids[np.argsort(count[top_ids])[::-1]]
deg = None
try:
    rowptr = np.load(sys.argv[2]) if len(sys.argv) > 2 else None
    if rowptr is not None:
        deg = rowptr[1:] - rowptr[:-1]
except Exception:                                                 # noqa: BLE001
    pass
print(f"{'node id':>12} {'batches seen':>13} {'% of batches':>13} {'degree':>10}")
for nid in top_ids:
    d = f"{int(deg[nid]):,}" if deg is not None else "-"
    print(f"{nid:>12,} {count[nid]:>13,} {count[nid] / n_batches * 100:>12.1f}% {d:>10}")
for k in (10, 100, 1000, 10000):
    kk = min(k, uniq)
    print(f"  top {k:>6,} nodes appear in a median "
          f"{np.median(srt[:kk]) / n_batches * 100:5.1f}% of batches, "
          f"absorb {cum[kk - 1] / total * 100:5.1f}% of touches")
print()

print("per-batch overlap with earlier batches (median, as % of batch nodes):")
for col in [c for c in reuse.columns if c.startswith("in_prev_")]:
    frac = reuse[col] / reuse["nodes"] * 100
    print(f"  {col:<12} {frac.median():6.1f}%   (min {frac.min():.1f}, max {frac.max():.1f})")
frac_new = reuse["new"] / reuse["nodes"] * 100
print(f"  never seen   {frac_new.median():6.1f}%   "
      f"(first batch {frac_new.iloc[0]:.1f}, last batch {frac_new.iloc[-1]:.1f})")
