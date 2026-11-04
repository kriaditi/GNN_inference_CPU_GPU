"""Gate for the replacement d3 ({2, 5, 0} instead of {2, 5, 11}): its median nodes per batch must lie
strictly between d2 and d4 at EVERY fanout. Bounds are the measured GH200 sample_check values of 4 Nov
(d2 = shards {2, 5}, d4 = shards {2, 5, 11, 0}; the same graphs on every machine).
Exit 0 = passes (timed runs go ahead), exit 1 = fails (stop and look).

    python gate_d3alt.py results_capacity_order/sample_check_dalt3.csv dalt3
"""
import csv
import sys

BOUNDS = {  # fanout: (d2, d4) median nodes per batch, logs_gh200/sample_check.log, 4 Nov
    "40/20": (94_965, 116_832),
    "5/5": (16_467, 19_098),
    "10/5": (23_339, 27_644),
    "15/10": (43_456, 51_280),
    "25/15": (69_385, 83_042),
    "60/30": (129_600, 165_084),
}

path, name = sys.argv[1], sys.argv[2]
rows = {r["fanout"]: float(r["nodes_per_batch_median"]) for r in csv.DictReader(open(path)) if r["subset"] == name}
ok = True
for f, (lo, hi) in BOUNDS.items():
    v = rows.get(f)
    good = v is not None and lo < v < hi
    ok &= good
    print(f"  {f:>5}: d2 {lo:>9,}  <  {name} {v if v is None else f'{v:>9,.0f}'}  <  d4 {hi:>9,}   {'ok' if good else 'FAIL'}")
print(f"[gate] {name}: {'PASS, every fanout between d2 and d4' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
