"""Order the METIS k=16 shards so the latency-vs-capacity subsets grow steadily in density.

Reads the graph only (CPU, no GPU, builds nothing). Counts edges exactly the way build_subset.py
keeps them: an edge belongs to a subset when BOTH ends are inside it, over all 5 relations.

  1. shard x shard edge matrix M[a, b] = edges whose destination is in shard a and source in shard b
     -> edges of any subset S = sum of M over S x S (no rebuild needed to know a subset's size)
  2. check: the old order (shards 0..n-1) must reproduce build_subset.py's counts
     (s1 136,527,296 / s2 301,964,282 / s3 376,312,418 / s4 589,406,568 / s8 1,374,349,932 edges)
  3. greedy order: start with the shard whose own edges per node is lowest; at each step add the
     shard that raises the subset's edges per node the LEAST without lowering it. If every remaining
     shard would lower it, add the one that lowers it least and flag that step.

Outputs (in --out):
  shard_stats_k16.csv     per shard: nodes, internal edges, internal edges per node, feature GB
  order_old_k16.csv       cumulative nodes, edges, edges/node for shards 0..n-1 (the current subsets)
  order_density_k16.csv   the same for the greedy order, one row per step (1..16)
  order_density_k16.json  the greedy order as a list (input for build_subset.py later)
  edge_matrix_k16.npy     M

    python order_shards.py --csr /unicorn/gnn/csr --hetero-dir /unicorn/gnn/csr_hetero \
        --metis /unicorn/gnn/metis --k 16 --out results_capacity_order
"""
import argparse
import csv
import json
import os
import time

import numpy as np

# same table as build_subset.py: (name, (colptr file, row file), dst type, src type, folder)
RELATIONS = [
    ("cites",          ("rowptr.npy", "col.npy"),         "paper",       "paper",       "csr"),
    ("writes",         ("p2a_rowptr.npy", "p2a_col.npy"), "paper",       "author",      "csr_hetero"),
    ("rev_writes",     ("a2p_rowptr.npy", "a2p_col.npy"), "author",      "paper",       "csr_hetero"),
    ("rev_affiliated", ("a2i_rowptr.npy", "a2i_col.npy"), "author",      "institution", "csr_hetero"),
    ("affiliated",     ("i2a_rowptr.npy", "i2a_col.npy"), "institution", "author",      "csr_hetero"),
]
TYPES = ["paper", "author", "institution"]
BYTES_PER_NODE = 768 * 2   # fp16 features


def edge_matrix(src_dir, fp, fr, part_dst, part_src, k, block):
    """k x k edge counts for one relation, read in blocks of destination nodes."""
    colptr = np.load(os.path.join(src_dir, fp), mmap_mode="r")
    row = np.load(os.path.join(src_dir, fr), mmap_mode="r")
    n_dst = colptr.size - 1
    assert n_dst == part_dst.size, f"{fp}: {n_dst} destinations, part has {part_dst.size}"
    m = np.zeros(k * k, dtype=np.int64)
    for i in range(0, n_dst, block):
        j = min(i + block, n_dst)
        ptr = np.asarray(colptr[i:j + 1], dtype=np.int64)
        if ptr[-1] == ptr[0]:
            continue
        pd = np.repeat(part_dst[i:j], np.diff(ptr))
        ps = part_src[np.asarray(row[ptr[0]:ptr[-1]])]
        m += np.bincount(pd * k + ps, minlength=k * k)
    return m.reshape(k, k), int(colptr[-1])


def cumulative(order, n_shard, M):
    rows = []
    for step in range(1, len(order) + 1):
        S = order[:step]
        nodes = int(n_shard[S].sum())
        edges = int(M[np.ix_(S, S)].sum())
        rows.append(dict(step=step, shard_added=order[step - 1], nodes=nodes, edges=edges,
                         edges_per_node=edges / nodes, feat_gb=nodes * BYTES_PER_NODE / 1e9))
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csr", required=True)
    ap.add_argument("--hetero-dir", required=True)
    ap.add_argument("--metis", required=True, help="directory with part_k{k}.npy")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--block", type=int, default=5_000_000, help="destination nodes per read")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    k, t_all = a.k, time.time()

    with open(os.path.join(a.hetero_dir, "meta.json")) as f:
        meta = json.load(f)
    num = {"paper": meta["num_papers"], "author": meta["num_authors"], "institution": meta["num_institutions"]}
    off = meta["offset"]
    N = sum(num.values())
    part = np.load(os.path.join(a.metis, f"part_k{k}.npy")).astype(np.int64)
    assert part.size == N, f"part has {part.size} entries, graph has {N}"
    ptype = {t: part[off[t]: off[t] + num[t]] for t in TYPES}
    n_shard = np.bincount(part, minlength=k).astype(np.int64)
    print(f"[load] {N:,} nodes, k={k}, shard sizes {n_shard.min():,} to {n_shard.max():,}", flush=True)

    # 1. edge matrix over all relations
    M = np.zeros((k, k), dtype=np.int64)
    total = 0
    for name, (fp, fr), dst_t, src_t, folder in RELATIONS:
        t0 = time.time()
        src_dir = a.csr if folder == "csr" else a.hetero_dir
        m, n_e = edge_matrix(src_dir, fp, fr, ptype[dst_t], ptype[src_t], k, a.block)
        assert m.sum() == n_e, f"{name}: matrix holds {m.sum()} edges, file has {n_e}"
        M += m
        total += n_e
        print(f"[{name}] {n_e:,} edges, {np.trace(m) / n_e * 100:.1f}% inside one shard ({time.time() - t0:.0f}s)", flush=True)
    print(f"[edges] {total:,} in all relations", flush=True)
    np.save(os.path.join(a.out, f"edge_matrix_k{k}.npy"), M)

    # per-shard stats
    stats = [dict(shard=p, nodes=int(n_shard[p]), internal_edges=int(M[p, p]),
                  internal_edges_per_node=M[p, p] / n_shard[p], feat_gb=n_shard[p] * BYTES_PER_NODE / 1e9)
             for p in range(k)]
    write_csv(os.path.join(a.out, f"shard_stats_k{k}.csv"), stats)
    print("\n[shards] shard  nodes  internal edges  edges/node", flush=True)
    for s in stats:
        print(f"  {s['shard']:>2}  {s['nodes']:>12,}  {s['internal_edges']:>14,}  {s['internal_edges_per_node']:6.2f}")

    # 2. check against the subsets build_subset.py made (shards 0..n-1)
    old = cumulative(list(range(k)), n_shard, M)
    write_csv(os.path.join(a.out, f"order_old_k{k}.csv"), old)
    print("\n[old order 0..n-1] compare with subset_info.json of s1, s2, s3, s4, s8 and the full graph")
    for r in old:
        print(f"  {r['step']:>2} shards  {r['nodes']:>12,} nodes  {r['edges']:>14,} edges  {r['edges_per_node']:6.2f}/node")

    # 3. greedy density order
    remaining = list(range(k))
    first = min(remaining, key=lambda p: M[p, p] / n_shard[p])
    order, flags = [first], [""]
    remaining.remove(first)
    nodes, edges = int(n_shard[first]), int(M[first, first])
    while remaining:
        cur = edges / nodes
        cand = []
        for q in remaining:
            e_add = int(M[order, q].sum() + M[q, order].sum() + M[q, q])
            cand.append(((edges + e_add) / (nodes + n_shard[q]), q, e_add))
        up = [c for c in cand if c[0] >= cur]
        d, q, e_add = min(up) if up else max(cand)
        flags.append("" if up else "LOWERS")
        order.append(q)
        remaining.remove(q)
        nodes += int(n_shard[q])
        edges += e_add
    new = cumulative(order, n_shard, M)
    for r, fl in zip(new, flags):
        r["flag"] = fl
    write_csv(os.path.join(a.out, f"order_density_k{k}.csv"), new)
    with open(os.path.join(a.out, f"order_density_k{k}.json"), "w") as f:
        json.dump(dict(k=k, order=order, rule="greedy: smallest rise in edges per node, never a drop if avoidable"), f, indent=1)
    print(f"\n[density order] {order}")
    for r in new:
        print(f"  {r['step']:>2} +shard {r['shard_added']:>2}  {r['nodes']:>12,} nodes  {r['edges']:>14,} edges  "
              f"{r['edges_per_node']:6.2f}/node  {r['feat_gb']:6.1f} GB  {r['flag']}")
    print(f"\n[done] {time.time() - t_all:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
