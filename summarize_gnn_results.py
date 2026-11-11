#!/usr/bin/env python3
"""High-level GNN inference summaries -> gnn_results/gnn_<method>_f40x20_b1024.csv.

Workload: R-SAGE (2 layers, hidden 256) on MAG240M, fanout 40x20, batch 1,024 seeds. Graph sizes s1-s8
are unions of METIS k=16 shards 0..n-1 (build_subset.py capacity sweep, 1,000 batches each); "full" is
the whole graph (244M nodes, 375 GB of fp16 features). Same runs as figures/scripts/03-Motivation-01.py
(CPU / GPU / PCIe) and 06-Evaluation-02 (Mezzanine).

  gnn_cpu_only_f40x20_b1024.csv          GH200 Grace CPU: sample + gather + compute (s1-s8, full)
  gnn_gpu_only_f40x20_b1024.csv          GH200 GPU, the whole subset in HBM (s1-s4; s4 = 94 GB is the
                                         largest that fits, so graph size is capped by HBM)
  gnn_pcie_h100_naive_f40x20_b1024.csv   Lambda H100 PCIe, naive GraphSAGE: features in host memory,
                                         gathered and copied over PCIe every batch (s1-s8; its 221 GB of
                                         RAM can't hold the full graph's 375 GB of features)
  gnn_mezzanine_metis_gh200_f40x20_b1024.csv
                                         GH200, METIS k=16 shards + GraphSAGE: each batch runs on its
                                         seeds' shard in HBM and fetches out-of-shard neighbors from
                                         Grace memory over NVLink-C2C (full graph, full accuracy)

Times are mean ms per batch; qps = seeds per second. val_acc is only measured on full-graph runs.

Usage: python summarize_gnn_results.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "gnn_results"
FANOUT = "40,20"
FEAT_BYTES = 768 * 2  # fp16 paper features per node
# Nodes per subset (capacity runs' subset_info.json, logged in logs_gh200/run_capacity.log).
NODES_M = {"s1": 15.146609, "s2": 30.544996, "s3": 45.780775, "s4": 61.179161, "s8": 122.044824,
           "full": 244.160499}
LEAD = ["method", "device", "graph", "num_nodes_m", "feature_gb", "fanout", "batch_size", "batches"]


def capacity(csv, tag, sizes):
    """Rows of a capacity-sweep summary for each size, keyed by graph size."""
    d = pd.read_csv(HERE / csv).set_index("tag")
    rows = [d.loc[tag.format(s)].to_dict() | {"graph": f"s{s}"} for s in sizes if tag.format(s) in d.index]
    return pd.DataFrame(rows)


def finish(df, method, device, cols):
    df = df.assign(method=method, device=device, fanout=FANOUT, batch_size=1024,
                   num_nodes_m=df["graph"].map(NODES_M))
    df["feature_gb"] = df["num_nodes_m"] * 1e6 * FEAT_BYTES / 1e9
    return df[LEAD + cols]


def main():
    full_cols = ["sample_ms", "gather_ms", "transfer_ms", "compute_ms", "wall_ms", "qps", "val_acc"]

    cpu = capacity("results_capacity_gh200/summary_capacity_cpu.csv", "gh200cap_s{}_cpu_b1024_f40x20_i1000",
                   [1, 2, 3, 4, 8])
    full = pd.read_csv(OUT / "raw" / "grace_cpu" / "full_graph.csv")
    full = full[(full["fanout"] == FANOUT) & (full["iters"] == 1000)].assign(graph="full")
    assert len(full) == 1
    cpu = pd.concat([cpu, full], ignore_index=True)
    cpu = cpu.rename(columns={"iters": "batches", "throughput_qps": "qps"})
    cpu = finish(cpu, "cpu_only", "gh200_grace", full_cols)

    gpu = capacity("results_capacity_gh200/summary_capacity_gpuonly.csv",
                   "gh200cap_s{}_gpuonly_f40x20_k1_drop_f40x20", [1, 2, 3, 4, 8])
    gpu = finish(gpu, "gpu_only", "gh200_hbm",
                 ["shard_gb_mean", "peak_gpu_gb", "sample_ms", "filter_ms", "gather_local_ms", "transfer_ms",
                  "compute_ms", "wall_ms", "qps", "load_gather_s_mean", "load_transfer_s_mean", "qps_incl_load"])
    gpu = gpu.rename(columns={"shard_gb_mean": "resident_gb"})

    pcie = capacity("results_capacity_h100/summary_capacity_h100.csv", "h100cap_s{}_nf_b1024_f40x20_i1000",
                    [1, 2, 3, 4, 8])
    pcie = pcie.rename(columns={"iters": "batches", "throughput_qps": "qps"})
    pcie = finish(pcie, "pcie_naive", "h100_pcie", full_cols)

    mz = pd.read_csv(OUT / "raw" / "gh200" / "shard_k16.csv")
    mz = mz[(mz["variant"] == "fetch") & (mz["fanout"] == FANOUT)].assign(graph="full")
    assert len(mz) == 1
    mz = finish(mz, "mezzanine_metis_k16", "gh200_nvlink_c2c",
                ["k", "frac_local", "mb_transfer", "sample_ms", "filter_ms", "gather_local_ms", "gather_remote_ms",
                 "transfer_ms", "compute_ms", "wall_ms", "qps", "shard_gb_mean", "qps_incl_load", "val_acc"])

    for name, df in [("cpu_only", cpu), ("gpu_only", gpu), ("pcie_h100_naive", pcie),
                     ("mezzanine_metis_gh200", mz)]:
        path = OUT / f"gnn_{name}_f40x20_b1024.csv"
        df.to_csv(path, index=False)
        print(f"{path.relative_to(HERE)}: " + ", ".join(f"{g}={w:.1f}ms" for g, w in zip(df["graph"], df["wall_ms"])))


if __name__ == "__main__":
    main()
