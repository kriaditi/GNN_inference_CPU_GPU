#!/usr/bin/env python3
"""High-level GNN inference summaries -> gnn_results/gnn_<method>_f40x20_b1024.csv.

Workload: R-SAGE (2 layers, hidden 256) on MAG240M, fanout 40x20, batch 1,024 seeds. Graph sizes d1-d12 are
the density-ordered capacity subsets: unions of the first n METIS k=16 shards, added from the sparsest to the
densest (results_capacity_order/order_density_k16.json), 1,000 batches each; "full" is the whole graph (244M
nodes, 375 GB of fp16 features). The points are the in_plot rows of table7_density_long_5nov.csv, which
leaves out d3 (it samples more nodes per batch than d4, so the curve would dip at d4).

  gnn_cpu_only_f40x20_b1024.csv          GH200 Grace CPU: sample + gather + compute (d1-d12, full)
  gnn_gpu_only_f40x20_b1024.csv          GH200 GPU, the whole subset in HBM (d1-d4; d4 = 93 GB is the
                                         largest that fits, so graph size is capped by HBM)
  gnn_pcie_h100_naive_f40x20_b1024.csv   Lambda H100 PCIe, naive GraphSAGE: features in host memory,
                                         gathered and copied over PCIe every batch (d1-d8; its 221 GB of
                                         RAM can't hold larger subsets)
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
TABLE = pd.read_csv(HERE / "table7_density_long_5nov.csv")
TABLE = TABLE[(TABLE["fanout"] == "40/20") & (TABLE["in_plot"] == "yes")]
NODES_M = dict(zip(TABLE["subset"], TABLE["nodes"] / 1e6))


def sizes(line):
    """Density subsets (as ints) table7 plots for one line, e.g. [1, 2, 4, 6, 8, 12]."""
    subs = TABLE.loc[TABLE["line"] == line, "subset"]
    return sorted(int(x[1:]) for x in subs if x.startswith("d"))


LEAD = ["method", "device", "graph", "num_nodes_m", "feature_gb", "fanout", "batch_size", "batches"]


def capacity(csv, tag, sizes):
    """Rows of a capacity-sweep summary for each size, keyed by graph size."""
    d = pd.read_csv(HERE / csv).set_index("tag")
    missing = [s for s in sizes if tag.format(s) not in d.index]
    assert not missing, f"{csv}: no rows for d{missing}"
    return pd.DataFrame([d.loc[tag.format(s)].to_dict() | {"graph": f"d{s}"} for s in sizes])


def finish(df, method, device, cols):
    df = df.assign(method=method, device=device, fanout=FANOUT, batch_size=1024,
                   num_nodes_m=df["graph"].map(NODES_M))
    df["feature_gb"] = df["num_nodes_m"] * 1e6 * FEAT_BYTES / 1e9
    return df[LEAD + cols]


def main():
    full_cols = ["sample_ms", "gather_ms", "transfer_ms", "compute_ms", "wall_ms", "qps", "val_acc"]

    cpu = capacity("results_capacity_density_gh200/summary_capacity_density_cpu.csv",
                   "gh200den_d{}_cpu_b1024_f40x20_i1000", sizes("cpu"))
    full = pd.read_csv(OUT / "raw" / "grace_cpu" / "full_graph.csv")
    full = full[(full["fanout"] == FANOUT) & (full["iters"] == 1000)].assign(graph="full")
    assert len(full) == 1
    cpu = pd.concat([cpu, full], ignore_index=True)
    cpu = cpu.rename(columns={"iters": "batches", "throughput_qps": "qps"})
    cpu = finish(cpu, "cpu_only", "gh200_grace", full_cols)

    gpu = capacity("results_capacity_density_gh200/summary_capacity_density_gpuonly.csv",
                   "gh200den_d{}_gpuonly_f40x20_k1_drop_f40x20", sizes("gpuonly"))
    gpu = finish(gpu, "gpu_only", "gh200_hbm",
                 ["shard_gb_mean", "peak_gpu_gb", "sample_ms", "filter_ms", "gather_local_ms", "transfer_ms",
                  "compute_ms", "wall_ms", "qps", "load_gather_s_mean", "load_transfer_s_mean", "qps_incl_load"])
    gpu = gpu.rename(columns={"shard_gb_mean": "resident_gb"})

    pcie = capacity("results_capacity_density_h100/summary_capacity_density_h100.csv",
                    "h100den_d{}_nf_b1024_f40x20_i1000", sizes("pcie"))
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
