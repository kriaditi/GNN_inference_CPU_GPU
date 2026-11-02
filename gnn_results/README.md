# gnn_results

One place to find the GNN inference results, grouped by machine. Every file here is a
**symlink** to a summary CSV in one of the `results_*` folders, so new runs written by the
existing scripts show up here automatically. Per-iteration CSVs stay in those folders.

Workload: R-SAGE (2 layers, hidden 256) inference on the full MAG240M graph
(~244M nodes, 375 GB of node features), batch 1,024 seeds unless the file says otherwise.
Fanout sweeps run 5x5 up to 60x30. Every run uses the same graph; none vary the graph size.

| Folder | Machine |
|---|---|
| `grace_cpu/` | GH200 Grace CPU only |
| `gh200/` | GH200 GPU, features in Grace memory (NVLink-C2C) |
| `h100/` | Lambda H100 (PCIe) |
| `h100nvl/` | Cluster H100 NVL (PCIe) |

| File | What it is | Source |
|---|---|---|
| `grace_cpu/full_graph.csv` | CPU sample + gather + compute | `results_gh200/summary_gh200_cpu.csv` |
| `gh200/full_graph_cpu_gather.csv` | CPU sample/gather, GPU compute | `results_gh200/summary_gh200_nvlink.csv` |
| `gh200/shard_k16.csv` | METIS k=16 shards (drop / fetch) | `results_shard_gh200/summary_shard_gh200.csv` |
| `gh200/shard_k16_rep2.csv` | repeat of the above | `results_shard_gh200_rep2/summary_shard_gh200_rep2.csv` |
| `gh200/shard_k16_topm.csv` | k=16, keep top-m shards | `results_topm_gh200/summary_topm_gh200.csv` |
| `gh200/stream.csv` | chunked feature streaming | `results_stream_gh200/summary_stream_gh200.csv` |
| `gh200/stream_auto.csv` | streaming, auto group size | `results_stream_gh200_auto/summary_stream_gh200_auto.csv` |
| `gh200/stream_naive.csv` | naive streaming, G=1, with accuracy | `results_stream_gh200_naive/summary_stream_gh200_naive.csv` |
| `gh200/sustained_shard_k16_6000.csv` | 6,000-batch sustained shard run | `results_sustained_gh200/summary_shard_gh200_6000.csv` |
| `gh200/sustained_stream_6000.csv` | 6,000-batch sustained streaming | `results_sustained_gh200/summary_stream_gh200_6000.csv` |
| `gh200/gpuonly_subgraph.csv` | largest shard only, in HBM | `results_gpuonly_gh200/summary_gpuonly_gh200.csv` |
| `h100/full_graph_cpu_gather.csv` | CPU sample/gather, GPU compute | `results/summary_h100.csv` |
| `h100/shard_k16.csv` | METIS k=16 shards (drop) | `results_shard_h100/summary_shard_h100_drop.csv` |
| `h100/gpuonly_subgraph.csv` | largest shard only, in HBM | `results_gpuonly_h100/summary_gpuonly_h100.csv` |
| `h100nvl/shard_k16.csv` | METIS k=16 shards (drop / fetch) | `results_shard_h100nvl/summary_shard_h100nvl.csv` |
| `h100nvl/gpuonly_subgraph.csv` | largest shard only, in HBM | `results_gpuonly_h100nvl/summary_gpuonly_h100nvl.csv` |
| `h100nvl/stream_naive_c62.6gb.csv` | naive streaming, 62.6 GB chunks | `results_stream_h100nvl_naive/summary_naive_f40x20_c62.6.csv` |
| `h100nvl/stream_naive_c75.01gb.csv` | naive streaming, 75.01 GB chunks | `results_stream_h100nvl_naive/summary_naive_f40x20_c75.01.csv` |

Notes:
- `gpuonly_subgraph` runs only see one METIS shard (1/5 to 1/8 of the graph), so their
  accuracy (~0.34-0.39) is not comparable to full-graph runs (~0.65).
- `shard_*` "drop" ignores neighbors outside the shard (accuracy ~0.63); "fetch" pulls them
  from host memory (full accuracy ~0.65). Use `qps_incl_load` to include shard load time.
- Left out: `results/summary.csv` (older homogeneous-model runs, mixed columns) and
  `results_gh200/summary_smoke.csv` (smoke tests).
