"""Figures for the H100 hetero baseline (job 378025).

    python h100_plots.py --summary results/summary_h100.csv --periter results --out figs

h01  accuracy / macro-F1 vs fanout (top) and latency per batch vs fanout (bottom)
h02  phase breakdown per fanout, stacked ms, with the gather share labelled
h03  energy per paper vs fanout
h04  gather time per row, iteration by iteration: the discrete-speed behaviour
h05  nodes per batch by type vs fanout: institutions saturate at ~2k
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OFF, DARK, MUTED, GRID = "#FAF9F6", "#16191D", "#6B6E73", "#E3E1DC"
# categorical, fixed order, validated (dataviz skill palette)
C_SAMPLE, C_GATHER, C_TRANSFER, C_COMPUTE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_PAPER, C_AUTHOR, C_INST = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": ["IBM Plex Sans", "DejaVu Sans"], "font.size": 10,
    "axes.facecolor": OFF, "figure.facecolor": OFF, "axes.edgecolor": GRID,
    "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def fanout_key(s):
    a, b = s.split(",")
    return int(a) * 1000 + int(b)


def load(summary_path):
    d = pd.read_csv(summary_path)
    d = d[d.iters == 1000].copy()            # the sweep points; sustained run shown separately
    d["key"] = d.fanout.map(fanout_key)
    return d.sort_values("key").reset_index(drop=True)


def h01(d, out):
    x = np.arange(len(d)); lab = d.fanout.str.replace(",", ", ")
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                               gridspec_kw=dict(hspace=0.12))
    a.plot(x, d.val_acc * 100, "-o", color=C_SAMPLE, lw=2, ms=7, label="validation accuracy")
    a.plot(x, d.val_macro_f1 * 100, "-o", color=C_GATHER, lw=2, ms=7, label="validation macro-F1")
    for i, r in d.iterrows():
        a.annotate(f"{r.val_acc*100:.1f}", (i, r.val_acc * 100), textcoords="offset points",
                   xytext=(0, 8), ha="center", fontsize=8.5, color=DARK)
        a.annotate(f"{r.val_macro_f1*100:.1f}", (i, r.val_macro_f1 * 100), textcoords="offset points",
                   xytext=(0, -14), ha="center", fontsize=8.5, color=DARK)
    a.set_ylim(45, 72); a.set_ylabel("percent")
    a.set_title("Accuracy barely moves with fanout; latency does", loc="left", fontsize=12,
                color=DARK, fontweight="semibold")
    a.legend(loc="center right", fontsize=9)
    b.bar(x, d.wall_ms, color=C_GATHER, width=0.6)
    for i, r in d.iterrows():
        b.annotate(f"{r.wall_ms:,.0f} ms", (i, r.wall_ms), textcoords="offset points",
                   xytext=(0, 4), ha="center", fontsize=8.5, color=DARK)
    b.set_ylabel("wall time per batch (ms)"); b.set_xlabel("fanout (hop 1, hop 2); batch 1,024")
    b.set_xticks(x); b.set_xticklabels(lab); b.set_ylim(0, d.wall_ms.max() * 1.18)
    fig.text(0.01, 0.005, "H100 NVL, R-GraphSAGE 2-layer/256 trained at 40,20, 375 GB table in RAM, "
             "fresh seeds, median of 1,000 iterations; accuracy on all 138,949 validation papers",
             fontsize=7.5, color=MUTED)
    fig.savefig(os.path.join(out, "h01_fanout_accuracy_latency.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def h02(d, out):
    x = np.arange(len(d)); lab = d.fanout.str.replace(",", ", ")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bottom = np.zeros(len(d))
    for col, c, name in [("sample_ms", C_SAMPLE, "sample (CPU)"), ("gather_ms", C_GATHER, "gather (CPU RAM)"),
                         ("transfer_ms", C_TRANSFER, "transfer (PCIe)"), ("compute_ms", C_COMPUTE, "compute (GPU)")]:
        ax.bar(x, d[col], bottom=bottom, color=c, width=0.6, label=name,
               edgecolor=OFF, linewidth=1.5)
        bottom += d[col].values
    for i, r in d.iterrows():
        share = r.gather_ms / r.wall_ms * 100
        ax.annotate(f"gather {share:.0f}%\ncompute {r.compute_ms:.0f} ms", (i, bottom[i]),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color=DARK)
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0, bottom.max() * 1.22)
    ax.set_ylabel("median ms per batch"); ax.set_xlabel("fanout (hop 1, hop 2); batch 1,024")
    ax.set_title("Where the time goes on the H100: the GPU waits for the gather", loc="left",
                 fontsize=12, color=DARK, fontweight="semibold")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(os.path.join(out, "h02_phase_breakdown.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def h03(d, out):
    x = np.arange(len(d)); lab = d.fanout.str.replace(",", ", ")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x, d.energy_mj_per_seed, color=C_COMPUTE, width=0.6)
    for i, r in d.iterrows():
        ax.annotate(f"{r.energy_mj_per_seed:.0f} mJ\n{r.mean_power_w:.0f} W", (i, r.energy_mj_per_seed),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5, color=DARK)
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0, d.energy_mj_per_seed.max() * 1.25)
    ax.set_ylabel("GPU energy per paper (mJ)"); ax.set_xlabel("fanout (hop 1, hop 2); batch 1,024")
    ax.set_title("Energy per paper scales with gather time, not with compute", loc="left",
                 fontsize=12, color=DARK, fontweight="semibold")
    fig.text(0.01, 0.005, "NVML board power, mean 91 to 98 W on a 400 W part: the GPU is mostly idle, "
             "so energy is time spent waiting", fontsize=7.5, color=MUTED)
    fig.savefig(os.path.join(out, "h03_energy.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def h04(periter_dir, out):
    runs = [("b1024_f5x5_i1000", "fanout 5, 5"), ("b1024_f40x20_i1000", "fanout 40, 20 (1,000 iters)"),
            ("b1024_f40x20_i6000", "fanout 40, 20 (6,000 iters, after requeue)")]
    fig, axes = plt.subplots(len(runs), 1, figsize=(7.2, 6.6), sharey=True,
                             gridspec_kw=dict(hspace=0.35))
    for ax, (tag, name) in zip(axes, runs):
        f = glob.glob(os.path.join(periter_dir, f"*{tag}_periter.csv"))
        if not f:
            continue
        p = pd.read_csv(f[0]); r = p.gather_s * 1e6 / p.nodes
        ax.plot(p.iter, r, lw=0.8, color=C_GATHER)
        ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
        ax.set_yscale("log"); ax.set_ylim(0.3, 40)
        fast = (r < 2).mean() * 100
        ax.set_title(f"{name}: median {r.median():.1f} µs/row, {fast:.0f}% of iterations in the fast mode",
                     loc="left", fontsize=10, color=DARK)
        ax.set_ylabel("µs per row")
    axes[-1].set_xlabel("iteration")
    fig.suptitle("Gather speed jumps between discrete levels (~1, ~3, ~10 µs/row) mid-run",
                 x=0.01, ha="left", fontsize=12, color=DARK, fontweight="semibold")
    fig.text(0.01, 0.005, "µs per row = gather time / nodes gathered (768-dim fp16 rows, 1.5 KB each) "
             "from the 375 GB table in RAM; dashed line = 1 µs/row. Shared GPU node.",
             fontsize=7.5, color=MUTED)
    fig.savefig(os.path.join(out, "h04_gather_two_speed.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def h05(d, out):
    x = np.arange(len(d)); lab = d.fanout.str.replace(",", ", ")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    b = np.zeros(len(d))
    for col, c, name in [("papers_per_batch", C_PAPER, "papers"), ("authors_per_batch", C_AUTHOR, "authors"),
                         ("insts_per_batch", C_INST, "institutions")]:
        ax.bar(x, d[col] / 1e3, bottom=b, color=c, width=0.6, label=name, edgecolor=OFF, linewidth=1.5)
        b += d[col].values / 1e3
    for i, r in d.iterrows():
        ax.annotate(f"{r.authors_per_batch/1e3:.0f}k authors\n{r.insts_per_batch:,.0f} inst.", (i, b[i]),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color=DARK)
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0, b.max() * 1.22)
    ax.set_ylabel("nodes gathered per batch (thousands)"); ax.set_xlabel("fanout (hop 1, hop 2); batch 1,024")
    ax.set_title("Institutions saturate at ~2,000 per batch: a 3 MB hot set every batch touches",
                 loc="left", fontsize=12, color=DARK, fontweight="semibold")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(os.path.join(out, "h05_nodes_by_type.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/summary_h100.csv")
    p.add_argument("--periter", default="results")
    p.add_argument("--out", default="figs")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load(a.summary)
    h01(d, a.out); h02(d, a.out); h03(d, a.out); h04(a.periter, a.out); h05(d, a.out)
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
