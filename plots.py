"""Regenerate every figure from the repo's result files.

    python plots.py                      # reads results/, writes figures/
    python plots.py --counts path.csv    # also the per-query histogram

Inputs (all optional except summary.csv; a missing input skips its figure):
  results/summary.csv
  results/recall_testdev2000_*.csv
  results/*_locality.txt            (trace_50297 recycled, traceall_65292 fresh)
  --counts  queries_*_counts.csv    (on the share, not in git)
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- palette (light surface), categorical slots in fixed order
C = {"sample": "#2a78d6", "gather": "#eb6834", "transfer": "#1baf7a",
     "compute": "#eda100", "gpu": "#2a78d6", "cpu": "#eb6834",
     "s3": "#1baf7a", "s4": "#eda100", "s5": "#e87ba4",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
PHASES = ["sample", "gather", "transfer", "compute"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
    "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})


def stacked(ax, df, xlabels, title, pct=False, width=0.6):
    """Stacked phase bars; df rows in x order with *_ms columns."""
    vals = df[[f"{p}_ms" for p in PHASES]].to_numpy()
    if pct:
        vals = vals / vals.sum(1, keepdims=True) * 100
    bottom = np.zeros(len(df))
    x = np.arange(len(df))
    for i, p in enumerate(PHASES):
        ax.bar(x, vals[:, i], width, bottom=bottom, color=C[p], label=p,
               edgecolor="white", linewidth=1.2)
        bottom += vals[:, i]
    ax.set_xticks(x, xlabels)
    ax.set_title(title, loc="left", fontsize=10, color=C["ink"])
    ax.set_ylabel("share of batch time (%)" if pct else "ms per batch")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=8)
    if not pct:
        for xi, tot in zip(x, vals.sum(1)):
            ax.text(xi, tot, f"{tot:,.0f}", ha="center", va="bottom",
                    fontsize=8, color=C["ink2"])


def fig_batch_latency(s, out):
    d = s[s.tag.str.startswith("series_cuda_")].sort_values("batch_size")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    stacked(ax, d, [f"{b:,}" for b in d.batch_size],
            "GPU: batch size vs latency, by phase (L4, 32 cores, fanout 15,10)")
    ax.text(len(d) - 0.55, d.wall_ms.max() * 1.02,
            "next step up\n(32,768) OOMs", ha="right", va="top", color=C["ink2"], fontsize=8)
    ax.set_xlabel("seeds per batch")
    fig.tight_layout()
    fig.savefig(f"{out}/01_batch_vs_latency_gpu.png")
    plt.close(fig)


def fig_cpu_vs_gpu(s, out):
    g = s[s.tag.str.startswith("series_cuda_")].sort_values("batch_size")
    c = s[s.tag.str.startswith("series_cpu_") & (s.batch_size <= 32768)]
    cb = s[s.tag.str.startswith("cpubeyond_")]
    if g.empty or c.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, ttl in [(axes[0], "wall_ms", "End to end"),
                         (axes[1], "compute_ms", "Compute phase only")]:
        ax.plot(g.batch_size, g[col], "o-", color=C["gpu"], lw=2, ms=6, label="GPU (L4)")
        ax.plot(c.sort_values("batch_size").batch_size, c.sort_values("batch_size")[col],
                "o-", color=C["cpu"], lw=2, ms=6, label="CPU (32 cores)")
        if not cb.empty:
            cb2 = cb.sort_values("batch_size")
            ax.plot(cb2.batch_size, cb2[col], "s--", color=C["cpu"], lw=1.5, ms=6,
                    label="CPU, seeds from all papers")
        ax.axvline(16384 * 1.35, color=C["ink2"], ls=":", lw=1)
        ax.text(16384 * 1.4, g[col].max(), "GPU OOM\n(23 GB)", fontsize=8,
                color=C["ink2"], va="top")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("seeds per batch")
        ax.set_ylabel("ms per batch")
        ax.set_title(ttl, loc="left", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("CPU keeps going where the GPU stops (fanout 15,10, 5 iterations)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/02_cpu_vs_gpu_batch.png")
    plt.close(fig)


def fig_fanout(s, out):
    d = s[s.tag.str.startswith("sweep_f")].copy()
    if d.empty:
        return
    d["f1"] = d.fanout.str.split(",").str[0].astype(int)
    d = d.sort_values("f1")
    labels = d.fanout.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    stacked(axes[0], d, labels, "Phase split vs fanout (batch 100)", pct=True)
    axes[0].set_xlabel("fanout")
    axes[1].plot(labels, d.throughput_qps, "o-", color=C["gpu"], lw=2, ms=6)
    axes[1].set_yscale("log")
    axes[1].set_title("Papers per second", loc="left", fontsize=10)
    axes[1].set_xlabel("fanout")
    nps = d.nodes_per_batch / (d.wall_ms / 1000) / 1e6
    axes[2].plot(labels, nps, "o-", color=C["s3"], lw=2, ms=6)
    axes[2].set_ylim(0, nps.max() * 1.2)
    axes[2].set_title("Million nodes per second", loc="left", fontsize=10)
    axes[2].set_xlabel("fanout")
    fig.tight_layout()
    fig.savefig(f"{out}/03_fanout_latency_throughput.png")
    plt.close(fig)


def fig_recall(s, out):
    files = sorted(glob.glob("results/recall_testdev*_*.csv"))
    if not files:
        return
    r = pd.read_csv(files[-1])
    r["f1"] = r.fanout.str.split(",").str[0].astype(int)
    g = r.groupby(["f1", "fanout"]).agg(r1=("recall_1hop", "mean"),
                                        r2m=("recall_2hop", "mean"),
                                        r2d=("recall_2hop", "median")).reset_index().sort_values("f1")
    sw = s[s.tag.str.startswith("sweep_f")].copy()
    sw["f1"] = sw.fanout.str.split(",").str[0].astype(int)
    sw = sw.sort_values("f1")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.plot(g.fanout, g.r1, "o-", color=C["gpu"], lw=2, ms=6, label="1-hop, mean")
    ax.plot(g.fanout, g.r2m, "o-", color=C["cpu"], lw=2, ms=6, label="2-hop, mean")
    ax.plot(g.fanout, g.r2d, "o--", color=C["cpu"], lw=1.5, ms=5, label="2-hop, median")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("recall vs exact neighbourhood")
    ax.set_xlabel("fanout")
    ax.set_title("Accuracy: how much of the true neighbourhood sampling sees",
                 loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(sw.fanout, sw.wall_ms / sw.batch_size, "o-", color=C["s3"], lw=2, ms=6)
    ax.set_ylabel("ms per paper")
    ax.set_xlabel("fanout")
    ax.set_title("Cost: latency per paper (GPU, batch 100)", loc="left", fontsize=10)
    fig.suptitle("Fanout trades accuracy for cost; 2-hop recall stays low at any affordable fanout",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/04_fanout_vs_recall.png")
    plt.close(fig)

    # recall by seed degree at 15,10
    d = r[r.fanout == "15,10"].copy()
    d["bucket"] = pd.cut(d.degree, [-1, 0, 15, 50, 200, 1e9],
                         labels=["0", "1-15", "16-50", "51-200", ">200"])
    b = d.groupby("bucket", observed=True)[["recall_1hop", "recall_2hop"]].mean()
    n = d.groupby("bucket", observed=True).size()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(b))
    ax.bar(x - 0.18, b.recall_1hop, 0.34, color=C["gpu"], label="1-hop", edgecolor="white")
    ax.bar(x + 0.18, b.recall_2hop, 0.34, color=C["cpu"], label="2-hop", edgecolor="white")
    ax.set_xticks(x, [f"{k}\n({v / len(d) * 100:.0f}% of seeds)" for k, v in n.items()])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall at fanout 15,10")
    ax.set_xlabel("seed degree (citations)")
    ax.set_title("Who loses under sampling: recall by seed degree", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/05_recall_by_degree.png")
    plt.close(fig)


def fig_sustained(s, out):
    rows = [("stress_b16384_f15x10", "burst\n5 iters"),
            ("scenB_i6000", "6,000 iters\nrecycled"),
            ("trace_b16384_f15x10", "6,000 iters\nrecycled"),
            ("traceall_b16384", "6,000 iters\nfresh seeds"),
            ("sustx_recycled", "1,000 iters\nexclusive"),
            ("sustx_fresh", "1,000 iters\nexclusive, fresh")]
    sel, labels = [], []
    for pre, lab in rows:
        m = s[s.tag.str.startswith(pre)]
        if not m.empty:
            sel.append(m.iloc[-1])
            labels.append(lab)
    if len(sel) < 2:
        return
    d = pd.DataFrame(sel)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    stacked(ax, d, labels,
            "Same config (batch 16384, fanout 15,10): gather varies 2.5 to 6x across runs; "
            "sample, transfer, compute do not")
    fig.tight_layout()
    fig.savefig(f"{out}/06_burst_vs_sustained.png")
    plt.close(fig)


def fig_nodes_per_sec(s, out):
    d = s[(s.fanout == "15,10") & (s.layers == 2) & (s.peak_gpu_gb > 0)
          & ~s.tag.str.startswith(("mag240m_s1024", "smoke"))].copy()
    if d.empty:
        return
    d["nps"] = d.nodes_per_batch / (d.wall_ms / 1000) / 1e6
    d["kind"] = np.where(d.iters >= 1000, "sustained (>=1000 iters)", "burst (<=200 iters)")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, col in [("burst (<=200 iters)", C["gpu"]), ("sustained (>=1000 iters)", C["cpu"])]:
        m = d[d.kind == k]
        ax.scatter(m.nodes_per_batch, m.nps, s=36, color=col, label=k, zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(0, d.nps.max() * 1.15)
    ax.set_xlabel("nodes per batch (working set)")
    ax.set_ylabel("million nodes per second")
    ax.set_title("GPU node throughput vs working set, burst and sustained", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/07_nodes_per_sec.png")
    plt.close(fig)


def parse_locality(path):
    txt = open(path).read()
    pts = re.findall(r"^\s*(\d+)%\s+[\d,]+\s+([\d.]+)%\s+([\d.]+)", txt, re.M)
    return [(int(a), float(b), float(c)) for a, b, c in pts]


def fig_hotset(out):
    files = {"recycled seeds (trace_50297)": glob.glob("results/trace_b16384*locality.txt"),
             "fresh seeds (trace_all_65292)": glob.glob("results/traceall_*locality.txt")}
    data = {k: parse_locality(v[0]) for k, v in files.items() if v}
    if not data:
        # fallback to the numbers from the job logs
        data = {"recycled seeds (trace_50297)": [(1, 16.0, .38), (5, 46.2, 1.88), (10, 64.9, 3.76),
                                                 (15, 75.7, 5.65), (25, 87.3, 9.41), (50, 97.2, 18.82)],
                "fresh seeds (trace_all_65292)": [(1, 13.2, 1.72), (5, 32.9, 8.62), (10, 48.2, 17.23),
                                                  (15, 60.0, 25.85), (25, 77.4, 43.08), (50, 97.4, 86.17)]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for (k, pts), col in zip(data.items(), [C["gpu"], C["cpu"]]):
        p = np.array(pts)
        axes[0].plot(p[:, 0], p[:, 1], "o-", color=col, lw=2, ms=6, label=k)
        axes[1].plot(p[:, 2], p[:, 1], "o-", color=col, lw=2, ms=6, label=k)
    axes[0].set_xlabel("top % of touched nodes kept")
    axes[0].set_ylabel("% of all feature reads served")
    axes[0].set_title("Hot set: a small fraction of nodes takes most reads", loc="left", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].axvline(23, color=C["ink2"], ls=":", lw=1)
    axes[1].text(23.5, 20, "L4 23 GB", fontsize=8, color=C["ink2"])
    axes[1].axvline(96, color=C["ink2"], ls=":", lw=1)
    axes[1].text(96.5, 20, "GH200 96 GB", fontsize=8, color=C["ink2"])
    axes[1].set_xlabel("cache size needed (GB, float16 features)")
    axes[1].set_ylabel("% of all feature reads served")
    axes[1].set_title("What a GPU-side cache would absorb", loc="left", fontsize=10)
    axes[1].set_xlim(0, 100)
    for ax in axes:
        ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(f"{out}/08_hotset_cache.png")
    plt.close(fig)


def fig_nfs(s, out):
    a = s[s.tag == "mag240m_s1024_b100_L2"]
    b = s[s.tag == "mag240m_inmem_L2"]
    if a.empty or b.empty:
        return
    d = pd.concat([a, b])
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar([0, 1], d.wall_ms, 0.55, color=[C["cpu"], C["gpu"]], edgecolor="white")
    ax.set_xticks([0, 1], ["features memmapped\nfrom NFS", "features resident\nin RAM"])
    ax.set_yscale("log")
    ax.set_ylabel("ms per batch (log)")
    ax.set_title("Where the feature table lives: 23,000x on the same code", loc="left", fontsize=10)
    for xi, tot in enumerate(d.wall_ms):
        ax.text(xi, tot * 1.3, f"{tot:,.0f} ms", ha="center", fontsize=8, color=C["ink2"])
    fig.tight_layout()
    fig.savefig(f"{out}/09_nfs_vs_ram.png")
    plt.close(fig)


def fig_counts(path, out):
    if not path or not os.path.exists(path):
        return
    c = pd.read_csv(path)
    bins = [0, 1, 25, 50, 75, 100, 125, 150, 160, 166]
    labels = ["1", "2-25", "26-50", "51-75", "76-100", "101-125", "126-150", "151-160", "161-166"]
    h = pd.cut(c.nodes_touched, bins, labels=labels).value_counts(sort=False) / len(c) * 100
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(labels, h.values, color=C["gpu"], edgecolor="white")
    ax.set_ylabel("% of queries")
    ax.set_xlabel("nodes pulled in by one query (max 166 at fanout 15,10)")
    ax.set_title(f"Per-query neighbourhood size ({len(c):,} queries)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/10_nodes_per_query.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--counts", default=None)
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = pd.read_csv(a.summary)
    for f in (fig_batch_latency, fig_cpu_vs_gpu, fig_fanout, fig_recall,
              fig_sustained, fig_nodes_per_sec, fig_nfs):
        f(s, a.out)
    fig_hotset(a.out)
    fig_counts(a.counts, a.out)
    print("wrote:", *sorted(os.listdir(a.out)), sep="\n  ")


if __name__ == "__main__":
    main()
