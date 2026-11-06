"""Charts for the simple 5 Nov deck (for Udit and Michael). Run from the repo root after build_dataset.py.
Every chart carries its own title with the batch definition and the table size, labelled axes, and a legend below.
Prints every plotted value. Outputs charts_simple/s*.png.
"""
import csv
import os

import numpy as np
import matplotlib
import matplotlib.ticker as mt
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAN = ["5/5", "10/5", "15/10", "25/15", "40/20", "60/30"]
BG, INK, INK2, GRID = "#fcfcfb", "#14213D", "#4A5568", "#e2e5ea"
COL = {"CPU-only": "#2a78d6", "GPU-only": "#eb6834", "GraphSAGE": "#eda100", "GraphSAGE + METIS": "#1baf7a", "Naive GPU": "#e87ba4",
       "GH200": "#2a78d6", "Cluster H100 NVL": "#eb6834", "Lambda H100": "#1baf7a"}
PH = [("sample_ms", "sample (CPU)", "#2a78d6"), ("filter_ms", "filter / plan (CPU)", "#86b6ef"), ("gather_cpu_ms", "gather rows (CPU)", "#eb6834"),
      ("transfer_ms", "transfer to GPU", "#eda100"), ("gather_gpu_ms", "gather rows (GPU)", "#1baf7a"), ("compute_ms", "compute", "#8e6bc7")]
BATCH = "batch = 1,024 seed papers; median of 1,000 batches per cell"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#9a9892", "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 13,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG, "legend.frameon": False})
os.makedirs("charts_simple", exist_ok=True)
R = list(csv.DictReader(open("data/all_rows.csv")))


def f(x):
    return float(x) if x not in ("", "nan") else float("nan")


def get(machine, method, fanout=None, note=None):
    return [r for r in R if r["machine"] == machine and r["method"] == method and (fanout is None or r["fanout"] == fanout) and (note is None or r["note"] == note)]


def one(machine, method, fanout, note=None):
    rs = get(machine, method, fanout, note)
    assert len(rs) == 1, (machine, method, fanout, note, len(rs))
    return rs[0]


def style(ax, ylog=False, ylabel="ms per batch (median)"):
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)
    if ylog:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mt.NullFormatter())


def save(fig, name, title=None):
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"charts_simple/{name}.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"-> charts_simple/{name}.png")


def grouped(ax, series, ylog=True, width=0.8, label_fmt="{:,.0f}"):
    n = len(series)
    x = np.arange(len(FAN))
    w = width / n
    for i, (lab, col, vals) in enumerate(series):
        xs = x - width / 2 + w * (i + 0.5)
        ax.bar(xs, vals, w * 0.92, color=col, label=lab)
        for xi, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(xi, v * 1.08 if ylog else v + 0.5, label_fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(FAN)
    ax.set_xlabel("fanout (neighbours sampled per node, hop 1 / hop 2)")
    style(ax, ylog)
    allv = np.array([v for _, _, vals in series for v in vals if not np.isnan(v)])
    if ylog and allv.size:
        ax.set_ylim(allv.min() / 2.5, allv.max() * 2.5)


def legend_below(ax, ncol):
    ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -0.16))


# s05: GH200, four methods on the full graph
fig, ax = plt.subplots(figsize=(16, 5.6))
S = []
for m in ["CPU-only", "Naive GPU", "GraphSAGE", "GraphSAGE + METIS"]:
    note = "numactl --membind=0" if m == "Naive GPU" else ""
    vals = [f(one("GH200", m, fo, note)["wall_ms"]) for fo in FAN]
    S.append((m, COL[m], vals))
    print("s05", m, [round(v, 1) for v in vals])
grouped(ax, S)
legend_below(ax, 4)
save(fig, "s05_gh200_methods", f"GH200 (NVLink-C2C), full graph, 375 GB feature table in host memory: ms per batch, log axis. {BATCH}.")

# s06: GraphSAGE + METIS, GH200 vs cluster
fig, ax = plt.subplots(figsize=(16, 5.6))
S = []
for mach, col, lab in [("GH200", COL["GH200"], "GH200, NVLink-C2C"), ("Cluster H100 NVL", COL["Cluster H100 NVL"], "H100 NVL, PCIe (cluster)")]:
    nt = "" if mach == "GH200" else None
    vals = [f(one(mach, "GraphSAGE + METIS", fo, nt)["wall_ms"]) for fo in FAN]
    gath = [f(one(mach, "GraphSAGE + METIS", fo, nt)["gather_cpu_ms"]) for fo in FAN]
    S.append((f"{lab}: whole batch", col, vals))
    S.append((f"{lab}: CPU gather of out-of-shard rows", col + "66", gath))
    print("s06", mach, [round(v, 1) for v in vals], "gather", [round(v, 1) for v in gath])
grouped(ax, S)
legend_below(ax, 2)
save(fig, "s06_metis_two_systems", f"GraphSAGE + METIS, k=16 (one 23 GB shard on the GPU, other rows fetched from host memory), full graph, NVLink-C2C vs PCIe. {BATCH} (1,008 here).")

# s07: GraphSAGE, GH200 vs cluster (the two shared-node cluster runs until 833128 is pushed)
fig, ax = plt.subplots(figsize=(16, 5.6))
S = [("GH200, NVLink-C2C", COL["GH200"], [f(one("GH200", "GraphSAGE", fo, "")["wall_ms"]) for fo in FAN])]
for note, col in [("exclusive node (833128)", COL["Cluster H100 NVL"]), ("19 Oct, shared node", "#f3a98a"), ("29 Oct, shared node", "#f7c9b5")]:
    rs = get("Cluster H100 NVL", "GraphSAGE", note=note)
    if rs:
        S.append((f"H100 NVL, PCIe (cluster), {note}", col, [f(one("Cluster H100 NVL", "GraphSAGE", fo, note)["wall_ms"]) for fo in FAN]))
for lab, _, vals in S:
    print("s07", lab, [round(v, 1) for v in vals])
grouped(ax, S)
legend_below(ax, 4)
save(fig, "s07_graphsage_two_systems", f"GraphSAGE (CPU samples and gathers only the needed rows, GPU computes), full graph, NVLink-C2C vs PCIe. {BATCH}.")

# s08: phases at 40/20, both systems
fig, ax = plt.subplots(figsize=(16, 6))
items = [("CPU-only", "GH200", ""), ("Naive GPU", "GH200", "numactl --membind=0"), ("Naive GPU", "Cluster H100 NVL", None),
         ("GraphSAGE", "GH200", ""), ("GraphSAGE", "Cluster H100 NVL", "exclusive node (833128)"),
         ("GraphSAGE + METIS", "GH200", ""), ("GraphSAGE + METIS", "Cluster H100 NVL", None),
         ("GPU-only, k=5", "GH200", ""), ("GPU-only, k=5", "Cluster H100 NVL", "")]
labels, y = [], 0
for m, mach, note in items:
    rs = get(mach, m, "40/20", note)
    if not rs:
        print("s08 missing", m, mach)
        continue
    r = rs[0]
    tot = f(r["wall_ms"])
    parts = [(lab, col, 0.0 if r[k] in ("", "nan") else f(r[k])) for k, lab, col in PH]
    ssum = sum(v for _, _, v in parts)
    left = 0
    for lab, col, v in parts:
        share = v / ssum * 100
        ax.barh(y, share, left=left, color=col, label=lab if y == 0 else None, height=0.72)
        if share >= 6:
            ax.text(left + share / 2, y, f"{v:,.0f} ms", ha="center", va="center", fontsize=9, color="white" if col in ("#2a78d6", "#eb6834", "#8e6bc7") else INK)
        left += share
    labels.append(f"{m.replace(', k=5', ' (77 GB part)')}, {'GH200 NVLink' if mach == 'GH200' else 'H100 PCIe'}:  {tot:,.0f} ms")
    print("s08", m, mach, round(tot, 1), [(lab, round(v, 1)) for lab, _, v in parts])
    y += 1
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("share of the batch time (%); the label in each segment is that phase's median ms")
ax.grid(True, axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=10)
save(fig, "s08_phases_4020", f"Where one batch's time goes at fanout 40/20 (about 274,000 nodes, 421 MB of feature rows per batch). {BATCH}.")

# s09: naive GPU, GH200 (cluster bars appear when its CSVs are in the repo)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [3, 2]})
S, rates = [], {}
for mach, col, lab in [("GH200", COL["GH200"], "GH200, NVLink-C2C"), ("Cluster H100 NVL", COL["Cluster H100 NVL"], "H100 NVL, PCIe (cluster)")]:
    note = "numactl --membind=0" if mach == "GH200" else None
    vals, gb = [], []
    for fo in FAN:
        rs = get(mach, "Naive GPU", fo, note)
        vals.append(f(rs[0]["wall_ms"]) if rs else np.nan)
        gb.append(f(rs[0]["stream_gbps"]) if rs else np.nan)
    S.append((lab, col, vals))
    rates[lab] = (col, gb)
    print("s09", mach, [round(v, 1) for v in vals], "GB/s", [round(v, 1) for v in gb])
grouped(ax, S)
ax.set_title("ms per batch (log axis)")
legend_below(ax, 2)
x = np.arange(len(FAN))
for i, (lab, (col, gb)) in enumerate(rates.items()):
    ax2.bar(x - 0.2 + 0.4 * i, gb, 0.36, color=col, label=lab)
    for xi, v in zip(x - 0.2 + 0.4 * i, gb):
        if not np.isnan(v):
            ax2.text(xi, v * 1.1, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
ax2.set_xticks(x)
ax2.set_xticklabels(FAN)
ax2.set_xlabel("fanout")
style(ax2, ylog=True, ylabel="host memory to GPU copy rate (GB/s)")
ax2.set_ylim(5, 1500)
ax2.set_title("copy rate = 375 GB / median copy time")
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16))
save(fig, "s09_naive_gpu", f"Naive GPU: all 5 METIS parts (375 GB) streamed through the GPU for every batch, G = 1. {BATCH}.")

# s10: GPU-only, same 77.3 GB part, GH200 vs H100 PCIe (cluster)
fig, ax = plt.subplots(figsize=(16, 5.6))
S = []
for mach, lab in [("GH200", "GH200 (H100 over NVLink-C2C, 96 GB)"), ("Cluster H100 NVL", "H100 NVL over PCIe (cluster, 94 GB)")]:
    vals = []
    for fo in FAN:
        rs = get(mach, "GPU-only, k=5", fo)
        vals.append(f(rs[0]["wall_ms"]) if rs else np.nan)
    S.append((lab, COL[mach], vals))
    print("s10", mach, [round(v, 1) for v in vals])
lam = [f(get("Lambda H100", "GPU-only, k=5", fo)[0]["wall_ms"]) if get("Lambda H100", "GPU-only, k=5", fo) else np.nan for fo in FAN]
print("s10 Lambda H100 (not drawn)", [round(v, 1) for v in lam])
grouped(ax, S)
legend_below(ax, 2)
save(fig, "s10_gpuonly_two_systems", f"GPU-only: the same 77.3 GB METIS k=5 part (50.3 M nodes) resident on each GPU; no feature rows cross the link. {BATCH}.")

# s12: accuracy vs shards searched (top-m), plus the four full-neighbourhood methods, 40/20
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [3, 2]})
pts = []
for m, lab in [("top2", "2"), ("top3", "3"), ("top4", "4"), ("top8", "8"), ("top16", "16")]:
    r = one("GH200", f"GraphSAGE + METIS, {m}", "40/20", "")
    pts.append((lab, f(r["wall_ms"]), f(r["val_acc"]) * 100))
r = one("GH200", "GraphSAGE + METIS", "40/20", "")
pts.append(("all 16, fetched", f(r["wall_ms"]), f(r["val_acc"]) * 100))
for lab, xms, acc in pts:
    ax.scatter(xms, acc, s=100, color=COL["GraphSAGE + METIS"], edgecolor=BG, zorder=3)
    ax.annotate(f"{lab} shards: {acc:.2f}% at {xms:.1f} ms", (xms, acc), xytext=(-8, -18) if lab.startswith("all") else (8, 8), textcoords="offset points", fontsize=10, color=INK, ha="right" if lab.startswith("all") else "left")
    print("s12", lab, round(xms, 1), round(acc, 2))
ax.set_xlabel("ms per batch (median), GH200, fanout 40/20")
style(ax, ylabel="validation accuracy (%), all 138,949 validation papers")
ax.set_title("GraphSAGE + METIS searching m of the 16 shards per batch")
names = ["CPU-only", "GraphSAGE", "Naive GPU", "GraphSAGE + METIS"]
accs = [f(one("GH200", n, "40/20", "numactl --membind=0" if n == "Naive GPU" else "")["val_acc"]) * 100 for n in names]
ax2.bar(range(4), accs, color=[COL[n] for n in names])
for i, v in enumerate(accs):
    ax2.text(i, v + 0.02, f"{v:.2f}%", ha="center", fontsize=10, color=INK)
ax2.set_xticks(range(4))
ax2.set_xticklabels(["CPU-only", "GraphSAGE", "Naive GPU", "GraphSAGE\n+ METIS"])
ax2.set_ylim(64, 65.6)
style(ax2, ylabel="validation accuracy (%)")
ax2.set_title("methods that see every neighbour, 40/20")
print("s12 full methods", dict(zip(names, [round(a, 2) for a in accs])))
save(fig, "s12_accuracy", "Accuracy on all 138,949 validation papers, GH200, fanout 40/20: searching more shards buys accuracy back; the full-neighbourhood methods agree within 0.1 points.")

# s13: streaming with G batches per pass, 40/20
fig, ax = plt.subplots(figsize=(16, 5.6))
for mach, col, mk, lab in [("GH200", COL["GH200"], "o", "GH200, NVLink-C2C"), ("Cluster H100 NVL", COL["Cluster H100 NVL"], "s", "H100 NVL, PCIe (cluster, shared node)")]:
    pts = sorted((int(r["G"]), f(r["wall_ms"]), r["note"]) for r in get(mach, "Streaming, G batches per pass", "40/20") if r["chunk"] == "24.0" and "6,000" not in r["note"])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=col, marker=mk, lw=2, ms=8, markeredgecolor=BG, label=f"{lab}, 24 GB chunks")
    for g, v, note in pts:
        ax.annotate(f"G={g}{' (auto)' if 'auto' in note else ''}: {v:,.0f} ms", (g, v), xytext=(6, 14 if g in (74, 110) else -14), textcoords="offset points", fontsize=9, color=INK2)
    print("s13", mach, [(g, round(v, 1), note) for g, v, note in pts])
ax.set_xscale("log")
ax.set_xlabel("G = batches that share one pass over the 375 GB table (log)")
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
style(ax, ylog=True, ylabel="ms per batch = pass time / G (log)")
legend_below(ax, 2)
save(fig, "s13_streaming_G", "Streaming with G batches sharing one pass over the 375 GB table, fanout 40/20: the figure per batch is pass time / G, and every batch waits for the whole pass (1.2 to 6.9 s on the GH200).")
print("done")
