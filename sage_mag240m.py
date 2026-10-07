"""GraphSAGE inference on MAG240M with phase-level timing.
=============================================================================
WHAT THE SCRIPT DOES
=============================================================================
It answers one question per paper: which of 153 subject areas is it in?
But the predictions are not the point. The point is measuring how long each
stage takes and how much energy it burns.

For each batch of seed papers:

  1. SAMPLE    Walk the CSR to pick a bounded set of neighbours.
               With fanout [15,10]: 15 cited papers, then 10 of each of
               those. Bounded on purpose, because some papers are cited
               100,000 times and using every neighbour would explode.
               Cost: pointer chasing through rowptr/col.       -> CPU

  2. GATHER    Read those nodes' 768-dim feature vectors out of the 175 GB
               memmapped file. Scattered node IDs mean random reads into a
               very large array. Usually the expensive phase.  -> CPU + disk

  3. TRANSFER  Copy that gathered block to the GPU over PCIe.
               Features stay float16 here so only half the bytes cross the
               link; the cast to float32 happens after arrival. -> PCIe

  4. COMPUTE   The GraphSAGE forward pass: aggregate neighbours, multiply
               by learned weights, apply ReLU, repeat per layer.
               Usually the cheapest phase, which is the finding.  -> GPU

  Output: for each seed paper, 153 scores. Highest score is the predicted
  subject area.

=============================================================================
WHY THE PHASES ARE TIMED SEPARATELY
=============================================================================
The project's whole question is where time and energy go when features
cannot fit in GPU memory. rocky has a 23 GB L4 against 1 TB of node RAM, so
the 175 GB feature table must live in CPU memory and stream across PCIe.
Splitting the timers tells you whether that streaming is the bottleneck.

Every measured iteration also checks that the four phase times sum to wall
clock. If they do not, the attribution is wrong and the breakdown means
nothing, so the script says so loudly rather than letting you build on it.

=============================================================================
SCALING LADDER
=============================================================================
Change one thing at a time. Each step should work before the next.

  python sage_mag240m.py --seeds 100   --fanout 5      --layers 1  --iters 5
  python sage_mag240m.py --seeds 1024  --fanout 15,10  --layers 2
  python sage_mag240m.py --seeds 88092 --fanout 15,10  --layers 2 --iters 50

Note: --fanout must have exactly --layers entries. One number per hop.

=============================================================================
WHY NOT NeighborLoader
=============================================================================
PyG's NeighborLoader would do sampling and gathering in one opaque step,
which makes them impossible to time separately. Since separating them is
the entire purpose here, we drive pyg_lib's sampler ourselves and do the
feature gather by hand.
"""
import argparse
import csv
import os
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# --------------------------------------------------------------------------
# THE MODEL
# --------------------------------------------------------------------------
class SAGE(torch.nn.Module):
    """GraphSAGE. One SAGEConv per layer, and one layer per hop.

    A SAGEConv does the gather-aggregate-update cycle: collect each node's
    neighbours, average them, concatenate with the node's own vector,
    multiply by a learned weight matrix.

    Weights here are random, because this project measures systems
    behaviour, not accuracy. Random weights cost exactly the same FLOPs and
    move exactly the same bytes as trained ones. If you later want real
    predictions, load trained weights; nothing else changes.

    Layer dimensions for 768-dim input, 2 layers, hidden 256, 153 classes:
        layer 0:  768 -> 256   then ReLU
        layer 1:  256 -> 153   (no activation, these are the class scores)
    """

    def __init__(self, in_dim, hidden, out_dim, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))

    def forward(self, x, edge_index):
        # x: (num_nodes_in_batch, feat_dim), edge_index: (2, num_edges)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:      # no ReLU on the final layer
                x = F.relu(x)
        return x


# --------------------------------------------------------------------------
class PowerSampler:
    """Poll GPU power in the background. NVML reports the WHOLE GPU, so use
    --exclusive in Slurm for any run whose energy numbers you intend to
    quote."""

    def __init__(self, interval=0.05, index=0):
        self.interval, self.samples = interval, []
        self._stop = threading.Event()
        self._t = None
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.ok = True
        except Exception as e:                                    # noqa: BLE001
            print(f"[power] NVML unavailable: {e}")

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append(
                    (time.perf_counter(),
                     self.nvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0))
            except Exception:                                     # noqa: BLE001
                pass
            time.sleep(self.interval)

    def start(self):
        if self.ok:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()

    def stop(self):
        if self.ok:
            self._stop.set()
            self._t.join(timeout=2.0)

    def energy(self, t0, t1):
        if not self.ok:
            return float("nan"), float("nan"), 0
        pts = [(t, w) for t, w in self.samples if t0 <= t <= t1]
        if len(pts) < 2:
            return float("nan"), float("nan"), len(pts)
        ts = np.array([p[0] for p in pts])
        ws = np.array([p[1] for p in pts])
        trap = getattr(np, "trapezoid", None) or np.trapz
        return float(trap(ws, ts)), float(ws.mean()), len(pts)


# --------------------------------------------------------------------------
class NodeTracer:
    """Count how often each node is sampled across the run.

    One int32 slot per paper (121.75M -> ~490 MB). Every batch does
    count[node_id] += 1 and records what fraction of its nodes were seen
    in the previous 1 / 10 / 100 batches. Answers: how much reuse does
    batching give, and is there a hot set worth caching or partitioning
    around?
    """

    def __init__(self, num_nodes, windows=(1, 10, 100)):
        self.count = np.zeros(num_nodes, dtype=np.int32)
        self.last_seen = np.full(num_nodes, -1, dtype=np.int32)
        self.windows = windows
        self.rows = []

    def record(self, batch_idx, node_id):
        idx = node_id.numpy()
        n = idx.size
        seen = self.last_seen[idx]
        row = dict(batch=batch_idx, nodes=n,
                   new=int((seen < 0).sum()))
        for w in self.windows:
            row[f"in_prev_{w}"] = int(((seen >= 0) &
                                       (seen >= batch_idx - w)).sum())
        self.rows.append(row)
        np.add.at(self.count, idx, 1)
        self.last_seen[idx] = batch_idx

    def save(self, out_dir, tag):
        np.save(os.path.join(out_dir, f"{tag}_nodecount.npy"), self.count)
        with open(os.path.join(out_dir, f"{tag}_batchreuse.csv"),
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        touched = int((self.count > 0).sum())
        total = int(self.count.sum())
        print(f"[trace] {touched:,} unique nodes touched, "
              f"{total:,} total touches, reuse x{total/max(touched,1):.2f}")


# --------------------------------------------------------------------------
def load_graph(csr_dir, root, in_memory=False):
    """Load the graph and features.

    in_memory=False  features stay on NFS, paged in per access (slow)
    in_memory=True   whole 187 GB table read into RAM once (fast after)
    """
    t0 = time.time()
    # CSR always resident: it is only ~22 GB and gets hammered by sampling
    rowptr = torch.from_numpy(
        np.load(os.path.join(csr_dir, "rowptr.npy"))).long()
    col = torch.from_numpy(
        np.load(os.path.join(csr_dir, "col.npy"))).long()
    print(f"[graph] CSR resident: rowptr {rowptr.numel():,} "
          f"col {col.numel():,}  ({time.time()-t0:.0f}s)")

    feat_path = os.path.join(root, "mag240m_kddcup2021", "processed",
                             "paper", "node_feat.npy")

    if in_memory:
        print("[graph] reading 187 GB feature table into RAM, "
              "this takes a while ...", flush=True)
        t0 = time.time()
        feat = np.load(feat_path)          # no mmap_mode = full read
        print(f"[graph] features RESIDENT {feat.shape} {feat.dtype} "
              f"({feat.nbytes/1e9:.1f} GB in RAM, {(time.time()-t0)/60:.1f} min)")
    else:
        feat = np.load(feat_path, mmap_mode="r")
        print(f"[graph] features MEMMAPPED {feat.shape} {feat.dtype} "
              f"({feat.nbytes/1e9:.1f} GB on disk)")

    return rowptr, col, feat

def sample_batch(rowptr, col, seeds, fanout):
    """One hop-by-hop neighbour sample. This is phase 1.

    Given seed papers and a fanout like [15, 10]:
      hop 1: pick 15 neighbours of each seed
      hop 2: pick 10 neighbours of each of those

    So 100 seeds with fanout [15, 10] touches at most
    100 + 100*15 + 100*15*10 = 16,600 nodes. Bounded, which is the whole
    reason GraphSAGE samples rather than using every neighbour.

    Returns:
      node_id     global paper IDs this batch needs, seeds first
      edge_index  (2, num_edges) using LOCAL indices 0..len(node_id)-1

    The renumbering matters: the model works on a small dense batch, so
    global ID 98,234,113 becomes local index 42. node_id is the mapping
    back, and it is what the gather phase uses to index the feature file.
    """
    import pyg_lib
    out = pyg_lib.sampler.neighbor_sample(
        rowptr, col, seeds, fanout,
        csc=False, replace=False, directed=True,
        disjoint=False, return_edge_id=True,
    )

    row, col_out, node_id = out[0], out[1], out[2]
    edge_index = torch.stack([row, col_out], dim=0)
    return node_id, edge_index


# --------------------------------------------------------------------------
def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("asked for cuda but no GPU is visible. "
                         "Are you on a login node? Pass --device cpu "
                         "if you meant to run on CPU.")
    dev = torch.device(args.device)
    # match torch's thread pool to the Slurm allocation; otherwise torch
    # defaults to all 48 cores even when the job was given 8 or 32
    n_thr = int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads()))
    torch.set_num_threads(n_thr)
    print(f"[run] torch threads = {n_thr}")
    fanout = [int(v) for v in args.fanout.split(",")]
    if len(fanout) != args.layers:
        raise SystemExit(f"--fanout has {len(fanout)} hops but "
                         f"--layers is {args.layers}; they must match")

    print(f"[run] device={dev}  seeds={args.seeds}  fanout={fanout}")

    rowptr, col, feat = load_graph(args.csr, args.root, args.in_memory)

    from ogb.lsc import MAG240MDataset
    ds = MAG240MDataset(root=args.root)
    split = ds.get_idx_split("test-dev")
    n_classes = ds.num_classes
    feat_dim = feat.shape[1]

    model = SAGE(feat_dim, args.hidden, n_classes, args.layers).to(dev).eval()

    # deterministic seed selection so runs are comparable
    rng = np.random.default_rng(0)
    if args.seed_mode == "all":
        # every paper is a candidate, each used at most once: no recycling.
        # needs seeds >= batch_size * iters to never repeat a batch.
        n_all = rowptr.numel() - 1
        pool = rng.choice(n_all, size=min(args.seeds, n_all), replace=False)
        print(f"[seeds] all mode: {len(pool):,} seeds drawn once from "
              f"{n_all:,} papers")
    elif args.seed_mode == "hub":
        deg = (rowptr[1:] - rowptr[:-1]).numpy()
        order = np.argsort(deg[split])[::-1][:args.seeds]
        pool = split[order]
        d = deg[pool]
        print(f"[seeds] hub mode: degree {d.min():,} to {d.max():,}, "
              f"mean {d.mean():.0f}")
    else:
        pool = split if args.seeds >= len(split) else \
            rng.choice(split, size=args.seeds, replace=False)
    pool = torch.from_numpy(np.sort(np.asarray(pool))).long()

    batches = [pool[i:i + args.batch_size]
               for i in range(0, len(pool), args.batch_size)]
    total = args.warmup + args.iters
    print(f"[run] {len(batches)} batches available, "
          f"{args.warmup} warmup + {args.iters} measured")

    tracer = NodeTracer(rowptr.numel() - 1) if args.trace_nodes else None

    power = PowerSampler(args.power_interval)
    power.start()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows, t_start, t_end = [], None, None

    with torch.no_grad():
        for i in range(total):
            seeds = batches[i % len(batches)]

            # ---- PHASE 1: SAMPLE (CPU) --------------------------------
            # Walk the CSR to decide which nodes this batch needs.
            # Pointer chasing through rowptr and col. No features yet.
            t0 = time.perf_counter()
            node_id, edge_index = sample_batch(rowptr, col, seeds, fanout)
            t1 = time.perf_counter()

            # tracing happens outside the four timers so it does not
            # pollute the phase breakdown; it costs a few ms per batch
            if tracer is not None:
                sample_s_pre = t1 - t0
                tracer.record(i, node_id)
                t1 = time.perf_counter()
                t0 = t1 - sample_s_pre   # shift so sample_s is unchanged

            # ---- PHASE 2: GATHER (CPU + disk) -------------------------
            # feat[idx] with a scattered index array is the expensive line
            # in this script. Each ID is a random jump into a 175 GB file,
            # so pages fault in from NFS on first touch. This is where
            # "features live in CPU memory, not GPU memory" costs you.
            idx = node_id.numpy()
            block = feat[idx]                     # disk -> RAM, fancy index
            x_cpu = torch.from_numpy(np.ascontiguousarray(block))
            t2 = time.perf_counter()

            # ---- PHASE 3: TRANSFER (PCIe) -----------------------------
            # Still float16 here, so only half the bytes cross the link.
            # synchronize() because CUDA copies are async and without it
            # we would time the request, not the copy.
            x = x_cpu.to(dev, non_blocking=False)
            ei = edge_index.to(dev, non_blocking=False)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            # ---- PHASE 4: COMPUTE (GPU) -------------------------------
            # CUDA events rather than perf_counter: kernel launches return
            # immediately, so a CPU timer here would measure the launch,
            # not the work, and report an impossibly fast GPU.
            if dev.type == "cuda":
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                x = x.float()
                out = model(x, ei)
                e1.record()
                torch.cuda.synchronize()
                compute_s = e0.elapsed_time(e1) / 1000.0
            else:
                x = x.float()
                out = model(x, ei)
                compute_s = time.perf_counter() - t3
            t4 = time.perf_counter()

            if i == 0:
                pred = out[:len(seeds)].argmax(dim=1)
                print(f"[check] batch nodes={node_id.numel():,} from "
                      f"{len(seeds)} seeds  (x{node_id.numel()/len(seeds):.1f})")
                print(f"[check] edges={edge_index.size(1):,}  "
                      f"x={tuple(x.shape)}  out={tuple(out.shape)}")
                print(f"[check] first 10 predicted classes: "
                      f"{pred[:10].tolist()}")

            # Discard warmup iterations. The first few include CUDA context
            # creation, allocator warmup, and cold page cache, so they are
            # several times slower and would skew every median.
            if i < args.warmup:
                continue
            if t_start is None:
                t_start = t0
            t_end = t4

            # The sanity check. Four phases should account for wall clock.
            # If they sum to MORE, something overlapped and is being
            # counted twice. If LESS, time is hiding somewhere unmeasured.
            # Either way the breakdown would be meaningless, so we track it.
            sample_s, gather_s, transfer_s = t1 - t0, t2 - t1, t3 - t2
            wall_s = t4 - t0
            attributed = sample_s + gather_s + transfer_s + compute_s
            rows.append(dict(
                iter=i - args.warmup,
                seeds=len(seeds),
                nodes=int(node_id.numel()),
                edges=int(edge_index.size(1)),
                bytes_moved=int(x_cpu.numel() * x_cpu.element_size()),
                sample_s=sample_s, gather_s=gather_s,
                transfer_s=transfer_s, compute_s=compute_s,
                wall_s=wall_s,
                drift_pct=abs(wall_s - attributed) / wall_s * 100,
            ))

    power.stop()
    if not rows:
        raise SystemExit("no measured iterations")

    # ---------------- sanity gate ----------------
    drift = np.array([r["drift_pct"] for r in rows])
    print("\n" + "=" * 60)
    print(f"attribution drift: median {np.median(drift):.2f}%  "
          f"max {drift.max():.2f}%")
    if np.median(drift) > args.drift_tolerance:
        print("!! phases do not sum to wall clock; the breakdown is suspect")
    else:
        print("attribution OK")

    med = lambda k: float(np.median([r[k] for r in rows]))       # noqa: E731
    e_j, w_mean, n_samp = power.energy(t_start, t_end)
    n_seeds = sum(r["seeds"] for r in rows)
    window = t_end - t_start

    summary = dict(
        tag=args.tag or f"mag240m_s{args.seeds}_b{args.batch_size}_L{args.layers}",
        seeds_total=n_seeds, batch_size=args.batch_size,
        fanout=args.fanout, layers=args.layers, hidden=args.hidden,
        iters=len(rows),
        sample_ms=med("sample_s") * 1000,
        gather_ms=med("gather_s") * 1000,
        transfer_ms=med("transfer_s") * 1000,
        compute_ms=med("compute_s") * 1000,
        wall_ms=med("wall_s") * 1000,
        drift_pct=float(np.median(drift)),
        nodes_per_batch=med("nodes"),
        mb_moved=med("bytes_moved") / 1e6,
        throughput_qps=n_seeds / window if window else float("nan"),
        mean_power_w=w_mean,
        energy_j=e_j,
        energy_mj_per_seed=(e_j / n_seeds * 1000) if n_seeds else float("nan"),
        power_samples=n_samp,
        peak_gpu_gb=(torch.cuda.max_memory_allocated() / 1e9
                     if dev.type == "cuda" else 0.0),
    )

    print("-" * 60)
    for k, v in summary.items():
        print(f"{k:<20}: {v:.4f}" if isinstance(v, float) else f"{k:<20}: {v}")
    print("-" * 60)
    tot = sum(summary[f"{p}_ms"] for p in
              ("sample", "gather", "transfer", "compute"))
    for p in ("sample", "gather", "transfer", "compute"):
        pct = summary[f"{p}_ms"] / tot * 100
        print(f"  {p:<9} {pct:5.1f}%  {'#' * int(pct / 2)}")
    print("=" * 60)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{summary['tag']}_periter.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sp = os.path.join(args.out, "summary.csv")
    exists = os.path.exists(sp)
    with open(sp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        if not exists:
            w.writeheader()
        w.writerow(summary)
    if tracer is not None:
        tracer.save(args.out, summary["tag"])
    print(f"wrote results to {args.out}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/share/gupta/ak2877/gnn/data")
    p.add_argument("--csr", default="/share/gupta/ak2877/gnn/csr")
    p.add_argument("--seeds", type=int, default=100,
                   help="how many papers to classify; start at 100")
    p.add_argument("--device", default="cuda",
                   help="cuda or cpu; cpu makes transfer zero and moves "
                   "compute off the GPU")
    p.add_argument("--fanout", default="5",
                   help="neighbours per hop, comma separated; "
                        "must have exactly --layers entries")
    p.add_argument("--seed-mode", default="random",
                   choices=["random", "hub", "all"],
                   help="hub = seed on the highest-degree papers, the "
                        "worst case for any real query distribution; "
                        "all = draw seeds once from every paper, no "
                        "recycling (set --seeds >= batch_size * iters)")    
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--in-memory", action="store_true",
                   help="load the whole feature table into RAM")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--power-interval", type=float, default=0.05)
    p.add_argument("--drift-tolerance", type=float, default=5.0)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="")
    p.add_argument("--trace-nodes", action="store_true",
                   help="count every sampled node across the run and "
                        "record per-batch reuse; writes <tag>_nodecount.npy "
                        "(~490 MB) and <tag>_batchreuse.csv")
    run(p.parse_args())


if __name__ == "__main__":
    main()
