"""GraphSAGE + METIS when EVERY batch needs a different shard: three ways to bring the shard in (GH200, 10 Nov).

Batches rotate through the shards (batch i uses shard i mod k; the seeds of a batch all come from that shard), so
no batch finds its shard already resident. The table is the METIS-ordered one, so a shard is one contiguous block of
host RAM and bringing it in is a plain copy.

  series    one GPU buffer. Copy the batch's shard RAM -> GPU, then run the batch.
  overlap   one GPU buffer. Start the copy on a side CUDA stream, run the CPU steps (sample, filter, fetch) meanwhile,
            wait for the copy just before the in-shard gather.
  pipeline  two GPU buffers. At the start of batch i, start copying the shard of batch i+1 into the OTHER buffer, so that
            copy overlaps the whole of batch i (CPU steps, GPU gather and compute). Batch i waits for its own shard
            (copied during batch i-1) only if that copy has not finished when its in-shard gather starts.

Per batch: wall_s (start of the batch to the end of compute), copy_s (the copy of this batch's shard, CUDA events),
wait_s (time the batch was blocked waiting for its shard), the usual steps, MB fetched, fraction of rows in the shard.
One summary row per (mode, fanout): medians, plus elapsed_ms_per_batch = total elapsed / batches (the batches run
back to back, so this is the rate a server would see). Accuracy: one untimed pass per fanout over all validation papers
through the same gather path (identical for the three modes), written on each row of that fanout.

    python shard_pipeline.py --weights ... --root ... --csr ... --hetero-dir ... --metis DIR --k 16 \
        --feat full_feat_metis_k16.npy --row-map rowmap_k16.npy --chunk-bounds bounds_k16.npy --in-memory \
        --modes series,overlap,pipeline --rounds 63 --out results_pipeline_gh200 --tag gh200_pipe
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

import hetero_common as hc


def stream_sync():
    torch.cuda.current_stream().synchronize()
    return time.perf_counter()


class Buffers:
    """n_buf GPU buffers, each large enough for the largest shard, and the shards' host slices (registered once)."""

    def __init__(self, feat, bounds, dev, n_buf):
        self.feat, self.bounds, self.dev = feat, bounds, dev
        nmax = int(np.diff(bounds).max())
        self.buf = [torch.empty((nmax, feat.shape[1]), dtype=torch.float16, device=dev) for _ in range(n_buf)]
        self.side = torch.cuda.Stream()
        self.host, self.mode = {}, {}

    def host_slice(self, p):
        if p not in self.host:
            t = torch.from_numpy(self.feat[int(self.bounds[p]):int(self.bounds[p + 1])])
            t0 = time.perf_counter()
            try:
                r = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
                ok = getattr(r, "value", r) == 0 or str(r).lower().endswith("success")
                self.mode[p] = "pinned (cudaHostRegister)" if ok else f"pageable (register returned {r})"
            except Exception as e:  # noqa: BLE001
                self.mode[p] = f"pageable (register failed: {e})"
            print(f"[host] shard {p}: {t.shape[0]:,} rows, {t.numel() * 2 / 1e9:.1f} GB, {self.mode[p]}, "
                  f"registered in {time.perf_counter() - t0:.2f} s", flush=True)
            self.host[p] = t
        return self.host[p]

    def copy_async(self, p, b):
        """Start copying shard p into buffer b on the side stream; returns the two timing events."""
        src = self.host_slice(p)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.side):
            ev0.record()
            self.buf[b][:src.shape[0]].copy_(src, non_blocking=True)
            ev1.record()
        return ev0, ev1

    def copy_blocking(self, p, b):
        src = self.host_slice(p)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.buf[b][:src.shape[0]].copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t0


def run_batch(g, feat, part, rowmap, bounds, p, xbuf, seeds, fanout, model, dev, start_hook=None, ev_own=None):
    """One batch whose seeds come from shard p, whose rows are (or will be) in GPU buffer xbuf.
    start_hook(): called at the start of the batch; returns (seconds it blocked, events of this batch's own copy or None).
    ev_own: events of this batch's own shard copy when it was started earlier (pipeline)."""
    t0 = time.perf_counter()
    blocked = 0.0
    if start_hook is not None:
        blocked, ev_new = start_hook()
        if ev_new is not None:
            ev_own = ev_new
    ts = time.perf_counter()
    node_id, edge_index = hc.sample_hetero(g, seeds, fanout)
    t1 = time.perf_counter()

    b0 = int(bounds[p])
    gid = {nt: node_id[nt].numpy() + g.offset[nt] for nt in hc.NODE_TYPES}
    local = {nt: part[gid[nt]] == p for nt in hc.NODE_TYPES}
    n_all = sum(v.size for v in gid.values())
    n_local = sum(int(v.sum()) for v in local.values())
    assert local["paper"][:len(seeds)].all(), "a seed is not in this batch's shard"
    loc_rows = {nt: torch.from_numpy(rowmap[gid[nt][local[nt]]] - b0) for nt in hc.NODE_TYPES}
    loc_pos = {nt: torch.from_numpy(np.flatnonzero(local[nt])) for nt in hc.NODE_TYPES}
    rem_pos = {nt: torch.from_numpy(np.flatnonzero(~local[nt])) for nt in hc.NODE_TYPES}
    rem_gid = {nt: gid[nt][~local[nt]] for nt in hc.NODE_TYPES}
    t2 = time.perf_counter()

    rem_cpu = {nt: torch.from_numpy(np.ascontiguousarray(feat[rowmap[rem_gid[nt]]]))
               if rem_gid[nt].size else torch.empty(0, feat.shape[1], dtype=torch.float16) for nt in hc.NODE_TYPES}
    t3 = time.perf_counter()

    ei = {k: v.to(dev) for k, v in edge_index.items()}
    loc_rows_d = {k: v.to(dev) for k, v in loc_rows.items()}
    loc_pos_d = {k: v.to(dev) for k, v in loc_pos.items()}
    rem_pos_d = {k: v.to(dev) for k, v in rem_pos.items()}
    rem_d = {k: v.to(dev) for k, v in rem_cpu.items()}
    t4 = stream_sync()

    wait_s = copy_s = 0.0
    t4b = t4
    if ev_own is not None:                      # the shard must be on the GPU from here on
        ev_own[1].synchronize()
        t4b = time.perf_counter()
        wait_s = t4b - t4
        copy_s = ev_own[0].elapsed_time(ev_own[1]) / 1000.0

    x = {}
    for nt in hc.NODE_TYPES:
        xt = torch.empty((gid[nt].size, feat.shape[1]), dtype=torch.float16, device=dev)
        if loc_pos_d[nt].numel():
            xt[loc_pos_d[nt]] = xbuf[loc_rows_d[nt]]
        if rem_pos_d[nt].numel():
            xt[rem_pos_d[nt]] = rem_d[nt]
        x[nt] = xt
    t5 = stream_sync()

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    out = model({k: v.float() for k, v in x.items()}, ei)
    e1.record()
    t6 = stream_sync()
    compute_s = e0.elapsed_time(e1) / 1000.0

    mb = (sum(v.numel() * v.element_size() for v in rem_cpu.values()) + sum(v.numel() * 8 for v in loc_rows.values())
          + sum(v.numel() * 8 for v in loc_pos.values()) + sum(v.numel() * 8 for v in rem_pos.values())
          + sum(v.numel() * v.element_size() for v in edge_index.values())) / 1e6
    row = dict(shard=p, seeds=len(seeds), nodes_sampled=n_all, frac_local=n_local / n_all, mb_transfer=mb,
               copy_block_s=blocked, copy_s=copy_s if ev_own is not None else blocked, wait_s=wait_s,
               sample_s=t1 - ts, filter_s=t2 - t1, gather_remote_s=t3 - t2, transfer_s=t4 - t3,
               gather_local_s=t5 - t4b, compute_s=compute_s, wall_s=t6 - t0)
    return out[:len(seeds)], row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csr", required=True)
    ap.add_argument("--hetero-dir", required=True)
    ap.add_argument("--feat", required=True, help="METIS-ordered table for this k")
    ap.add_argument("--row-map", required=True)
    ap.add_argument("--chunk-bounds", required=True)
    ap.add_argument("--metis", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--fanouts", default="40/20,5/5,10/5,15/10,25/15,60/30")
    ap.add_argument("--modes", default="series,overlap,pipeline")
    ap.add_argument("--rounds", type=int, default=63, help="timed rounds over the k shards (63 x 16 = 1,008 batches)")
    ap.add_argument("--warm-rounds", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=20_000_000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--out", default="results_pipeline")
    ap.add_argument("--summary", default="")
    ap.add_argument("--tag", default="pipe")
    a = ap.parse_args()

    dev = torch.device("cuda")
    fanouts = [[int(v) for v in f.split("/")] for f in a.fanouts.split(",")]
    modes = [m for m in a.modes.split(",") if m]
    assert all(m in ("series", "overlap", "pipeline") for m in modes), modes
    os.makedirs(a.out, exist_ok=True)
    sp = a.summary or os.path.join(a.out, "summary_pipeline.csv")

    g = hc.HeteroGraph(a.csr, a.hetero_dir)
    feat = hc.load_features(a.feat, a.in_memory)
    from ogb.lsc import MAG240MDataset
    ds = MAG240MDataset(root=a.root)
    n_classes = ds.num_classes
    labels = np.nan_to_num(ds.paper_label, nan=-1).astype(np.int64)
    val_idx = np.asarray(ds.get_idx_split("valid"))

    part = np.load(os.path.join(a.metis, f"part_k{a.k}.npy")).astype(np.int64)
    rowmap = np.load(a.row_map)
    bounds = np.load(a.chunk_bounds).astype(np.int64)
    N = g.num["paper"] + g.num["author"] + g.num["institution"]
    assert part.size == N and rowmap.size == N and bounds.size == a.k + 1 and bounds[-1] == N
    chk = np.random.default_rng(3).choice(N, size=1000, replace=False)
    assert np.all(part[chk] == np.searchsorted(bounds, rowmap[chk], side="right") - 1), "row map does not agree with the part file"

    model = hc.RSAGE(feat.shape[1], a.hidden, n_classes, 2).to(dev).eval()
    ck = torch.load(a.weights, map_location=dev)
    model.load_state_dict(ck["model"])

    # same seed pool and the same batches as shard_infer.py (which skips 2 warm-up batches per shard)
    rng = np.random.default_rng(0)
    P = g.num["paper"]
    pool = np.sort(rng.choice(P, size=min(a.seeds, P), replace=False))
    pool_part = part[pool]
    need = 2 + a.warm_rounds + a.rounds
    per_shard = {}
    for p in range(a.k):
        sp_ = pool[pool_part == p]
        bl = [torch.from_numpy(sp_[i:i + a.batch_size]).long() for i in range(0, min(sp_.size, need * a.batch_size), a.batch_size)]
        bl = [b for b in bl if b.numel() == a.batch_size]
        assert len(bl) >= need, f"shard {p} has only {len(bl)} full batches, {need} needed"
        per_shard[p] = bl[2:]
    order = [(r, p) for r in range(a.warm_rounds + a.rounds) for p in range(a.k)]
    print(f"[plan] k={a.k}, {len(order)} batches per (mode, fanout): {a.warm_rounds} warm-up round(s) + {a.rounds} timed, "
          f"shards rotate every batch; modes {modes}", flush=True)

    B = Buffers(feat, bounds, dev, 2 if "pipeline" in modes else 1)
    print(f"[gpu] {len(B.buf)} buffer(s) of {B.buf[0].shape[0] * feat.shape[1] * 2 / 1e9:.1f} GB", flush=True)
    for p in range(a.k):                       # register every shard's host slice once, before any timing
        B.host_slice(p)

    exists = os.path.exists(sp)
    writer = None
    with torch.no_grad(), open(sp, "a", newline="") as fh:
        for f in fanouts:
            fs = "x".join(map(str, f))
            acc = float("nan")
            n_eval = 0
            if not a.no_val:                   # one accuracy pass per fanout, shard by shard, same gather path
                correct = 0
                for p in range(a.k):
                    B.copy_blocking(p, 0)
                    vp = val_idx[part[val_idx] == p]
                    for i in range(0, vp.size, a.batch_size):
                        vb = torch.from_numpy(vp[i:i + a.batch_size]).long()
                        out, _ = run_batch(g, feat, part, rowmap, bounds, p, B.buf[0], vb, f, model, dev)
                        correct += int((out.argmax(1).cpu().numpy() == labels[vb.numpy()]).sum())
                        n_eval += vb.numel()
                acc = correct / max(n_eval, 1)
                print(f"[val] fanout {f}: {acc * 100:.2f}% on {n_eval:,} papers", flush=True)

            for mode in modes:
                rows = []
                ev_own = None
                if mode == "pipeline":         # the first batch's shard goes in before the clock starts
                    ev_own = B.copy_async(order[0][1], 0)
                t_first = None
                for i, (r, p) in enumerate(order):
                    b = i % 2 if mode == "pipeline" else 0
                    nxt = {}

                    def hook(i=i, p=p, b=b, mode=mode, nxt=nxt):
                        if mode == "series":
                            return B.copy_blocking(p, 0), None
                        if mode == "overlap":
                            return 0.0, B.copy_async(p, 0)
                        if i + 1 < len(order):   # pipeline: start the NEXT batch's shard into the other buffer
                            nxt["ev"] = B.copy_async(order[i + 1][1], 1 - b)
                        return 0.0, None

                    if r == a.warm_rounds and p == 0:
                        t_first = time.perf_counter()
                    _, row = run_batch(g, feat, part, rowmap, bounds, p, B.buf[b], per_shard[p][r], f, model, dev,
                                       start_hook=hook, ev_own=ev_own if mode == "pipeline" else None)
                    if mode == "pipeline":
                        ev_own = nxt.get("ev")
                    if r >= a.warm_rounds:
                        row.update(round=r - a.warm_rounds, mode=mode)
                        rows.append(row)
                elapsed = time.perf_counter() - t_first
                torch.cuda.synchronize()

                tag = f"{a.tag}_k{a.k}_{mode}_f{fs}"
                with open(os.path.join(a.out, tag + "_periter.csv"), "w", newline="") as ph:
                    pw = csv.DictWriter(ph, fieldnames=list(rows[0])); pw.writeheader(); pw.writerows(rows)
                med = lambda key: float(np.median([x[key] for x in rows]))  # noqa: E731
                s = dict(tag=tag, k=a.k, mode=mode, fanout=",".join(map(str, f)), batches=len(rows),
                         buffers=2 if mode == "pipeline" else 1,
                         wall_ms=med("wall_s") * 1e3, elapsed_ms_per_batch=elapsed / len(rows) * 1e3,
                         copy_ms=med("copy_s") * 1e3, wait_ms=med("wait_s") * 1e3,
                         sample_ms=med("sample_s") * 1e3, filter_ms=med("filter_s") * 1e3,
                         gather_remote_ms=med("gather_remote_s") * 1e3, transfer_ms=med("transfer_s") * 1e3,
                         gather_local_ms=med("gather_local_s") * 1e3, compute_ms=med("compute_s") * 1e3,
                         nodes_sampled=med("nodes_sampled"), frac_local=med("frac_local"), mb_transfer=med("mb_transfer"),
                         shard_gb_mean=float(np.diff(bounds).mean() * feat.shape[1] * 2 / 1e9),
                         val_papers=n_eval, val_acc=acc, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(s))
                    if not exists:
                        writer.writeheader()
                writer.writerow(s)
                fh.flush()
                print(f"[summary] {tag}: wall {s['wall_ms']:.1f} ms, elapsed/batch {s['elapsed_ms_per_batch']:.1f} ms, "
                      f"copy {s['copy_ms']:.1f} ms, wait {s['wait_ms']:.1f} ms, local {s['frac_local'] * 100:.1f}%, "
                      f"acc {acc * 100:.2f}%", flush=True)
    print(f"[done] -> {sp}", flush=True)


if __name__ == "__main__":
    main()
