"""METIS-shard inference: one shard resident in GPU memory, seeds routed to it.

Baselines this produces (Michael, 27 Oct): "METIS-based shard filtering,
followed by GraphSAGE on the METIS shard", on the H100 (PCIe) and the GH200
(NVLink-C2C). The node-filtering baselines stay in sage_mag240m.py, unchanged.

Per k (part_k{k}.npy from metis_partition.py):

  seeds     the same fresh random pool as sage_mag240m.py --seed-mode all
            (default_rng(0), drawn once from all papers, sorted), routed to
            shards by their METIS label: shard p serves the pool's seeds whose
            label is p, in pool order, 1,024 per batch.

  for each shard p:
    load    shard p's feature rows: CPU gather from the in-RAM table, then
            copy to GPU memory, in chunks. Both parts timed (load_gather_s,
            load_transfer_s). Counted once per shard.
    batches for every fanout and both variants:
      drop  sampled nodes outside p are removed (with their edges); all
            remaining rows come from the resident shard, gathered on the GPU
      fetch sampled nodes outside p are gathered on the CPU from the full
            table and sent over the link; rows inside p come from the shard
    accuracy  the validation papers whose label is p, same variant, untimed

  phases per batch (sum = wall, checked like the baseline harness):
    sample         hetero neighbour sample on the CPU (pyg_lib)
    filter         CPU: shard lookup per sampled node, keep masks, edge remap
    gather_remote  CPU fancy-index of out-of-shard rows (fetch only, else 0)
    transfer       to GPU: local row indices, edges, remote rows (fetch)
    gather_local   GPU index into the resident shard
    compute        R-GraphSAGE forward (CUDA events)

Output: <out>/<tag>_<variant>_f<fanout>_periter.csv and one row per
(variant, fanout) appended to --summary.

    python shard_infer.py --k 16 --in-memory --weights best.pt \
        --root ... --csr ... --hetero-dir ... --feat ... --metis ... \
        --out results_shard --summary results_shard/summary_shard_gh200.csv
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

import hetero_common as hc

VARIANTS = ("drop", "fetch")


def variant_m(v):
    """'drop' -> 1, 'fetch' / 'coherent' / 'cpuonly' -> None (all rows used), 'topM' -> M (home shard + M-1 most-hit shards).
    fetchload: fetch, with the shard copied RAM -> GPU inside every batch (the shard movement is in the batch time).
    coherent: the shard stays in host RAM and the GPU gathers its rows in place over the link (no copy into HBM).
    cpuonly: every row gathered by the CPU from RAM and the model run on the CPU (same batches, for comparison)."""
    if v == "drop":
        return 1
    if v in ("fetch", "fetchload", "coherent", "cpuonly"):
        return None
    assert v.startswith("top") and v[3:].isdigit() and int(v[3:]) >= 2, v
    return int(v[3:])


def now_sync(dev, stream_only=False):
    """Wall clock after the GPU work is done: the whole device, or only the current stream (so an overlapped
    shard copy on its side stream is not waited for)."""
    if dev.type == "cuda":
        if stream_only:
            torch.cuda.current_stream().synchronize()
        else:
            torch.cuda.synchronize()
    return time.perf_counter()


class Shard:
    """Resident shard p: feature rows on the device plus a global->row map."""

    def __init__(self, n_total, dim, dev, keep_host=False, rowmap=None, bounds=None, coherent=False, register=True):
        self.row = np.full(n_total, -1, dtype=np.int64)   # global id -> shard row
        self.dim, self.dev, self.x, self.p = dim, dev, None, None
        self.keep_host, self.host = keep_host, None       # host source of the shard, for --swap-per-batch
        self.rowmap, self.bounds = rowmap, bounds         # METIS-ordered table: shard p = rows bounds[p]..bounds[p+1]-1
        self.registered = None
        self.coherent, self.xh = coherent, None           # coherent: GPU-side view of the shard's host memory (no HBM copy)
        self.register = register                          # False: another Shard object registers this host memory

    def load(self, feat, part, p, chunk=2_000_000):
        if self.x is not None or self.xh is not None:
            self.row[self.ids] = -1
            del self.x
            self.x = None
            self.xh = None
            self._unregister()
            self.host = None
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        self.ids = np.flatnonzero(part == p)
        n = self.ids.size
        t_gather = t_transfer = 0.0
        if self.coherent:
            if self.bounds is not None:                       # ordered table: the shard is a slice of the in-RAM table
                b0, b1 = int(self.bounds[p]), int(self.bounds[p + 1])
                assert b1 - b0 == n
                host = torch.from_numpy(feat[b0:b1])
            else:                                             # original table: assemble the shard's rows once (untimed)
                b0 = None
                host = torch.empty((n, self.dim), dtype=torch.float16)
                for i in range(0, n, chunk):
                    j = min(i + chunk, n)
                    host[i:j] = torch.from_numpy(np.ascontiguousarray(feat[self.ids[i:j]]))
            t0 = time.perf_counter()
            self.host = host
            self.xh = self._map_host(self.host)
            self.x = None
            t_transfer = now_sync(self.dev) - t0           # time to make the host slice GPU-visible (no bytes copied)
            chk = np.random.default_rng(5).choice(n, size=min(2000, n), replace=False)
            idx = torch.from_numpy(chk).to(self.dev)
            got = self.xh[idx].cpu()
            assert torch.equal(got, self.host[torch.from_numpy(chk)]), "coherent view does not read the host rows correctly"
            self.row[self.ids] = (self.rowmap[self.ids] - b0) if b0 is not None else np.arange(n, dtype=np.int64)
            self.p = p
            return n, 0.0, t_transfer
        self.x = torch.empty((n, self.dim), dtype=torch.float16, device=self.dev)
        if self.bounds is not None:
            # ordered table: the shard is one contiguous slice of the in-RAM table; no row assembly, no second copy
            b0, b1 = int(self.bounds[p]), int(self.bounds[p + 1])
            assert b1 - b0 == n, f"shard {p}: bounds give {b1 - b0} rows, part file gives {n}"
            src = torch.from_numpy(feat[b0:b1])                     # a view, nothing is copied on the host
            if self.keep_host:
                self.host = src
                if self.register:
                    self._register(src)
                else:
                    self.host_mode = "host slice registered by the coherent view"
            for i in range(0, n, chunk):
                j = min(i + chunk, n)
                t1 = now_sync(self.dev)
                self.x[i:j].copy_(src[i:j], non_blocking=False)
                t2 = now_sync(self.dev)
                t_transfer += t2 - t1
            self.row[self.ids] = self.rowmap[self.ids] - b0
        else:
            if self.keep_host:
                self.host = torch.empty((n, self.dim), dtype=torch.float16, pin_memory=(self.dev.type == "cuda"))
            for i in range(0, n, chunk):
                j = min(i + chunk, n)
                t0 = time.perf_counter()
                block = torch.from_numpy(np.ascontiguousarray(feat[self.ids[i:j]]))
                if self.keep_host:
                    self.host[i:j].copy_(block)
                t1 = now_sync(self.dev)
                self.x[i:j].copy_(block.to(self.dev, non_blocking=False))
                t2 = now_sync(self.dev)
                t_gather += t1 - t0
                t_transfer += t2 - t1
            self.row[self.ids] = np.arange(n, dtype=np.int64)
        self.p = p
        return n, t_gather, t_transfer

    def _map_host(self, t):
        """Return a CUDA tensor that aliases host tensor t: page-lock + map it (cudaHostRegisterMapped) and wrap the
        device pointer; on a coherent system the GPU then reads the host rows over the link on demand."""
        import ctypes
        import glob
        nbytes = t.numel() * t.element_size()
        rt = torch.cuda.cudart()
        r = rt.cudaHostRegister(t.data_ptr(), nbytes, 2)         # 2 = cudaHostRegisterMapped
        code = getattr(r, "value", r)
        reg_ok = code == 0 or str(r).lower().endswith("success")
        dev_ptr = t.data_ptr()
        self.host_mode = "coherent: host pointer used directly (not registered)"
        if reg_ok:
            self.registered = t
            libs = glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*.so*")) or ["libcudart.so"]
            try:
                lib = ctypes.CDLL(libs[0])
                dp = ctypes.c_void_p()
                e = lib.cudaHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(t.data_ptr()), ctypes.c_uint(0))
                if e == 0 and dp.value:
                    dev_ptr = dp.value
                    self.host_mode = "coherent: cudaHostRegisterMapped + cudaHostGetDevicePointer"
                else:
                    self.host_mode = f"coherent: registered, GetDevicePointer returned {e}, host pointer used"
            except Exception as ex:  # noqa: BLE001
                self.host_mode = f"coherent: registered, host pointer used ({ex})"

        class _View:
            __cuda_array_interface__ = dict(shape=tuple(t.shape), typestr="<f2", data=(dev_ptr, False), version=3,
                                            strides=None, descr=[("", "<f2")])
        return torch.as_tensor(_View(), device=self.dev)

    def _register(self, t):
        """Page-lock the shard's slice of the in-RAM table (cudaHostRegister) so the swap is a pinned copy; no extra memory."""
        if self.dev.type != "cuda":
            return
        try:
            rt = torch.cuda.cudart()
            r = rt.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
            code = getattr(r, "value", r)
            if code == 0 or str(r).lower().endswith("success"):
                self.registered = t
                self.host_mode = "pinned (cudaHostRegister)"
            else:
                self.host_mode = f"pageable (cudaHostRegister returned {r})"
        except Exception as e:  # noqa: BLE001
            self.host_mode = f"pageable (cudaHostRegister failed: {e})"

    def _unregister(self):
        if self.registered is not None:
            try:
                torch.cuda.cudart().cudaHostUnregister(self.registered.data_ptr())
            except Exception:  # noqa: BLE001
                pass
            self.registered = None

    def swap(self):
        """Copy the whole resident shard from host memory to the device again: one shard swap."""
        t0 = now_sync(self.dev)
        self.x.copy_(self.host, non_blocking=True)
        return now_sync(self.dev) - t0

    def swap_start(self):
        """Issue the shard copy on a side stream and return at once; (ev0, ev1) time the copy on the GPU."""
        if not hasattr(self, "stream"):
            self.stream = torch.cuda.Stream()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            ev0.record()
            self.x.copy_(self.host, non_blocking=True)
            ev1.record()
        return ev0, ev1


def run_batch(g, feat, shard, part, seeds, fanout, variant, model, dev, swap=False, overlap=False, model_cpu=None):
    rm = shard.rowmap
    if variant == "fetchload":
        variant, swap = "fetch", True
    if variant == "cpuonly":
        return run_batch_cpu(g, feat, shard, part, seeds, fanout, model_cpu, rm)
    """One batch through the shard pipeline. Returns (logits, row dict).
    swap=True: the batch starts by copying the whole resident shard host->device (one shard swap per batch).
    overlap=True: that copy runs on a side stream while the CPU samples, filters and fetches; the batch waits for it
    just before the in-shard gather (swap_wait_s = the part of the copy that was not hidden)."""
    t0 = time.perf_counter()
    swap_s = swap_wait_s = swap_block_s = 0.0
    ev = None
    if swap and overlap:
        ev = shard.swap_start()
    elif swap:
        swap_s = swap_block_s = shard.swap()
    node_id, edge_index = hc.sample_hetero(g, seeds, fanout)
    t1 = time.perf_counter()

    # ---- filter (CPU): which sampled nodes are in the resident shard
    gid = {nt: node_id[nt].numpy() + g.offset[nt] for nt in hc.NODE_TYPES}
    local = {nt: shard.row[gid[nt]] >= 0 for nt in hc.NODE_TYPES}
    n_all = sum(v.size for v in gid.values())
    n_local = sum(int(v.sum()) for v in local.values())
    assert local["paper"][:len(seeds)].all(), "a seed is not in the resident shard"
    m_keep = variant_m(variant)
    node_shard = hist = None
    if m_keep is not None and m_keep > 1:
        node_shard = {nt: part[gid[nt]] for nt in hc.NODE_TYPES}
        hist = np.bincount(np.concatenate([node_shard[nt] for nt in hc.NODE_TYPES]), minlength=shard.k)
        others = [p for p in np.argsort(-hist) if p != shard.p and hist[p] > 0][:m_keep - 1]
        kept_shards = np.zeros(hist.size, dtype=bool); kept_shards[shard.p] = True; kept_shards[others] = True
        keep = {nt: kept_shards[node_shard[nt]] for nt in hc.NODE_TYPES}
    if variant == "drop":
        keep = local
    if m_keep is not None:
        newpos = {nt: np.cumsum(keep[nt]) - 1 for nt in hc.NODE_TYPES}
        ei_cpu = {}
        for et, ei in edge_index.items():
            s, _, d = et
            e = ei.numpy()
            m = keep[s][e[0]] & keep[d][e[1]]
            if m.any():
                ei_cpu[et] = torch.from_numpy(np.stack([newpos[s][e[0][m]], newpos[d][e[1][m]]]))
        n_rows = {nt: int(keep[nt].sum()) for nt in hc.NODE_TYPES}
        if variant == "drop":
            loc_rows = {nt: torch.from_numpy(shard.row[gid[nt][keep[nt]]]) for nt in hc.NODE_TYPES}
            loc_pos = None
            rem_gid = None
        else:  # topM: kept nodes only; home rows from the shard, other kept rows fetched from RAM
            kg = {nt: gid[nt][keep[nt]] for nt in hc.NODE_TYPES}
            kl = {nt: local[nt][keep[nt]] for nt in hc.NODE_TYPES}
            loc_rows = {nt: torch.from_numpy(shard.row[kg[nt][kl[nt]]]) for nt in hc.NODE_TYPES}
            loc_pos = {nt: torch.from_numpy(np.flatnonzero(kl[nt])) for nt in hc.NODE_TYPES}
            rem_pos = {nt: torch.from_numpy(np.flatnonzero(~kl[nt])) for nt in hc.NODE_TYPES}
            rem_gid = {nt: kg[nt][~kl[nt]] for nt in hc.NODE_TYPES}
    else:  # fetch
        ei_cpu = dict(edge_index)
        n_rows = {nt: gid[nt].size for nt in hc.NODE_TYPES}
        loc_rows = {nt: torch.from_numpy(shard.row[gid[nt][local[nt]]]) for nt in hc.NODE_TYPES}
        loc_pos = {nt: torch.from_numpy(np.flatnonzero(local[nt])) for nt in hc.NODE_TYPES}
        rem_pos = {nt: torch.from_numpy(np.flatnonzero(~local[nt])) for nt in hc.NODE_TYPES}
        rem_gid = {nt: gid[nt][~local[nt]] for nt in hc.NODE_TYPES}
    t2 = time.perf_counter()

    # ---- gather_remote (CPU, fetch only)
    rem_cpu = None
    if variant != "drop":
        rem_cpu = {nt: torch.from_numpy(np.ascontiguousarray(feat[rem_gid[nt] if rm is None else rm[rem_gid[nt]]]))
                   if rem_gid[nt].size else torch.empty(0, feat.shape[1], dtype=torch.float16)
                   for nt in hc.NODE_TYPES}
    t3 = time.perf_counter()

    # ---- transfer (to device)
    ei = {k: v.to(dev) for k, v in ei_cpu.items()}
    loc_rows_d = {k: v.to(dev) for k, v in loc_rows.items()}
    if variant != "drop":
        loc_pos_d = {k: v.to(dev) for k, v in loc_pos.items()}
        rem_pos_d = {k: v.to(dev) for k, v in rem_pos.items()}
        rem_d = {k: v.to(dev) for k, v in rem_cpu.items()}
    t4 = now_sync(dev, stream_only=ev is not None)
    t4b = t4
    if ev is not None:                       # wait for the overlapped shard copy before the in-shard gather
        ev[1].synchronize()
        t4b = time.perf_counter()
        swap_wait_s = t4b - t4
        swap_s = ev[0].elapsed_time(ev[1]) / 1000.0

    # ---- gather_local (device): rows of the resident shard (coherent: read in place from host memory over the link)
    src = shard.xh if variant == "coherent" else shard.x
    if variant == "drop":
        x = {nt: src[loc_rows_d[nt]] for nt in hc.NODE_TYPES}
    else:
        x = {}
        for nt in hc.NODE_TYPES:
            xt = torch.empty((n_rows[nt], shard.dim), dtype=torch.float16, device=dev)
            if loc_pos_d[nt].numel():
                xt[loc_pos_d[nt]] = src[loc_rows_d[nt]]
            if rem_pos_d[nt].numel():
                xt[rem_pos_d[nt]] = rem_d[nt]
            x[nt] = xt
    t5 = now_sync(dev)

    # ---- compute
    if dev.type == "cuda":
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        out = model({k: v.float() for k, v in x.items()}, ei)
        e1.record()
        torch.cuda.synchronize()
        compute_s = e0.elapsed_time(e1) / 1000.0
    else:
        out = model({k: v.float() for k, v in x.items()}, ei)
        compute_s = time.perf_counter() - t5
    t6 = time.perf_counter()
    if hist is None:   # drop / fetch: count shards reached outside the timed region
        hist = np.bincount(np.concatenate([part[gid[nt]] for nt in hc.NODE_TYPES]), minlength=shard.k)
    shards_hit = int((hist > 0).sum())

    bytes_xfer = sum(v.numel() * v.element_size() for v in loc_rows.values())
    bytes_xfer += sum(v.numel() * v.element_size() for v in ei_cpu.values())
    if variant != "drop":
        bytes_xfer += sum(v.numel() * v.element_size() for v in rem_cpu.values())
        bytes_xfer += sum(v.numel() * v.element_size() for v in loc_pos.values())
        bytes_xfer += sum(v.numel() * v.element_size() for v in rem_pos.values())
    if variant == "coherent":                 # in-shard rows also cross the link, read in place by the GPU
        bytes_xfer += n_local * shard.dim * 2
    sample_s, filter_s, grem_s, xfer_s, gloc_s = t1 - t0 - swap_block_s, t2 - t1, t3 - t2, t4 - t3, t5 - t4b
    wall_s = t6 - t0
    attributed = (swap_wait_s if ev is not None else swap_s) + sample_s + filter_s + grem_s + xfer_s + gloc_s + compute_s
    row = dict(seeds=len(seeds), nodes_sampled=n_all, nodes_local=n_local,
               nodes_used=sum(n_rows.values()), frac_local=n_local / n_all,
               shards_hit=shards_hit, frac_kept=sum(n_rows.values()) / n_all,
               edges_used=sum(v.size(1) for v in ei_cpu.values()),
               mb_transfer=bytes_xfer / 1e6, swap_s=swap_s, swap_wait_s=swap_wait_s,
               sample_s=sample_s, filter_s=filter_s, gather_remote_s=grem_s,
               transfer_s=xfer_s, gather_local_s=gloc_s, compute_s=compute_s,
               wall_s=wall_s, drift_pct=abs(wall_s - attributed) / wall_s * 100)
    return out[:len(seeds)], row


def run_batch_cpu(g, feat, shard, part, seeds, fanout, model_cpu, rm):
    """cpuonly variant on the same batches: CPU samples, gathers every row from RAM, runs the model on the CPU."""
    t0 = time.perf_counter()
    node_id, edge_index = hc.sample_hetero(g, seeds, fanout)
    t1 = time.perf_counter()
    gid = {nt: node_id[nt].numpy() + g.offset[nt] for nt in hc.NODE_TYPES}
    local = {nt: shard.row[gid[nt]] >= 0 for nt in hc.NODE_TYPES}
    n_all = sum(v.size for v in gid.values())
    n_local = sum(int(v.sum()) for v in local.values())
    t2 = time.perf_counter()
    x = {nt: torch.from_numpy(np.ascontiguousarray(feat[gid[nt] if rm is None else rm[gid[nt]]])) for nt in hc.NODE_TYPES}
    t3 = time.perf_counter()
    out = model_cpu({k: v.float() for k, v in x.items()}, dict(edge_index))
    t6 = time.perf_counter()
    hist = np.bincount(np.concatenate([part[gid[nt]] for nt in hc.NODE_TYPES]), minlength=shard.k)
    mb = sum(v.numel() * v.element_size() for v in x.values()) / 1e6
    sample_s, filter_s, grem_s, compute_s = t1 - t0, t2 - t1, t3 - t2, t6 - t3
    wall_s = t6 - t0
    row = dict(seeds=len(seeds), nodes_sampled=n_all, nodes_local=n_local, nodes_used=n_all, frac_local=n_local / n_all,
               shards_hit=int((hist > 0).sum()), frac_kept=1.0, edges_used=sum(v.size(1) for v in edge_index.values()),
               mb_transfer=mb, swap_s=0.0, swap_wait_s=0.0, sample_s=sample_s, filter_s=filter_s, gather_remote_s=grem_s,
               transfer_s=0.0, gather_local_s=0.0, compute_s=compute_s, wall_s=wall_s,
               drift_pct=abs(wall_s - (sample_s + filter_s + grem_s + compute_s)) / wall_s * 100)
    return out[:len(seeds)], row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csr", required=True)
    ap.add_argument("--hetero-dir", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--metis", required=True, help="directory with part_k{k}.npy")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--shards", default="", help="comma list of shards to visit; default all")
    ap.add_argument("--fanouts", default="40/20,5/5,10/5,15/10,25/15,60/30")
    ap.add_argument("--variants", default="drop,fetch", help="drop, fetch, and/or topM (home shard + M-1 most-hit shards), e.g. drop,top2,top3,fetch")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--batches-per-shard", type=int, default=63,
                    help="timed batches per shard per (fanout, variant); 63 x 16 shards ~ 1,000")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=20_000_000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--val-batches", type=int, default=0, help="limit accuracy pass per shard (smoke)")
    ap.add_argument("--no-val", action="store_true",
                    help="skip the accuracy pass (needed on a subset graph from build_subset.py: its paper ids are renumbered)")
    ap.add_argument("--unsorted-seeds", action="store_true",
                    help="keep the seed pool in default_rng(0) order instead of sorting it, so with --k 1 the batches are "
                         "exactly the ones sage_mag240m.py --seed-mode all uses (GPU-only vs node filtering on the same seeds)")
    ap.add_argument("--row-map", default="", help="METIS-ordered --feat: rowmap_k<k>.npy (global id -> row of the ordered table)")
    ap.add_argument("--chunk-bounds", default="", help="METIS-ordered --feat: bounds_k<k>.npy (first row of each shard, plus N)")
    ap.add_argument("--swap-per-batch", action="store_true",
                    help="copy the whole resident shard from pinned host memory to the GPU before every batch (swap_s)")
    ap.add_argument("--swap-overlap", action="store_true",
                    help="with --swap-per-batch: run that copy on a side stream while the CPU samples, filters and fetches")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results_shard")
    ap.add_argument("--summary", default="")
    ap.add_argument("--tag", default="shard")
    a = ap.parse_args()

    dev = torch.device(a.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no GPU visible")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    fanouts = [[int(v) for v in f.split("/")] for f in a.fanouts.split(",")]
    variants = [v for v in a.variants.split(",") if v]
    for v in variants:
        variant_m(v)   # raises on an unknown variant
    os.makedirs(a.out, exist_ok=True)
    sp = a.summary or os.path.join(a.out, "summary_shard.csv")

    g = hc.HeteroGraph(a.csr, a.hetero_dir)
    feat = hc.load_features(a.feat, a.in_memory)
    from ogb.lsc import MAG240MDataset
    ds = MAG240MDataset(root=a.root)
    n_classes = ds.num_classes
    labels = np.nan_to_num(ds.paper_label, nan=-1).astype(np.int64)
    val_idx = np.asarray(ds.get_idx_split("valid"))

    part = np.load(os.path.join(a.metis, f"part_k{a.k}.npy")).astype(np.int64)
    N = g.num["paper"] + g.num["author"] + g.num["institution"]
    assert part.size == N, f"part file has {part.size} entries, graph has {N}"
    shards = [int(v) for v in a.shards.split(",")] if a.shards else list(range(a.k))

    model = hc.RSAGE(feat.shape[1], a.hidden, n_classes, 2).to(dev).eval()
    ck = torch.load(a.weights, map_location=dev)
    model.load_state_dict(ck["model"])
    print(f"[model] {a.weights}: epoch {ck.get('epoch')}, val acc {ck.get('val_acc', float('nan')):.4f}", flush=True)

    # same seed pool as sage_mag240m.py --seed-mode all
    rng = np.random.default_rng(0)
    P = g.num["paper"]
    pool = rng.choice(P, size=min(a.seeds, P), replace=False)
    if not a.unsorted_seeds:
        pool = np.sort(pool)
    pool_part = part[pool]
    print(f"[seeds] {pool.size:,} fresh seeds; per shard: "
          + ", ".join(str(int((pool_part == p).sum())) for p in shards), flush=True)

    rowmap = bounds = None
    if a.chunk_bounds:
        assert a.row_map, "--chunk-bounds needs --row-map"
        rowmap = np.load(a.row_map)
        bounds = np.load(a.chunk_bounds).astype(np.int64)
        assert rowmap.size == N and bounds.size == a.k + 1 and bounds[0] == 0 and bounds[-1] == N, "row map / bounds do not match --feat / --k"
        chk = np.random.default_rng(3).choice(N, size=1000, replace=False)
        assert np.all(part[chk] == np.searchsorted(bounds, rowmap[chk], side="right") - 1), "row map does not agree with the part file"
        print(f"[ordered table] {a.k} shards contiguous in --feat; shard GB: "
              + ", ".join(f"{(bounds[i + 1] - bounds[i]) * feat.shape[1] * 2 / 1e9:.1f}" for i in range(a.k)), flush=True)
    coherent = "coherent" in variants
    need_host = a.swap_per_batch or "fetchload" in variants
    co_run = coherent and len(variants) > 1
    shard = Shard(N, feat.shape[1], dev, keep_host=need_host, rowmap=rowmap, bounds=bounds,
                  coherent=coherent and len(variants) == 1, register=not (co_run and bounds is not None))
    shard_copy = None
    if coherent and len(variants) > 1:        # coherent together with fetch: a second Shard object holds the host view
        shard_copy = Shard(N, feat.shape[1], dev, rowmap=rowmap, bounds=bounds, coherent=True)
        shard_copy.k = a.k
    model_cpu = None
    if "cpuonly" in variants:
        model_cpu = hc.RSAGE(feat.shape[1], a.hidden, n_classes, 2).eval()
        model_cpu.load_state_dict({k: v.cpu() for k, v in ck["model"].items()})
    shard.k = a.k
    rows = {(v, tuple(f)): [] for v in variants for f in fanouts}
    conf = {(v, tuple(f)): np.zeros((n_classes, n_classes), dtype=np.int64) for v in variants for f in fanouts}
    loads = []
    need = a.warmup + a.batches_per_shard

    with torch.no_grad():
        for p in shards:
            n_rows, lg, lt = shard.load(feat, part, p)
            if shard_copy is not None:
                shard_copy.load(feat, part, p)
                print(f"[shard {p}] coherent view ready: {shard_copy.host_mode}", flush=True)
            loads.append(dict(shard=p, rows=n_rows, gb=n_rows * feat.shape[1] * 2 / 1e9,
                              load_gather_s=lg, load_transfer_s=lt))
            print(f"\n[shard {p}] loaded {n_rows:,} rows ({loads[-1]['gb']:.1f} GB): "
                  f"CPU gather {lg:.1f} s, transfer {lt:.2f} s"
                  + (f", swap source {getattr(shard, 'host_mode', 'n/a')}" if a.swap_per_batch else "")
                  + (f", peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB" if dev.type == "cuda" else ""),
                  flush=True)
            seeds_p = pool[pool_part == p]
            batches = [torch.from_numpy(seeds_p[i:i + a.batch_size]).long()
                       for i in range(0, min(seeds_p.size, need * a.batch_size), a.batch_size)]
            batches = [b for b in batches if b.numel() == a.batch_size]
            if len(batches) < need:
                print(f"[shard {p}] only {len(batches)} full batches available (< {need})", flush=True)
            vp = np.empty(0, dtype=np.int64) if a.no_val else val_idx[part[val_idx] == p]
            vbatches = [torch.from_numpy(vp[i:i + a.batch_size]).long() for i in range(0, vp.size, a.batch_size)]
            if a.val_batches:
                vbatches = vbatches[:a.val_batches]
            for f in fanouts:
                for v in variants:
                    key = (v, tuple(f))
                    sh = shard_copy if (v == "coherent" and shard_copy is not None) else shard
                    for i, b in enumerate(batches):
                        _, r = run_batch(g, feat, sh, part, b, f, v, model, dev, swap=a.swap_per_batch and v == "fetch",
                                         overlap=a.swap_overlap, model_cpu=model_cpu)
                        if i >= a.warmup:
                            r.update(shard=p, iter=i - a.warmup)
                            rows[key].append(r)
                    for b in vbatches:
                        out, _ = run_batch(g, feat, sh, part, b, f, v, model, dev, model_cpu=model_cpu)
                        pred = out.argmax(1).cpu().numpy()
                        y = labels[b.numpy()]
                        np.add.at(conf[key], (y, pred), 1)
                    rr = rows[key][-max(1, len(batches) - a.warmup):]
                    print(f"[shard {p}] {v:<5} fanout {f}: wall {np.median([x['wall_s'] for x in rr])*1e3:7.1f} ms  "
                          f"swap {np.median([x['swap_s'] for x in rr])*1e3:6.1f} ms (wait {np.median([x['swap_wait_s'] for x in rr])*1e3:5.1f})  "
                          f"local {np.median([x['frac_local'] for x in rr])*100:5.1f}%  "
                          f"val papers {sum(x.numel() for x in vbatches):,}", flush=True)

    # ---- summaries
    load_total = sum(x["load_gather_s"] + x["load_transfer_s"] for x in loads)
    with open(os.path.join(a.out, f"{a.tag}_k{a.k}_loads.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(loads[0])); w.writeheader(); w.writerows(loads)
    exists = os.path.exists(sp)
    with open(sp, "a", newline="") as fh:
        w = None
        for (v, f), rr in rows.items():
            if not rr:
                continue
            fs = "x".join(map(str, f))
            with open(os.path.join(a.out, f"{a.tag}_k{a.k}_{v}_f{fs}_periter.csv"), "w", newline="") as ph:
                pw = csv.DictWriter(ph, fieldnames=list(rr[0])); pw.writeheader(); pw.writerows(rr)
            med = lambda key: float(np.median([x[key] for x in rr]))  # noqa: E731
            batch_time = sum(x["wall_s"] for x in rr)
            seeds_total = sum(x["seeds"] for x in rr)
            c = conf[(v, f)]
            tp = np.diag(c).astype(float)
            n_eval = int(c.sum())
            acc = tp.sum() / n_eval if n_eval else float("nan")
            prec = tp / np.maximum(c.sum(0), 1); rec = tp / np.maximum(c.sum(1), 1)
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
            present = c.sum(1) > 0
            s = dict(tag=f"{a.tag}_k{a.k}_{v}_f{fs}", k=a.k, variant=v, fanout=",".join(map(str, f)),
                     shards=len(loads), batches=len(rr), seeds_total=seeds_total,
                     swap_ms=med("swap_s") * 1e3, swap_wait_ms=med("swap_wait_s") * 1e3, sample_ms=med("sample_s") * 1e3, filter_ms=med("filter_s") * 1e3,
                     gather_remote_ms=med("gather_remote_s") * 1e3, transfer_ms=med("transfer_s") * 1e3,
                     gather_local_ms=med("gather_local_s") * 1e3, compute_ms=med("compute_s") * 1e3,
                     wall_ms=med("wall_s") * 1e3, drift_pct=med("drift_pct"),
                     nodes_sampled=med("nodes_sampled"), nodes_used=med("nodes_used"),
                     frac_local=med("frac_local"), mb_transfer=med("mb_transfer"),
                     shards_hit=med("shards_hit"), frac_kept=med("frac_kept"),
                     qps=seeds_total / batch_time,
                     shard_gb_mean=float(np.mean([x["gb"] for x in loads])),
                     load_gather_s_mean=float(np.mean([x["load_gather_s"] for x in loads])),
                     load_transfer_s_mean=float(np.mean([x["load_transfer_s"] for x in loads])),
                     qps_incl_load=seeds_total / (batch_time + load_total),
                     val_papers=n_eval, val_acc=acc,
                     val_macro_f1=float(f1[present].mean()) if present.any() else float("nan"),
                     peak_gpu_gb=(torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0))
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(s))
                if not exists:
                    w.writeheader()
            w.writerow(s)
            print(f"[summary] {s['tag']}: wall {s['wall_ms']:.1f} ms, local {s['frac_local']*100:.1f}%, "
                  f"qps {s['qps']:.0f} (incl. load {s['qps_incl_load']:.0f}), acc {acc*100:.2f}% "
                  f"F1 {s['val_macro_f1']:.4f} on {n_eval:,}, drift {s['drift_pct']:.2f}%", flush=True)
    print(f"[done] {len(loads)} shard loads, total load {load_total:.1f} s -> {sp}", flush=True)


if __name__ == "__main__":
    main()
