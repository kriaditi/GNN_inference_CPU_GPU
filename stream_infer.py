"""Streaming (brute-force) GraphSAGE inference: the whole feature table is
streamed to the GPU in chunks and the GPU picks out the rows the queries need.

Baseline requested by Udit / Michael (28 Oct): instead of the CPU gathering
the sampled rows (node filtering, sage_mag240m.py), move every chunk of the
375 GB table to the GPU, select the needed rows there, then compute there.

One "pass" = the full table, chunk by chunk (contiguous slices, no CPU gather).
G batches share one pass:

  1. sample   all G batches on the CPU (pyg_lib, same sampler as everywhere)
  2. plan     CPU: union of the G batches' global row ids, sorted; chunk bounds
  3. index    copy the sorted ids + their slots to the GPU
  4. stream   for each chunk: copy chunk to GPU (stream_transfer), then pick
              the rows of that chunk the union needs (stream_select, on GPU)
  5. per batch: assemble its rows from the union buffer (GPU index) and run
              the model (compute, CUDA events)

  G = 1        one full pass per batch
  G = N        one pass serves N batches
  G = auto     the largest G whose union buffer fits next to one chunk

Every pass logs each phase; their sum is checked against the pass wall time.
The first --check-batches batches are verified bit-exact against the CPU
gather used by the node-filtering baseline (same rows, same order).

    python stream_infer.py --in-memory --weights best.pt --fanouts 40/20 \
        --batches-per-pass 1,8,64,auto --chunk-gb 8,47 --batches 1000 ...
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

import hetero_common as hc

SELECT_ROWS = 1_000_000   # rows per on-device select slice (~1.5 GB temporary)


def sync_now(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def run_pass(g, feat, model, dev, seed_batches, fanout, chunk_rows, chunk_buf, check, preds=None):
    """Serve len(seed_batches) batches with one full streaming pass.
    preds: a list; when given, each batch's argmax over its seeds is appended (accuracy pass)."""
    T = dict(sample_s=0.0, plan_s=0.0, index_s=0.0, stream_transfer_s=0.0,
             stream_select_s=0.0, assemble_s=0.0, compute_s=0.0)
    t_pass = time.perf_counter()

    # 1. sample (CPU)
    t0 = time.perf_counter()
    samples = [hc.sample_hetero(g, s, fanout) for s in seed_batches]
    T["sample_s"] = time.perf_counter() - t0

    # 2. plan (CPU): union of needed global ids, sorted; per-batch inverse
    t0 = time.perf_counter()
    per_batch_gid = []
    for node_id, _ in samples:
        per_batch_gid.append({nt: node_id[nt].numpy() + g.offset[nt] for nt in hc.NODE_TYPES})
    all_gid = np.concatenate([np.concatenate([b[nt] for nt in hc.NODE_TYPES]) for b in per_batch_gid])
    union, inverse = np.unique(all_gid, return_inverse=True)       # sorted ascending
    n_chunks = (feat.shape[0] + chunk_rows - 1) // chunk_rows
    bounds = np.searchsorted(union, np.arange(n_chunks + 1, dtype=np.int64) * chunk_rows)
    T["plan_s"] = time.perf_counter() - t0

    # 3. index (to device)
    t0 = time.perf_counter()
    union_d = torch.from_numpy(union).to(dev)
    inverse_d = torch.from_numpy(inverse).to(dev)
    out = torch.empty((union.size, feat.shape[1]), dtype=torch.float16, device=dev)
    T["index_s"] = sync_now(dev) - t0

    # 4. stream every chunk; select needed rows on the device
    for c in range(n_chunks):
        a = c * chunk_rows
        b = min(a + chunk_rows, feat.shape[0])
        t0 = time.perf_counter()
        buf = chunk_buf[: b - a]
        buf.copy_(torch.from_numpy(feat[a:b]))
        t1 = sync_now(dev)
        lo, hi = int(bounds[c]), int(bounds[c + 1])
        for s in range(lo, hi, SELECT_ROWS):           # slices keep the temporary small
            e = min(s + SELECT_ROWS, hi)
            out[s:e] = buf[union_d[s:e] - a]
        t2 = sync_now(dev)
        T["stream_transfer_s"] += t1 - t0
        T["stream_select_s"] += t2 - t1

    # 5. per batch: assemble from the union buffer, compute
    pos = 0
    n_nodes = []
    for bi, (node_id, edge_index) in enumerate(samples):
        t0 = time.perf_counter()
        x = {}
        for nt in hc.NODE_TYPES:
            n = per_batch_gid[bi][nt].size
            x[nt] = out[inverse_d[pos:pos + n]]
            pos += n
        ei = {k: v.to(dev) for k, v in edge_index.items()}
        t1 = sync_now(dev)
        T["assemble_s"] += t1 - t0
        if bi < check:
            ref = hc.gather_features(feat, node_id, g)
            for nt in hc.NODE_TYPES:
                assert torch.equal(x[nt].cpu(), ref[nt]), f"streamed rows differ from CPU gather ({nt})"
        if dev.type == "cuda":
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            out_logits = model({k: v.float() for k, v in x.items()}, ei)
            e1.record()
            torch.cuda.synchronize()
            T["compute_s"] += e0.elapsed_time(e1) / 1000.0
        else:
            t2 = time.perf_counter()
            out_logits = model({k: v.float() for k, v in x.items()}, ei)
            T["compute_s"] += time.perf_counter() - t2
        if preds is not None:
            preds.append(out_logits[:seed_batches[bi].numel()].argmax(1).cpu())
        n_nodes.append(sum(v.numel() for v in node_id.values()))

    wall = time.perf_counter() - t_pass
    attributed = sum(T.values())
    row = dict(T)
    row.update(batches=len(seed_batches), seeds=sum(s.numel() for s in seed_batches),
               nodes_per_batch_mean=float(np.mean(n_nodes)), union_rows=int(union.size),
               union_gb=union.size * feat.shape[1] * 2 / 1e9, chunks=n_chunks,
               streamed_gb=feat.shape[0] * feat.shape[1] * 2 / 1e9,
               pass_wall_s=wall, drift_pct=abs(wall - attributed) / wall * 100)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csr", required=True)
    ap.add_argument("--hetero-dir", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--fanouts", default="40/20")
    ap.add_argument("--batches-per-pass", default="1,8,64,auto")
    ap.add_argument("--chunk-gb", default="8,47")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=1000, help="timed batches per config")
    ap.add_argument("--warmup-passes", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=20_000_000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--check-batches", type=int, default=3, help="bit-exact check vs CPU gather")
    ap.add_argument("--margin-gb", type=float, default=8.0, help="GPU memory kept free for auto G")
    ap.add_argument("--val", action="store_true", help="accuracy pass on the validation papers through the streaming path, once per chunk size and fanout")
    ap.add_argument("--val-G", type=int, default=8, help="batches per pass in the accuracy pass (accuracy does not depend on G)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results_stream")
    ap.add_argument("--summary", default="")
    ap.add_argument("--tag", default="stream")
    a = ap.parse_args()

    dev = torch.device(a.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no GPU visible")
    os.makedirs(a.out, exist_ok=True)
    sp = a.summary or os.path.join(a.out, "summary_stream.csv")

    g = hc.HeteroGraph(a.csr, a.hetero_dir)
    feat = hc.load_features(a.feat, a.in_memory)
    from ogb.lsc import MAG240MDataset
    ds = MAG240MDataset(root=a.root)
    n_classes = ds.num_classes
    labels = np.nan_to_num(ds.paper_label, nan=-1).astype(np.int64)
    val_idx = np.asarray(ds.get_idx_split("valid"))
    model = hc.RSAGE(feat.shape[1], a.hidden, n_classes, 2).to(dev).eval()
    model.load_state_dict(torch.load(a.weights, map_location=dev)["model"])

    rng = np.random.default_rng(0)                        # same pool as the baselines
    P = g.num["paper"]
    pool = torch.from_numpy(np.sort(rng.choice(P, size=min(a.seeds, P), replace=False))).long()
    row_bytes = feat.shape[1] * 2
    fanouts = [[int(v) for v in f.split("/")] for f in a.fanouts.split(",")]
    chunks_gb = [float(v) for v in a.chunk_gb.split(",")]
    gs = a.batches_per_pass.split(",")

    with torch.no_grad():
        for chunk_gb in chunks_gb:
            chunk_rows = int(chunk_gb * 1e9 // row_bytes)
            try:
                chunk_buf = torch.empty((chunk_rows, feat.shape[1]), dtype=torch.float16, device=dev)
            except torch.cuda.OutOfMemoryError as e:
                print(f"[{a.tag}] chunk {chunk_gb:g} GB does not fit on the GPU: {str(e)[:120]}", flush=True)
                torch.cuda.empty_cache()
                continue
            for f in fanouts:
                val_acc, val_f1 = float("nan"), float("nan")
                if a.val:
                    tv = time.perf_counter()
                    vb = [torch.from_numpy(val_idx[i:i + a.batch_size]).long() for i in range(0, val_idx.size, a.batch_size)]
                    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
                    for i in range(0, len(vb), a.val_G):
                        group = vb[i:i + a.val_G]
                        preds = []
                        run_pass(g, feat, model, dev, group, f, chunk_rows, chunk_buf, 0, preds=preds)
                        for b, p in zip(group, preds):
                            np.add.at(conf, (labels[b.numpy()], p.numpy()), 1)
                    tp = np.diag(conf).astype(float)
                    val_acc = tp.sum() / conf.sum()
                    prec = tp / np.maximum(conf.sum(0), 1); rec = tp / np.maximum(conf.sum(1), 1)
                    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
                    present = conf.sum(1) > 0
                    val_f1 = float(f1[present].mean())
                    print(f"[val] chunk {chunk_gb:g} GB fanout {f}: accuracy {val_acc:.4f}  macro-F1 {val_f1:.4f} "
                          f"on {int(conf.sum()):,} papers ({time.perf_counter() - tv:.0f}s, G={a.val_G})", flush=True)
                for gspec in gs:
                    if gspec == "auto":
                        probe = hc.sample_hetero(g, pool[:a.batch_size], f)[0]
                        per_batch = sum(v.numel() for v in probe.values()) * row_bytes
                        free = (torch.cuda.mem_get_info()[0] if dev.type == "cuda" else 64e9) - a.margin_gb * 1e9
                        G = max(1, int(free // per_batch))
                        print(f"[{a.tag}] auto: free after margin {free/1e9:.1f} GB, per-batch "
                              f"{per_batch/1e6:.1f} MB ({sum(v.numel() for v in probe.values()):,} rows) -> G={G}",
                              flush=True)
                    else:
                        G = int(gspec)
                    tag = f"{a.tag}_c{chunk_gb:g}gb_G{G}{'auto' if gspec == 'auto' else ''}_f{'x'.join(map(str, f))}"
                    n_pass = a.warmup_passes + max(1, -(-a.batches // G))
                    need = n_pass * G * a.batch_size
                    if need > pool.numel():
                        n_pass = pool.numel() // (G * a.batch_size)
                    print(f"\n[{tag}] chunk {chunk_gb:g} GB ({chunk_rows:,} rows), G={G}, "
                          f"{n_pass} passes ({a.warmup_passes} warmup)", flush=True)
                    rows, i = [], 0
                    for p in range(n_pass):
                        seed_batches = [pool[(i + j) * a.batch_size:(i + j + 1) * a.batch_size] for j in range(G)]
                        i += G
                        try:
                            r = run_pass(g, feat, model, dev, seed_batches, f, chunk_rows, chunk_buf,
                                         a.check_batches if p == 0 else 0)
                        except torch.cuda.OutOfMemoryError as e:
                            print(f"[{tag}] OOM at G={G}: {str(e)[:120]}", flush=True)
                            torch.cuda.empty_cache()
                            break
                        if p < a.warmup_passes:
                            if p == 0:
                                print(f"[{tag}] check: first {a.check_batches} batches bit-exact vs CPU gather", flush=True)
                            continue
                        r["pass"] = p - a.warmup_passes
                        rows.append(r)
                        if r["pass"] % max(1, len(range(n_pass)) // 10) == 0:
                            print(f"[{tag}] pass {r['pass']}: wall {r['pass_wall_s']:.2f} s, stream transfer "
                                  f"{r['stream_transfer_s']:.2f} s, union {r['union_gb']:.1f} GB, "
                                  f"drift {r['drift_pct']:.2f}%", flush=True)
                    if not rows:
                        continue
                    with open(os.path.join(a.out, f"{tag}_perpass.csv"), "w", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                    med = lambda k: float(np.median([x[k] for x in rows]))  # noqa: E731
                    total = sum(x["pass_wall_s"] for x in rows)
                    seeds = sum(x["seeds"] for x in rows)
                    s = dict(tag=tag, chunk_gb=chunk_gb, G=G, G_mode=gspec, fanout=",".join(map(str, f)),
                             passes=len(rows), batches=sum(x["batches"] for x in rows), seeds_total=seeds,
                             pass_wall_ms=med("pass_wall_s") * 1e3,
                             wall_ms_per_batch=med("pass_wall_s") / G * 1e3,
                             sample_ms_per_pass=med("sample_s") * 1e3, plan_ms_per_pass=med("plan_s") * 1e3,
                             index_ms_per_pass=med("index_s") * 1e3,
                             stream_transfer_ms_per_pass=med("stream_transfer_s") * 1e3,
                             stream_select_ms_per_pass=med("stream_select_s") * 1e3,
                             assemble_ms_per_pass=med("assemble_s") * 1e3, compute_ms_per_pass=med("compute_s") * 1e3,
                             streamed_gb_per_pass=rows[0]["streamed_gb"],
                             stream_gbps=rows[0]["streamed_gb"] / med("stream_transfer_s"),
                             union_gb=med("union_gb"), nodes_per_batch=med("nodes_per_batch_mean"),
                             drift_pct=med("drift_pct"), qps=seeds / total,
                             peak_gpu_gb=(torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0),
                             val_papers=int(val_idx.size) if a.val else 0, val_acc=val_acc, val_macro_f1=val_f1)
                    exists = os.path.exists(sp)
                    with open(sp, "a", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(s))
                        if not exists:
                            w.writeheader()
                        w.writerow(s)
                    print(f"[summary] {tag}: {s['wall_ms_per_batch']:.1f} ms/batch, pass {s['pass_wall_ms']:.0f} ms, "
                          f"stream {s['stream_gbps']:.1f} GB/s, qps {s['qps']:.0f}, drift {s['drift_pct']:.2f}%", flush=True)
            del chunk_buf
            if dev.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
