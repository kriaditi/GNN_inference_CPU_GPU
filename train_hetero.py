"""Train R-GraphSAGE on MAG240M (heterogeneous), so inference can report
real accuracy / macro-F1 instead of random-weight predictions.

Data: OGB split. 1,112,392 labelled train papers, 138,949 validation
papers (test labels are hidden by OGB, so validation is our ground truth).
Features for papers, authors and institutions from full_feat.npy (375 GB,
in RAM). Sampling, gather and model are the ones the inference harness
uses, imported from hetero_common.py.

    python train_hetero.py --fanout 40,20 --epochs 40 --in-memory
    python train_hetero.py --smoke          # 20 steps on memmapped features

Outputs (in --out):
    hetero_train_log.csv   epoch, loss, val_acc, val_macro_f1, seconds
    best.pt                weights at best validation macro-F1
    last.pt                weights + optimiser + epoch, for --resume
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from hetero_common import (HeteroGraph, RSAGE, check_directions,
                           gather_features, load_features, sample_hetero)


def batch_forward(model, g, feat, seeds, fanout, dev):
    node_id, edge_index = sample_hetero(g, seeds, fanout)
    x = gather_features(feat, node_id, g)
    x = {k: v.to(dev, non_blocking=True).float() for k, v in x.items()}
    ei = {k: v.to(dev, non_blocking=True) for k, v in edge_index.items()}
    out = model(x, ei)[:seeds.numel()]
    n_nodes = sum(v.numel() for v in node_id.values())
    return out, node_id, edge_index, n_nodes


@torch.no_grad()
def evaluate(model, g, feat, idx, labels, fanout, dev, batch_size, n_classes, max_batches=None):
    model.eval()
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long, device=dev)
    nb = 0
    for i in range(0, idx.numel(), batch_size):
        seeds = idx[i:i + batch_size]
        out, *_ = batch_forward(model, g, feat, seeds, fanout, dev)
        pred = out.argmax(1)
        y = labels[seeds].to(dev)
        conf += torch.bincount(y * n_classes + pred, minlength=n_classes ** 2).view(n_classes, n_classes)
        nb += 1
        if max_batches and nb >= max_batches:
            break
    tp = conf.diag().float()
    acc = (tp.sum() / conf.sum()).item()
    prec = tp / conf.sum(0).clamp(min=1).float()
    rec = tp / conf.sum(1).clamp(min=1).float()
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    present = conf.sum(1) > 0                      # classes that occur in this split
    macro_f1 = f1[present].mean().item()
    model.train()
    return acc, macro_f1, int(conf.sum().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/share/gupta/ak2877/gnn/data")
    p.add_argument("--csr", default="/share/gupta/ak2877/gnn/csr")
    p.add_argument("--hetero", default="/share/gupta/ak2877/gnn/csr_hetero")
    p.add_argument("--feat", default="/share/gupta/ak2877/gnn/data/full_feat.npy")
    p.add_argument("--in-memory", action="store_true")
    p.add_argument("--fanout", default="40,20")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="results/hetero")
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    p.add_argument("--smoke", action="store_true",
                   help="20 train steps + 20 validation batches, memmapped features")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible; training needs one")
    dev = torch.device("cuda")
    n_thr = int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads()))
    torch.set_num_threads(n_thr)
    fanout = [int(v) for v in args.fanout.split(",")]
    os.makedirs(args.out, exist_ok=True)
    print(f"[train] fanout={fanout} batch={args.batch_size} epochs={args.epochs} "
          f"threads={n_thr} gpu={torch.cuda.get_device_name(0)}", flush=True)

    g = HeteroGraph(args.csr, args.hetero)
    feat = load_features(args.feat, args.in_memory and not args.smoke)

    from ogb.lsc import MAG240MDataset
    ds = MAG240MDataset(root=args.root)
    n_classes = ds.num_classes
    labels = torch.from_numpy(np.nan_to_num(ds.paper_label, nan=-1).astype(np.int64))
    train_idx = torch.from_numpy(ds.get_idx_split("train")).long()
    valid_idx = torch.from_numpy(ds.get_idx_split("valid")).long()
    assert (labels[train_idx] >= 0).all() and (labels[valid_idx] >= 0).all()
    print(f"[data] train {train_idx.numel():,}  valid {valid_idx.numel():,}  "
          f"classes {n_classes}", flush=True)

    model = RSAGE(feat.shape[1], args.hidden, n_classes, len(fanout)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)

    start_epoch, best_f1 = 0, -1.0
    last_path, best_path = os.path.join(args.out, "last.pt"), os.path.join(args.out, "best.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch, best_f1 = ck["epoch"] + 1, ck["best_f1"]
        print(f"[resume] from epoch {ck['epoch']} (best macro-F1 so far {best_f1:.4f})", flush=True)

    log_path = os.path.join(args.out, "hetero_train_log.csv")
    if not os.path.exists(log_path) or not args.resume:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "val_acc", "val_macro_f1",
                                    "nodes_per_batch", "seconds"])

    rng = np.random.default_rng(0)
    model.train()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        perm = torch.from_numpy(rng.permutation(train_idx.numel()))
        order = train_idx[perm]
        losses, node_counts = [], []
        n_steps = (order.numel() + args.batch_size - 1) // args.batch_size
        if args.smoke:
            n_steps = 20
        for step in range(n_steps):
            seeds = order[step * args.batch_size:(step + 1) * args.batch_size]
            out, node_id, edge_index, n_nodes = batch_forward(model, g, feat, seeds, fanout, dev)
            if epoch == start_epoch and step == 0:
                check_directions(g, seeds, node_id, edge_index)
                print("[check] nodes per type in first batch: " +
                      ", ".join(f"{k} {v.numel():,}" for k, v in node_id.items()) +
                      f"  (x{n_nodes/seeds.numel():.0f} per seed)", flush=True)
                print(f"[check] peak GPU after first batch: "
                      f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
            loss = F.cross_entropy(out, labels[seeds].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item()); node_counts.append(n_nodes)
            if step % 100 == 0 or args.smoke:
                el = time.time() - t0
                print(f"[epoch {epoch}] step {step}/{n_steps}  loss {loss.item():.3f}  "
                      f"nodes {n_nodes:,}  {el/(step+1):.2f} s/step", flush=True)

        acc, f1, n_eval = evaluate(model, g, feat, valid_idx, labels, fanout, dev,
                                   args.batch_size, n_classes,
                                   max_batches=20 if args.smoke else None)
        secs = time.time() - t0
        print(f"[epoch {epoch}] loss {np.mean(losses):.4f}  val acc {acc:.4f}  "
              f"val macro-F1 {f1:.4f} (on {n_eval:,})  {secs/60:.1f} min", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{np.mean(losses):.5f}", f"{acc:.5f}",
                                    f"{f1:.5f}", int(np.median(node_counts)), int(secs)])
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": acc,
                        "val_macro_f1": f1, "fanout": fanout, "hidden": args.hidden,
                        "n_classes": n_classes}, best_path)
            print(f"[ckpt] new best macro-F1 {f1:.4f} -> {best_path}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best_f1": best_f1}, last_path)
        if args.smoke:
            print("[smoke] done", flush=True)
            break


if __name__ == "__main__":
    main()
