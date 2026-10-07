"""Convert MAG240M's citation edge list into CSR, once."""
import argparse, os, time
import numpy as np
from ogb.lsc import MAG240MDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rowptr_p = os.path.join(a.out, "rowptr.npy")
    col_p = os.path.join(a.out, "col.npy")

    if os.path.exists(rowptr_p) and os.path.exists(col_p):
        print("CSR already exists, nothing to do")
        return

    ds = MAG240MDataset(root=a.root)
    n = ds.num_papers
    print(f"papers: {n:,}", flush=True)

    print("[1/4] loading edges ...", flush=True)
    t0 = time.time()
    ei = ds.edge_index("paper", "paper")
    print(f"      {ei.shape[1]:,} edges  {ei.nbytes/1e9:.1f} GB  "
          f"({time.time()-t0:.0f}s)", flush=True)

    src, dst = ei[0], ei[1]
    del ei

    print("[2/4] symmetrising ...", flush=True)
    src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
    print(f"      {src.shape[0]:,} directed edges", flush=True)

    print("[3/4] sorting into CSR (slow, memory heavy) ...", flush=True)
    t0 = time.time()
    order = np.argsort(src, kind="stable")
    col = dst[order].astype(np.int32)
    src_sorted = src[order]
    del order, dst, src

    counts = np.bincount(src_sorted, minlength=n)
    del src_sorted
    rowptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=rowptr[1:])
    del counts
    print(f"      done in {(time.time()-t0)/60:.1f} min", flush=True)

    print("[4/4] writing ...", flush=True)
    np.save(rowptr_p, rowptr)
    np.save(col_p, col)
    deg = np.diff(rowptr)
    print(f"      rowptr {rowptr.nbytes/1e9:.2f} GB   col {col.nbytes/1e9:.1f} GB")
    print(f"      degree: mean {deg.mean():.1f}  max {deg.max():,}  "
          f"isolated {(deg==0).sum():,}")
    print("done", flush=True)


if __name__ == "__main__":
    main()