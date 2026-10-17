"""Build the heterogeneous CSRs for MAG240M (step 1 of the hetero pipeline).

MAG240M has three node types and three edge types. The paper-cites-paper
CSR already exists (csr/rowptr.npy, csr/col.npy, symmetrised). This script
adds the other relations, in BOTH directions so the sampler can walk from a
paper to its authors and on to their institutions:

    a2p  author  -> paper        (author writes paper)        386M edges
    p2a  paper   -> author       (reverse of a2p)             386M edges
    a2i  author  -> institution  (author affiliated with)      44M edges
    i2a  institution -> author   (reverse of a2i)              44M edges

Each relation is stored as <name>_rowptr.npy / <name>_col.npy, int64, with
LOCAL ids per node type (paper 0..P-1, author 0..A-1, institution 0..I-1),
which is what pyg_lib's hetero sampler expects. meta.json records the node
counts and the offsets into full_feat.npy (papers 0, authors P, institutions
P+A) so the gather can turn (type, local id) into a row of the 375 GB table.

    python build_hetero_csr.py --root /share/gupta/ak2877/gnn/data \
        --out /share/gupta/ak2877/gnn/csr_hetero

Memory: one relation at a time; 386M edges x (2 cols + sort index + output)
int64 is about 15 GB peak. CPU only.
"""
import argparse
import json
import os
import time

import numpy as np


def to_csr(src, dst, n_src, name, out_dir):
    """Sort edges by src, write rowptr (n_src+1) and col (E) as int64."""
    t0 = time.time()
    order = np.argsort(src, kind="stable")
    col = np.ascontiguousarray(dst[order]).astype(np.int64)
    del order
    deg = np.bincount(src, minlength=n_src).astype(np.int64)
    rowptr = np.zeros(n_src + 1, dtype=np.int64)
    np.cumsum(deg, out=rowptr[1:])
    assert rowptr[-1] == col.size, f"{name}: rowptr end {rowptr[-1]} != edges {col.size}"
    np.save(os.path.join(out_dir, f"{name}_rowptr.npy"), rowptr)
    np.save(os.path.join(out_dir, f"{name}_col.npy"), col)
    nz = deg[deg > 0]
    print(f"[{name}] {col.size:,} edges, {n_src:,} rows, "
          f"{(deg == 0).sum():,} rows empty, degree mean {nz.mean():.1f} "
          f"max {deg.max():,}  ({time.time()-t0:.0f}s)", flush=True)
    return rowptr, col


def check_reverse(r_fwd, c_fwd, r_rev, c_rev, name, n=5, rng=None):
    """For a few source nodes, every target must list the source back."""
    rng = rng or np.random.default_rng(0)
    nz = np.flatnonzero(r_fwd[1:] > r_fwd[:-1])
    for s in rng.choice(nz, size=n, replace=False):
        for t in c_fwd[r_fwd[s]:r_fwd[s + 1]][:3]:
            back = c_rev[r_rev[t]:r_rev[t + 1]]
            assert s in back, f"{name}: {s} -> {t} but reverse missing"
    print(f"[check] {name}: forward/reverse consistent on {n} samples", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/share/gupta/ak2877/gnn/data")
    p.add_argument("--out", default="/share/gupta/ak2877/gnn/csr_hetero")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from ogb.lsc import MAG240MDataset
    ds = MAG240MDataset(root=args.root)
    P, A, I = ds.num_papers, ds.num_authors, ds.num_institutions
    print(f"[meta] papers {P:,} authors {A:,} institutions {I:,}", flush=True)

    # ---- author writes paper, both directions -------------------------
    t0 = time.time()
    ei = ds.edge_index("author", "writes", "paper")      # (2, E): author, paper
    print(f"[load] author-writes-paper {ei.shape[1]:,} edges ({time.time()-t0:.0f}s)", flush=True)
    assert ei[0].max() < A and ei[1].max() < P
    ra, ca = to_csr(ei[0], ei[1], A, "a2p", args.out)
    rp, cp = to_csr(ei[1], ei[0], P, "p2a", args.out)
    check_reverse(ra, ca, rp, cp, "a2p/p2a")
    papers_with_author = int((rp[1:] > rp[:-1]).sum())
    del ei, ra, ca, rp, cp

    # ---- author affiliated with institution, both directions ----------
    t0 = time.time()
    ei = ds.edge_index("author", "institution")          # (2, E): author, institution
    print(f"[load] author-institution {ei.shape[1]:,} edges ({time.time()-t0:.0f}s)", flush=True)
    assert ei[0].max() < A and ei[1].max() < I
    ra, ca = to_csr(ei[0], ei[1], A, "a2i", args.out)
    ri, ci = to_csr(ei[1], ei[0], I, "i2a", args.out)
    check_reverse(ra, ca, ri, ci, "a2i/i2a")
    authors_with_inst = int((ra[1:] > ra[:-1]).sum())
    del ei, ra, ca, ri, ci

    meta = dict(
        num_papers=P, num_authors=A, num_institutions=I, num_nodes=P + A + I,
        offset=dict(paper=0, author=P, institution=P + A),
        relations=dict(
            p2p="paper cites paper (symmetrised), in ../csr/{rowptr,col}.npy",
            a2p="author writes paper", p2a="paper written by author",
            a2i="author affiliated with institution", i2a="institution has author"),
        papers_with_author=papers_with_author,
        authors_with_institution=authors_with_inst,
        dtype="int64", ids="local per node type",
        feat="/share/gupta/ak2877/gnn/data/full_feat.npy",
    )
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] papers with >=1 author: {papers_with_author:,}/{P:,}  "
          f"authors with >=1 institution: {authors_with_inst:,}/{A:,}", flush=True)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
