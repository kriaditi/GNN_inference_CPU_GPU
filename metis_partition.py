"""METIS partitioning of the full MAG240M heterogeneous graph.

Two stages, so the graph is built once and partitioned several times:

  --build          build the global undirected graph (244,160,499 vertices:
                   papers, then authors, then institutions, same order as
                   full_feat.npy) from the paper CSR and the four hetero CSRs,
                   drop self-loops and duplicate edges, spot-check symmetry,
                   save xadj / adjncy to --out (about 30 GB).
  --k K            load that graph and call METIS_PartGraphKway for K parts,
                   unit vertex weights (every row is 1.5 KB, so balanced node
                   counts = balanced GB), write part_k{K}.npy (uint8 if K<256)
                   and part_k{K}.json with per-part sizes and cut statistics.
  --smoke          both stages on a slice (1M papers, 200k authors, all
                   institutions), k=4, in a few minutes. Exercises the library
                   load, the 64-bit index path and every output file.

METIS is called through ctypes from libmetis.so built with IDXTYPEWIDTH=64
(the adjacency has 3.46B entries, past 32-bit). The index and real widths and
the option indices are read from metis.h at run time so a 5.1 or 5.2 build
both work.

    export METIS_HOME=$HOME/local
    python metis_partition.py --build
    python metis_partition.py --k 8
"""
import argparse
import ctypes
import json
import os
import re
import time

import numpy as np

ROOT = "/share/gupta/ak2877/gnn"


# --------------------------------------------------------------------------
# METIS via ctypes
# --------------------------------------------------------------------------
class Metis:
    def __init__(self, home):
        inc = os.path.join(home, "include", "metis.h")
        src = open(inc).read()
        self.idx_w = int(re.search(r"^#define IDXTYPEWIDTH\s+(\d+)", src, re.M).group(1))
        self.real_w = int(re.search(r"^#define REALTYPEWIDTH\s+(\d+)", src, re.M).group(1))
        if self.idx_w != 64:
            raise SystemExit(f"libmetis has IDXTYPEWIDTH={self.idx_w}; this graph needs 64. "
                             "Rebuild METIS with i64=1.")
        enum = re.search(r"typedef enum \{(.*?)\} moptions_et;", src, re.S).group(1)
        names = [t.strip() for t in re.sub(r"/\*.*?\*/", "", enum, flags=re.S).split(",") if t.strip()]
        self.opt = {n: i for i, n in enumerate(names)}
        self.nopt = int(re.search(r"#define METIS_NOPTIONS\s+(\d+)", src).group(1))
        self.idx_t = ctypes.c_int64
        self.real_t = ctypes.c_double if self.real_w == 64 else ctypes.c_float
        self.np_idx = np.int64
        libdir = os.path.join(home, "lib")
        for dep in ("libGKlib.so",):
            p = os.path.join(libdir, dep)
            if os.path.exists(p):
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        self.lib = ctypes.CDLL(os.path.join(libdir, "libmetis.so"), mode=ctypes.RTLD_GLOBAL)
        print(f"[metis] {libdir}/libmetis.so  idx {self.idx_w}-bit  real {self.real_w}-bit  "
              f"{len(self.opt)} options", flush=True)

    def part_kway(self, xadj, adjncy, nparts, seed=0, ufactor=30, dbg=True):
        assert xadj.dtype == self.np_idx and adjncy.dtype == self.np_idx
        assert xadj.flags.c_contiguous and adjncy.flags.c_contiguous
        n = xadj.size - 1
        options = np.empty(self.nopt, dtype=self.np_idx)
        self.lib.METIS_SetDefaultOptions(options.ctypes.data_as(ctypes.POINTER(self.idx_t)))
        options[self.opt["METIS_OPTION_NUMBERING"]] = 0
        options[self.opt["METIS_OPTION_SEED"]] = seed
        options[self.opt["METIS_OPTION_UFACTOR"]] = ufactor      # 30 = 3% imbalance
        if dbg:
            options[self.opt["METIS_OPTION_DBGLVL"]] = 1          # METIS_DBG_INFO: progress to stdout
        part = np.empty(n, dtype=self.np_idx)
        nv, nc, np_, objval = (self.idx_t(n), self.idx_t(1), self.idx_t(nparts), self.idx_t(0))
        P = ctypes.POINTER(self.idx_t)
        rc = self.lib.METIS_PartGraphKway(
            ctypes.byref(nv), ctypes.byref(nc),
            xadj.ctypes.data_as(P), adjncy.ctypes.data_as(P),
            None, None, None,                       # vwgt, vsize, adjwgt: unit
            ctypes.byref(np_), None, None,          # tpwgts, ubvec: uniform
            options.ctypes.data_as(P), ctypes.byref(objval), part.ctypes.data_as(P))
        if rc != 1:                                 # METIS_OK == 1
            raise RuntimeError(f"METIS_PartGraphKway returned {rc} (1=OK, -2 input, -3 memory, -4 other)")
        return part, int(objval.value)


# --------------------------------------------------------------------------
# graph build
# --------------------------------------------------------------------------
def load_csr(d, name):
    return (np.load(os.path.join(d, f"{name}_rowptr.npy")),
            np.load(os.path.join(d, f"{name}_col.npy")))


def build_graph(csr_dir, het_dir, out_dir, smoke=False):
    t0 = time.time()
    with open(os.path.join(het_dir, "meta.json")) as f:
        meta = json.load(f)
    P, A, I = meta["num_papers"], meta["num_authors"], meta["num_institutions"]
    pp_r, pp_c = np.load(os.path.join(csr_dir, "rowptr.npy")), np.load(os.path.join(csr_dir, "col.npy"))
    p2a_r, p2a_c = load_csr(het_dir, "p2a")     # paper -> author (local author ids)
    a2p_r, a2p_c = load_csr(het_dir, "a2p")     # author -> paper
    a2i_r, a2i_c = load_csr(het_dir, "a2i")     # author -> institution
    i2a_r, i2a_c = load_csr(het_dir, "i2a")     # institution -> author
    print(f"[build] inputs loaded ({time.time()-t0:.0f}s)", flush=True)

    if smoke:
        # slice: first P papers, first A authors, all institutions; keep only
        # edges with both endpoints inside the slice
        P, A = min(P, 1_000_000), min(A, 200_000)
        def slice_csr(r, c, nrows, keep):
            r = r[:nrows + 1]; c = c[:r[-1]]
            m = keep(c)
            rows = np.repeat(np.arange(nrows, dtype=np.int64), np.diff(r))
            deg = np.bincount(rows[m], minlength=nrows).astype(np.int64)
            nr = np.zeros(nrows + 1, np.int64); np.cumsum(deg, out=nr[1:])
            return nr, c[m]
        pp_r, pp_c = slice_csr(pp_r, pp_c, P, lambda c: c < P)
        p2a_r, p2a_c = slice_csr(p2a_r, p2a_c, P, lambda c: c < A)
        a2p_r, a2p_c = slice_csr(a2p_r, a2p_c, A, lambda c: c < P)
        a2i_r, a2i_c = slice_csr(a2i_r, a2i_c, A, lambda c: c >= 0)
        i2a_r, i2a_c = slice_csr(i2a_r, i2a_c, I, lambda c: c < A)
        print(f"[smoke] slice: {P:,} papers, {A:,} authors, {I:,} institutions", flush=True)

    N = P + A + I
    OA, OI = P, P + A                              # global offsets
    deg = np.zeros(N, dtype=np.int64)
    deg[:P] = np.diff(pp_r) + np.diff(p2a_r)
    deg[OA:OI] = np.diff(a2p_r) + np.diff(a2i_r)
    deg[OI:] = np.diff(i2a_r)
    xadj = np.zeros(N + 1, dtype=np.int64); np.cumsum(deg, out=xadj[1:])
    E = int(xadj[-1])
    print(f"[build] {N:,} vertices, {E:,} adjacency entries before cleanup "
          f"({E*8/1e9:.1f} GB)", flush=True)

    adjncy = np.empty(E, dtype=np.int64)

    def place(row_lo, src_r, src_c, offset_col, slot_offset_by_row):
        """Copy one relation's lists into adjncy at per-row positions."""
        d = np.diff(src_r)
        if d.sum() == 0:
            return
        start = xadj[row_lo:row_lo + d.size] + slot_offset_by_row
        pos = np.repeat(start, d) + (np.arange(d.sum(), dtype=np.int64) - np.repeat(src_r[:-1], d))
        adjncy[pos] = src_c + offset_col
        del pos

    # papers: citations first, then authors
    place(0, pp_r, pp_c, 0, 0)
    place(0, p2a_r, p2a_c, OA, np.diff(pp_r))
    # authors: papers first, then institutions
    place(OA, a2p_r, a2p_c, 0, 0)
    place(OA, a2i_r, a2i_c, OI, np.diff(a2p_r))
    # institutions: authors
    place(OI, i2a_r, i2a_c, OA, 0)
    del pp_c, p2a_c, a2p_c, a2i_c, i2a_c
    print(f"[build] adjacency filled ({time.time()-t0:.0f}s)", flush=True)

    # cleanup: sort within rows, drop self-loops and duplicates.
    # key = row * N + col sorts by row then col; N < 2^28 so key < 2^56.
    row = np.repeat(np.arange(N, dtype=np.int64), deg)
    key = row * N + adjncy
    del adjncy
    key.sort()
    row = key // N
    col = key - row * N
    del key
    keep = col != row
    keep[1:] &= (row[1:] != row[:-1]) | (col[1:] != col[:-1])
    n_self = int((col == row).sum())
    row, col = row[keep], col[keep]
    n_dup = E - n_self - row.size
    adjncy = np.ascontiguousarray(col); del col
    deg = np.bincount(row, minlength=N).astype(np.int64); del row
    xadj = np.zeros(N + 1, dtype=np.int64); np.cumsum(deg, out=xadj[1:])
    print(f"[build] removed {n_self:,} self-loops and {n_dup:,} duplicates; "
          f"{adjncy.size:,} entries = {adjncy.size//2:,} undirected edges "
          f"({time.time()-t0:.0f}s)", flush=True)
    assert adjncy.size % 2 == 0, "odd adjacency count: graph is not symmetric"

    # symmetry spot check: for sampled vertices, every neighbour lists them back
    rng = np.random.default_rng(0)
    nz = np.flatnonzero(deg > 0)
    for v in rng.choice(nz, size=min(200, nz.size), replace=False):
        for u in adjncy[xadj[v]:xadj[v] + min(5, deg[v])]:
            back = adjncy[xadj[u]:xadj[u + 1]]
            j = np.searchsorted(back, v)
            assert j < back.size and back[j] == v, f"asymmetric: {v} -> {u} but not back"
    print("[build] symmetry spot check OK (200 vertices)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "graph_xadj.npy"), xadj)
    np.save(os.path.join(out_dir, "graph_adjncy.npy"), adjncy)
    ginfo = dict(num_vertices=N, num_papers=P, num_authors=A, num_institutions=I,
                 offset=dict(paper=0, author=OA, institution=OI),
                 undirected_edges=int(adjncy.size // 2), self_loops_removed=n_self,
                 duplicates_removed=int(n_dup), smoke=smoke,
                 isolated_vertices=int((deg == 0).sum()))
    with open(os.path.join(out_dir, "graph_meta.json"), "w") as f:
        json.dump(ginfo, f, indent=2)
    print(f"[build] saved to {out_dir} ({(time.time()-t0)/60:.1f} min)\n{json.dumps(ginfo, indent=2)}", flush=True)
    return xadj, adjncy, ginfo


# --------------------------------------------------------------------------
# partition + statistics
# --------------------------------------------------------------------------
def partition(metis, out_dir, k, seed=0):
    t0 = time.time()
    with open(os.path.join(out_dir, "graph_meta.json")) as f:
        g = json.load(f)
    xadj = np.load(os.path.join(out_dir, "graph_xadj.npy"))
    adjncy = np.load(os.path.join(out_dir, "graph_adjncy.npy"))
    N, P, OA, OI = g["num_vertices"], g["num_papers"], g["offset"]["author"], g["offset"]["institution"]
    print(f"[part] graph loaded: {N:,} vertices, {adjncy.size//2:,} edges "
          f"({time.time()-t0:.0f}s); METIS k={k} ...", flush=True)

    t1 = time.time()
    part, objval = metis.part_kway(xadj, adjncy, k, seed=seed)
    print(f"[part] METIS done in {(time.time()-t1)/60:.1f} min, edge cut {objval:,}", flush=True)

    dtype = np.uint8 if k < 256 else np.int32
    np.save(os.path.join(out_dir, f"part_k{k}.npy"), part.astype(dtype))

    # per-part composition
    bytes_per_row = 768 * 2
    parts = []
    for p in range(k):
        m = part == p
        n_p, n_a, n_i = int(m[:P].sum()), int(m[OA:OI].sum()), int(m[OI:].sum())
        parts.append(dict(part=p, nodes=int(m.sum()), papers=n_p, authors=n_a, institutions=n_i,
                          feature_gb=round(int(m.sum()) * bytes_per_row / 1e9, 1)))
    sizes = np.array([q["nodes"] for q in parts])
    imbalance = float(sizes.max() / sizes.mean())

    # locality by relation: fraction of edges whose two ends share a part
    row = np.repeat(np.arange(N, dtype=np.int64), np.diff(xadj))
    same = part[row] == part[adjncy]
    src_type = np.where(row < OA, 0, np.where(row < OI, 1, 2))
    dst_type = np.where(adjncy < OA, 0, np.where(adjncy < OI, 1, 2))
    def frac(a, b, st=src_type, dt=dst_type, sm=same):
        m = (st == a) & (dt == b)
        return float(sm[m].mean()) if m.any() else float("nan")
    locality = dict(paper_paper=frac(0, 0), paper_author=frac(0, 1), author_institution=frac(1, 2),
                    all=float(same.mean()))
    del row, same, src_type, dst_type

    # authors whose papers are all in the author's own part
    stats = dict(k=k, seed=seed, edge_cut=objval, edge_cut_fraction=objval / (adjncy.size // 2),
                 imbalance=round(imbalance, 4), max_part_gb=max(q["feature_gb"] for q in parts),
                 within_part_edge_fraction=locality, parts=parts,
                 metis_minutes=round((time.time()-t1)/60, 1), smoke=g.get("smoke", False))
    with open(os.path.join(out_dir, f"part_k{k}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[part] k={k}: max part {stats['max_part_gb']} GB, imbalance {imbalance:.3f}, "
          f"edge cut {stats['edge_cut_fraction']*100:.2f}%, within-part edges: "
          + ", ".join(f"{a} {b*100:.1f}%" for a, b in locality.items()), flush=True)
    for q in parts:
        print(f"        part {q['part']}: {q['feature_gb']:6.1f} GB  papers {q['papers']:,}  "
              f"authors {q['authors']:,}  inst {q['institutions']:,}", flush=True)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csr", default=f"{ROOT}/csr")
    p.add_argument("--hetero", default=f"{ROOT}/csr_hetero")
    p.add_argument("--out", default=f"{ROOT}/metis")
    p.add_argument("--metis-home", default=os.environ.get("METIS_HOME", os.path.expanduser("~/local")))
    p.add_argument("--build", action="store_true", help="build and save the global graph")
    p.add_argument("--k", type=int, default=0, help="partition the saved graph into k parts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="build + k=4 on a small slice")
    a = p.parse_args()

    metis = Metis(a.metis_home)
    if a.smoke:
        out = a.out.rstrip("/") + "_smoke"
        build_graph(a.csr, a.hetero, out, smoke=True)
        partition(metis, out, 4, seed=a.seed)
        print("[smoke] done", flush=True)
        return
    if a.build:
        build_graph(a.csr, a.hetero, a.out)
    if a.k:
        partition(metis, a.out, a.k, seed=a.seed)
    if not a.build and not a.k:
        p.error("give --build, --k K, or --smoke")


if __name__ == "__main__":
    main()
