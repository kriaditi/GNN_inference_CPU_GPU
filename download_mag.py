from ogb.lsc import MAG240MDataset
import time, sys

ROOT = sys.argv[1]
t0 = time.time()
ds = MAG240MDataset(root=ROOT)
print("papers      :", ds.num_papers, flush=True)
print("feat dim    :", ds.num_paper_features, flush=True)
print("classes     :", ds.num_classes, flush=True)
print("elapsed min :", (time.time() - t0) / 60, flush=True)
