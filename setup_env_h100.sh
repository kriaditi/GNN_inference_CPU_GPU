#!/bin/bash
# One-time env setup on the Lambda H100 (x86_64). Creates conda env "gnn".
# torch 2.8.0 + CUDA 12.8 is used because pyg_lib publishes wheels for that pair.
# (GH200 runs torch 2.11 / pyg_lib 0.9 on aarch64; versions are printed at the end for the record.)
set -e
eval "$(conda shell.bash hook)"
conda create -y -n gnn python=3.11
conda activate gnn
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install torch_geometric ogb numpy pynvml

python - <<'EOF'
import torch, pyg_lib, torch_geometric, ogb, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("pyg_lib", pyg_lib.__version__, "torch_geometric", torch_geometric.__version__,
      "ogb", ogb.__version__, "numpy", numpy.__version__)
x = torch.ones(1024, device="cuda"); print("cuda test ok:", float(x.sum()))
EOF
