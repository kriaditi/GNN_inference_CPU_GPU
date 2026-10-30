#!/bin/bash
# Lambda H100 (PCIe, 80 GB, 221 GB RAM): GPU-only METIS shard runs, drop variant only.
# Feature table stays memmapped (375 GB > 221 GB RAM). Drop never reads the full
# table during batches, so only the one-time shard load touches disk (reported separately).
#
#   RUN A  k=16, all 16 shards, 63 timed batches each, 6 fanouts  (same settings as GH200 drop)
#   RUN B  GPU-only ladder: largest shard of k = 16, 8, 6, 5; 1,000 timed batches;
#          one process per (k, fanout) so an out-of-memory at a big fanout keeps the rest.
#
# Run inside tmux:  tmux new -s runs ; bash run_gpuonly_h100.sh ; (Ctrl-b d to detach)

cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy --metis $D/metis"
FANOUTS="5/5 10/5 15/10 25/15 40/20 60/30"
mkdir -p results_shard_h100 results_gpuonly_h100 logs_h100
LOG=logs_h100/run_all.log

# ---- preflight: record machine and data facts
{
  echo "start $(date)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,pcie.link.gen.current,pcie.link.width.current --format=csv
  free -g; nproc; lscpu | grep "Model name"
  python -c "import torch, pyg_lib; print('torch', torch.__version__, 'pyg_lib', pyg_lib.__version__)"
  md5sum shard_infer.py hetero_common.py $D/metis/part_k*.npy $D/results/h100het_378025/best.pt
  stat -c "%s %n" $D/data/full_feat.npy
} 2>&1 | tee $LOG

# ---- RUN A: k=16, all shards, drop, same settings as GH200
python shard_infer.py $PATHS --k 16 --variants drop --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 \
  --batches-per-shard 63 --out results_shard_h100 --summary results_shard_h100/summary_shard_h100_drop.csv \
  --tag h100_shard 2>&1 | tee logs_h100/runA_shard_k16.log
echo "RUN A k16 done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG

# ---- RUN B: GPU-only ladder, largest shard of each k
for K in 6 5; do
  P=$(python -c "import numpy as np; c=np.bincount(np.load('$D/metis/part_k$K.npy').astype(np.int64)); print(int(c.argmax()))")
  echo "k=$K largest shard p=$P" | tee -a $LOG
  for F in $FANOUTS; do
    python shard_infer.py $PATHS --k $K --shards $P --variants drop --fanouts $F \
      --batches-per-shard 1000 --out results_gpuonly_h100 --summary results_gpuonly_h100/summary_gpuonly_h100.csv \
      --tag h100_gpuonly_k${K}_f${F/\//x} 2>&1 | tee logs_h100/runB_k${K}_f${F/\//x}.log
    echo "RUN B k$K p$P fanout $F done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
  done
done
echo "all done $(date)" | tee -a $LOG
