#!/bin/bash
# GH200 follow-ups (29 Oct), same scripts as the overnight run:
#   1. streaming "auto" only, every fanout, chunk 8 and 24 GB, one fresh process per
#      fanout so auto reads free GPU memory without cached blocks from G=64 runs
#   2. GPU-only point for the GH200: largest shard of k=8 (47.4 GB), drop, 1,000 batches,
#      6 fanouts, table memmapped (no 19 GB GPU cost), one process per fanout
# Run inside tmux:  tmux new -s extra ; bash run_extra_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy"
FANOUTS="5/5 10/5 15/10 25/15 40/20 60/30"
mkdir -p results_stream_gh200_auto results_gpuonly_gh200 logs_gh200
LOG=logs_gh200/run_extra.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; } | tee $LOG

# 1. streaming auto, fresh process per fanout
for F in $FANOUTS; do
  python stream_infer.py $PATHS --in-memory --fanouts $F --batches-per-pass auto --chunk-gb 8,24 \
    --batches 1000 --out results_stream_gh200_auto --summary results_stream_gh200_auto/summary_stream_gh200_auto.csv \
    --tag gh200_stream_auto 2>&1 | tee logs_gh200/stream_auto_f${F/\//x}.log
  echo "stream auto $F done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
done

# 2. GPU-only, largest shard of k=8, drop, memmapped table
K=8
P=$(python -c "import numpy as np; c=np.bincount(np.load('$D/metis/part_k$K.npy').astype(np.int64)); print(int(c.argmax()))")
echo "k=$K largest shard p=$P" | tee -a $LOG
for F in $FANOUTS; do
  python shard_infer.py $PATHS --metis $D/metis --k $K --shards $P --variants drop --fanouts $F \
    --batches-per-shard 1000 --out results_gpuonly_gh200 --summary results_gpuonly_gh200/summary_gpuonly_gh200.csv \
    --tag gh200_gpuonly_k${K}_f${F/\//x} 2>&1 | tee logs_gh200/gpuonly_k${K}_f${F/\//x}.log
  echo "gpuonly k$K p$P fanout $F done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
done
echo "all done $(date)" | tee -a $LOG
