#!/bin/bash
# GH200 overnight follow-ups (29 Oct), same scripts and configs as before:
#   1. shard k=16 drop+fetch, full repeat (run-to-run variance)           ~16 min
#   2. sustained anchor 40/20 x 6,000 batches: shard drop+fetch           ~15 min
#      (375 batches per shard x 16 = 6,000)
#   3. sustained anchor 40/20 x 6,000 batches: streaming G=64 and G=1     ~2.5 h
#   4. GPU-only point with the 30 GB cleared: largest shard of k=5 (77.3 GB),
#      drop, memmapped table, 1,000 batches, one process per fanout       ~2.5 h
# Run after run_extra_gh200.sh finishes (one GPU job at a time):
#   tmux new -s sustained ; bash run_sustained_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy"
FANOUTS="5/5 10/5 15/10 25/15 40/20 60/30"
mkdir -p results_shard_gh200_rep2 results_sustained_gh200 results_gpuonly_gh200 logs_gh200
LOG=logs_gh200/run_sustained.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; } | tee $LOG

# 1. shard k=16 repeat
python shard_infer.py $PATHS --metis $D/metis --k 16 --in-memory \
  --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 --variants drop,fetch --batches-per-shard 63 \
  --out results_shard_gh200_rep2 --summary results_shard_gh200_rep2/summary_shard_gh200_rep2.csv --tag gh200_shard_rep2 \
  2>&1 | tee logs_gh200/shard_k16_rep2.log
echo "shard k16 repeat done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG

# 2. shard sustained 40/20, 6,000 batches (375 per shard)
python shard_infer.py $PATHS --metis $D/metis --k 16 --in-memory \
  --fanouts 40/20 --variants drop,fetch --batches-per-shard 375 \
  --out results_sustained_gh200 --summary results_sustained_gh200/summary_shard_gh200_6000.csv --tag gh200_shard_6000 \
  2>&1 | tee logs_gh200/shard_k16_6000.log
echo "shard sustained done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG

# 3. streaming sustained 40/20, 6,000 batches, G=64 then G=1, chunk 24 GB
python stream_infer.py $PATHS --in-memory --fanouts 40/20 --batches-per-pass 64,1 --chunk-gb 24 \
  --batches 6000 --out results_sustained_gh200 --summary results_sustained_gh200/summary_stream_gh200_6000.csv \
  --tag gh200_stream_6000 2>&1 | tee logs_gh200/stream_6000.log
echo "stream sustained done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG

# 4. GPU-only, largest shard of k=5, drop, memmapped table
K=5
P=$(python -c "import numpy as np; c=np.bincount(np.load('$D/metis/part_k$K.npy').astype(np.int64)); print(int(c.argmax()))")
echo "k=$K largest shard p=$P" | tee -a $LOG
for F in $FANOUTS; do
  python shard_infer.py $PATHS --metis $D/metis --k $K --shards $P --variants drop --fanouts $F \
    --batches-per-shard 1000 --out results_gpuonly_gh200 --summary results_gpuonly_gh200/summary_gpuonly_gh200.csv \
    --tag gh200_gpuonly_k${K}_f${F/\//x} 2>&1 | tee logs_gh200/gpuonly_k${K}_f${F/\//x}.log
  echo "gpuonly k$K p$P fanout $F done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
done
echo "all done $(date)" | tee -a $LOG
