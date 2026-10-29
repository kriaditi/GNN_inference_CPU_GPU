#!/bin/bash
# Overnight runs on the GH200 (no Slurm). Run inside tmux:
#   tmux new -s runs ; bash run_all_gh200.sh ; (Ctrl-b d to detach)
# One run at a time: every run holds the 375 GB table in RAM (476 GB available).
# Each step writes its own log; a failed step does not stop the next one.

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy"
mkdir -p results_shard_gh200 results_stream_gh200 logs_gh200
echo "start $(date)" | tee logs_gh200/run_all.log

# 1. METIS shard, k=16, all 16 shards, both variants (drop, fetch), six fanouts,
#    63 timed batches per shard per config (= 1,008 batches, ~1,000 like the baselines)
python shard_infer.py $PATHS --metis $D/metis --k 16 --in-memory \
  --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 --variants drop,fetch --batches-per-shard 63 \
  --out results_shard_gh200 --summary results_shard_gh200/summary_shard_gh200.csv --tag gh200_shard \
  2>&1 | tee logs_gh200/shard_k16.log
echo "shard k16 done $(date) exit ${PIPESTATUS[0]}" | tee -a logs_gh200/run_all.log

# 2. Streaming: chunk 8 GB and 47 GB; G = 1, 8, 64, auto batches per full pass;
#    1,000 timed batches per config. One fanout per process so finished
#    fanouts are saved even if a later one is stopped. 40,20 first.
for F in 40/20 5/5 60/30 15/10 10/5 25/15; do
  python stream_infer.py $PATHS --in-memory --fanouts $F --batches-per-pass 1,8,64,auto --chunk-gb 8,24 \
    --batches 1000 --out results_stream_gh200 --summary results_stream_gh200/summary_stream_gh200.csv \
    --tag gh200_stream 2>&1 | tee logs_gh200/stream_f${F/\//x}.log
  echo "stream $F done $(date) exit ${PIPESTATUS[0]}" | tee -a logs_gh200/run_all.log
done
echo "all done $(date)" | tee -a logs_gh200/run_all.log
