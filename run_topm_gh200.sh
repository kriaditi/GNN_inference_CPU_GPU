#!/bin/bash
# GH200, 30 Oct: "search additional shards". k=16, home shard resident; per batch keep the
# home shard plus the M-1 most-hit shards (rows fetched from RAM), drop the rest.
# Variants top2, top3, top4, top8 sit between drop (M=1) and fetch (M=16); top16 must
# reproduce fetch and is included as the check. 63 batches per shard, 6 fanouts, accuracy
# on all validation papers. Also records shards_hit and frac_kept per batch.
# Run after run_naive_stream_gh200.sh:  tmux new -s topm ; bash run_topm_gh200.sh

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy"
OUT=results_topm_gh200
mkdir -p $OUT logs_gh200
LOG=logs_gh200/run_topm.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; md5sum shard_infer.py; } | tee $LOG

python shard_infer.py $PATHS --metis $D/metis --k 16 --in-memory \
  --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 --variants top2,top3,top4,top8,top16 --batches-per-shard 63 \
  --out $OUT --summary $OUT/summary_topm_gh200.csv --tag gh200_topm 2>&1 | tee logs_gh200/topm_k16.log
echo "topm k16 done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
echo "all done $(date)" | tee -a $LOG
