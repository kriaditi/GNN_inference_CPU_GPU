#!/bin/bash
# GH200, 6 Nov: GraphSAGE + METIS with ONE SHARD SWAP PER BATCH (6 Nov meeting action item).
# The 375 GB table is written once in METIS k=16 order (each 23 GB shard one contiguous slice), so the shard
# swap is a plain copy of that slice from RAM to the GPU with no second host copy (the first attempt OOM-ed
# node 0 with a separate pinned copy). Before every batch the resident shard is re-copied host -> GPU
# (swap_ms); out-of-shard rows are fetched from RAM through the row map, as before.
#   0. build  full_feat_metis_k16.npy + rowmap_k16 + bounds_k16 (untimed, ~1 h, skipped if present)
#   1. smoke  2 shards x 4 batches, 40/20, no accuracy pass                       ~3 min
#   2. full   16 shards x 63 batches, 6 fanouts, fetch, accuracy pass              ~1.5 h
#   tmux new -s swap ; bash run_swap_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=/unicorn/gnn
MO=$D/metis_ordered
PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero"
ORD="--feat $MO/full_feat_metis_k16.npy --row-map $MO/rowmap_k16.npy --chunk-bounds $MO/bounds_k16.npy"
OUT=results_shard_swap_gh200
mkdir -p $OUT logs_gh200
LOG=logs_gh200/run_swap.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; md5sum shard_infer.py build_metis_ordered_feat.py; df -h $D/data | tail -1; } | tee -a $LOG

# 0. ordered table, k=16
if [ -f $MO/info_k16.json ]; then
  echo "ordered table k=16 present, build skipped" | tee -a $LOG
else
  python build_metis_ordered_feat.py --feat $D/data/full_feat.npy --metis $D/metis --k 16 --out $MO 2>&1 | tee logs_gh200/metis_order_k16.log
  echo "build k=16 done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
  [ -f $MO/info_k16.json ] || { echo "build failed, stopping" | tee -a $LOG; exit 1; }
fi
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

# 1. smoke
numactl --membind=0 python shard_infer.py $PATHS $ORD --metis $D/metis --k 16 --in-memory --swap-per-batch \
  --shards 0,1 --fanouts 40/20 --variants fetch --batches-per-shard 4 --no-val \
  --out $OUT/smoke --summary $OUT/smoke/summary_smoke.csv --tag gh200_swap_smoke 2>&1 | tee logs_gh200/swap_smoke.log
echo "smoke done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
grep -q "^\[summary\]" logs_gh200/swap_smoke.log || { echo "smoke failed, stopping" | tee -a $LOG; exit 1; }

# 2. full run
numactl --membind=0 python shard_infer.py $PATHS $ORD --metis $D/metis --k 16 --in-memory --swap-per-batch \
  --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 --variants fetch --batches-per-shard 63 \
  --out $OUT --summary $OUT/summary_shard_swap_gh200.csv --tag gh200_swap 2>&1 | tee logs_gh200/swap_k16.log
echo "swap k16 done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
echo "all done $(date)" | tee -a $LOG
