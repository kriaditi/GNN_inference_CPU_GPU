#!/bin/bash
# GH200, 6 Nov (Udit's question): the 23 GB shard copy OVERLAPPED with the CPU work of the same batch.
# Same as run_swap_gh200.sh (k=16 ordered table, shard copied RAM -> GPU before every batch) but the copy runs on a
# side CUDA stream while the CPU samples, filters and fetches the out-of-shard rows; the batch waits for the copy
# only at the in-shard gather. Columns: swap_ms = the copy's own duration (CUDA events), swap_wait_ms = the part
# still outstanding when the GPU needed the shard (the part that was not hidden). wall_ms includes everything.
#   1. smoke  2 shards x 4 batches, 40/20, no accuracy pass     ~3 min
#   2. full   16 shards x 63 batches, 6 fanouts, accuracy pass  ~20 min
#   tmux new -s ovl ; bash run_swap_overlap_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=/unicorn/gnn
MO=$D/metis_ordered
PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero"
ORD="--feat $MO/full_feat_metis_k16.npy --row-map $MO/rowmap_k16.npy --chunk-bounds $MO/bounds_k16.npy"
OUT=results_shard_swap_gh200
mkdir -p $OUT logs_gh200
LOG=logs_gh200/run_swap_overlap.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; md5sum shard_infer.py; } | tee -a $LOG
[ -f $MO/info_k16.json ] || { echo "ordered table k=16 missing" | tee -a $LOG; exit 1; }
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

# 1. smoke
numactl --membind=0 python shard_infer.py $PATHS $ORD --metis $D/metis --k 16 --in-memory --swap-per-batch --swap-overlap \
  --shards 0,1 --fanouts 40/20 --variants fetch --batches-per-shard 4 --no-val \
  --out $OUT/smoke --summary $OUT/smoke/summary_smoke_overlap.csv --tag gh200_swapovl_smoke 2>&1 | tee logs_gh200/swap_overlap_smoke.log
echo "smoke done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
grep -q "^\[summary\]" logs_gh200/swap_overlap_smoke.log || { echo "smoke failed, stopping" | tee -a $LOG; exit 1; }

# 2. full run
numactl --membind=0 python shard_infer.py $PATHS $ORD --metis $D/metis --k 16 --in-memory --swap-per-batch --swap-overlap \
  --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 --variants fetch --batches-per-shard 63 \
  --out $OUT --summary $OUT/summary_shard_swap_overlap_gh200.csv --tag gh200_swapovl 2>&1 | tee logs_gh200/swap_overlap_k16.log
echo "swap overlap k16 done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
echo "all done $(date)" | tee -a $LOG
