#!/bin/bash
# GH200, 10 Nov (Michael): coherent GPU search at smaller shards, k = 128, 256, 512, 1024 (2.9 / 1.5 / 0.73 / 0.37 GB).
# For each k, ALL shards, the same batches through four paths (about 1,024 timed batches per cell):
#   cpuonly   : CPU gathers every row from RAM and runs the model on the CPU
#   fetch     : shard already in HBM (resident); GPU gathers in-shard rows there; CPU fetches the out-of-shard rows
#   fetchload : the same, with the shard copied RAM -> HBM inside every batch (the shard movement is in the batch time)
#   coherent  : shard stays in host RAM; GPU gathers the in-shard rows in place over NVLink-C2C; CPU fetches the rest
# 6 fanouts, accuracy on all validation papers. No 375 GB ordered table is needed for these k: each small shard is
# assembled from the original table into host memory once (untimed). If an ordered table exists for a k (8..64) it is used.
# Needs /unicorn/gnn/metis/part_k<k>.npy (cluster job metis_part_k128_1024.sub).
#   tmux new -s cohk ; bash run_coherent_k_gh200.sh ; (Ctrl-b d)
#   other k:      KS="8 16 32 64" bash run_coherent_k_gh200.sh
#   one shard:    SHARDS=0 bash run_coherent_k_gh200.sh        (default: all shards)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=4   # 12 Nov: cap glibc arenas (the k=8 run grew to 424 GB and was OOM-killed)

D=/unicorn/gnn
MO=$D/metis_ordered
PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero"
OUT=results_coherent_k_gh200
mkdir -p $OUT logs_gh200
LOG=logs_gh200/run_coherent_k.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; md5sum shard_infer.py; df -h $D/data | tail -1; } | tee -a $LOG
for K in ${KS:-128 256 512 1024}; do [ -f $D/metis/part_k$K.npy ] || { echo "missing $D/metis/part_k$K.npy" | tee -a $LOG; exit 1; }; done
while pgrep -f "shard_infer.py|shard_pipeline.py" > /dev/null; do sleep 120; done
SH=""; [ -n "$SHARDS" ] && SH="--shards $SHARDS"

feat_args() {   # ordered table if it exists for this k, else the original table
  if [ -f $MO/info_k$1.json ]; then echo "--feat $MO/full_feat_metis_k$1.npy --row-map $MO/rowmap_k$1.npy --chunk-bounds $MO/bounds_k$1.npy"
  else echo "--feat $D/data/full_feat.npy"; fi
}

# smoke: smallest k in the list, 2 shards, 4 batches, 40/20, no accuracy pass
K0=$(echo ${KS:-128 256 512 1024} | awk '{print $1}')
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
numactl --membind=0 python shard_infer.py $PATHS $(feat_args $K0) --metis $D/metis --k $K0 --in-memory \
  --shards 0,1 --fanouts 40/20 --variants cpuonly,fetch,fetchload,coherent --batches-per-shard 4 --warmup 1 --no-val \
  --out $OUT/smoke --summary $OUT/smoke/summary_smoke.csv --tag gh200_cohk_smoke 2>&1 | tee logs_gh200/coherent_k_smoke.log
echo "smoke done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
[ "$(grep -c '^\[summary\]' logs_gh200/coherent_k_smoke.log)" -ge 4 ] || { echo "smoke failed, stopping" | tee -a $LOG; exit 1; }

for K in ${KS:-128 256 512 1024}; do
  BPS=$((1024 / K)); [ $BPS -lt 1 ] && BPS=1
  [ -n "$SHARDS" ] && BPS=1000
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
  # 12 Nov: the GPU paths and CPU-only in two separate processes (same batches, same summary file)
  for V in fetch,fetchload,coherent cpuonly; do
    VN=${V%%,*}; [ "$V" = cpuonly ] && VN=cpu || VN=gpu
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    numactl --membind=0 python shard_infer.py $PATHS $(feat_args $K) --metis $D/metis --k $K --in-memory $SH \
      --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 --variants $V --batches-per-shard $BPS --warmup 1 \
      --out $OUT --summary $OUT/summary_coherent_k_gh200.csv --tag gh200_cohk_k${K} 2>&1 | tee logs_gh200/coherent_k_k${K}_${VN}.log
    echo "k=$K $VN done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
  done
done
echo "all done $(date)" | tee -a $LOG
