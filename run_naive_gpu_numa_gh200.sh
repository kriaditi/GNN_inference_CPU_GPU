#!/bin/bash
# GH200, 4 Nov evening: naive GPU rerun with the process memory pinned to the Grace CPU memory.
# numactl -H: node 0 = Grace memory (441,526 MB, 64 CPUs), node 1 = GPU memory (96,768 MB, no CPUs).
# The first run fit only at 40/20: at the other 5 fanouts the 77.3 GB part buffer (71.96 GiB) could not be
# allocated although 80.9 to 82.2 GiB showed free. Inference: part of the 375 GB table was placed in node 1.
# Here: numactl --membind=0 (all host memory on node 0) and PyTorch expandable segments. All 6 fanouts,
# same settings as run_naive_gpu_gh200.sh, own summary file, so the first 40/20 run stays for comparison.
#   tmux new -s numa ; bash run_naive_gpu_numa_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=/unicorn/gnn
MO=$D/metis_ordered
OUT=results_naive_gpu_gh200
LOG=logs_gh200/run_naive_gpu_numa.log
{ echo "start $(date)"; numactl -H | grep -E "size|free"; nvidia-smi --query-gpu=memory.used --format=csv
  md5sum stream_infer.py hetero_common.py $MO/rowmap_k5.npy $MO/bounds_k5.npy; } 2>&1 | tee $LOG

wait_gpu () {   # drop the OS file cache, then wait up to 20 min for an empty GPU
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null
  for i in $(seq 1 20); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$U" -lt 1000 ] && break; sleep 60
  done
  echo "  GPU used before start: ${U} MiB, node 1 free $(numactl -H | awk '/node 1 free/{print $4}') MB  $(date)" | tee -a $LOG
}

PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero"
for F in 40/20 5/5 10/5 15/10 25/15 60/30; do
  wait_gpu
  numactl --membind=0 python stream_infer.py $PATHS --feat $MO/full_feat_metis_k5.npy --row-map $MO/rowmap_k5.npy \
    --chunk-bounds $MO/bounds_k5.npy --in-memory --drop-caches --fanouts $F --batches-per-pass 1 --val --val-G 1 \
    --batches 1000 --out $OUT --summary $OUT/summary_naive_gpu_gh200_numa.csv --tag gh200_naivegpu_numa \
    2>&1 | tee logs_gh200/naivegpu_numa_f${F/\//x}.log
  echo "naive GPU (numa) $F exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
done
echo "all done $(date)" | tee -a $LOG
