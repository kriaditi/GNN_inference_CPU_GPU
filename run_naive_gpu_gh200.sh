#!/bin/bash
# GH200, 4 Nov: naive GPU (Michael) = streaming with the METIS k=5 parts as the chunks.
# Every batch (G=1), all 5 parts go RAM -> GPU in turn and the GPU keeps the rows the batch needs;
# sampling on the CPU on the full graph, so every neighbour is found; accuracy pass through the same path.
#   1. reorder the table by part once (untimed): /unicorn/gnn/metis_ordered (375 GB, skipped if built)
#   2. stream_infer.py, one process per fanout, 1,000 timed batches, table in RAM (--in-memory)
#   tmux new -s naivegpu ; bash run_naive_gpu_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
MO=$D/metis_ordered
OUT=results_naive_gpu_gh200
mkdir -p $OUT logs_gh200
LOG=logs_gh200/run_naive_gpu.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; free -g | head -2; df -h /unicorn | tail -1
  md5sum stream_infer.py hetero_common.py build_metis_ordered_feat.py $D/metis/part_k5.npy; } 2>&1 | tee $LOG

# 1. reorder the table by METIS k=5 part (once)
if [ ! -f $MO/info_k5.json ]; then
  python build_metis_ordered_feat.py --feat $D/data/full_feat.npy --metis $D/metis --k 5 --out $MO \
    2>&1 | tee logs_gh200/build_metis_ordered_k5.log
  RC=${PIPESTATUS[0]}
  echo "reorder done $(date) exit $RC" | tee -a $LOG
  [ $RC -eq 0 ] || { echo "reorder failed, stopping" | tee -a $LOG; exit 1; }
fi
cat $MO/info_k5.json | tee -a $LOG

wait_gpu () {   # drop the OS file cache (shows up as used GPU memory on the GH200), then wait up to 20 min for an empty GPU
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null
  for i in $(seq 1 20); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$U" -lt 1000 ] && break; sleep 60
  done
  echo "  GPU used before start: ${U} MiB  $(date)" | tee -a $LOG
}

# 2. naive GPU, one process per fanout
PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero"
for F in 40/20 5/5 10/5 15/10 25/15 60/30; do
  wait_gpu
  python stream_infer.py $PATHS --feat $MO/full_feat_metis_k5.npy --row-map $MO/rowmap_k5.npy --chunk-bounds $MO/bounds_k5.npy \
    --in-memory --drop-caches --fanouts $F --batches-per-pass 1 --val --val-G 1 --batches 1000 \
    --out $OUT --summary $OUT/summary_naive_gpu_gh200.csv --tag gh200_naivegpu \
    2>&1 | tee logs_gh200/naivegpu_f${F/\//x}.log
  echo "naive GPU $F exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
done
echo "all done $(date)" | tee -a $LOG
