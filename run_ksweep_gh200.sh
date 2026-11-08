#!/bin/bash
# GH200: shard size sweep, k = 32 and 64 (Udit, 6 Nov): "if not the 23 GB shard, smaller k: do the numbers change,
# what is the downside, where is the breakpoint". For each k, exactly the two runs we have for k = 16:
#   resident : shard resident on the GPU, out-of-shard rows fetched from RAM (like results_shard_gh200, fetch)
#   swap     : the same plus the shard load from RAM to HBM inside every batch (like results_shard_swap_gh200)
#   swapovl  : the same load overlapped with the CPU steps (like summary_shard_swap_overlap_gh200)
# 6 fanouts, fetch variant, ~1,024 timed batches per cell (k x batches_per_shard), accuracy on all validation
# papers. Table written once per k in METIS order (375 GB each; 1.6 TB free on 6 Nov).
# Needs /unicorn/gnn/metis/part_k32.npy and part_k64.npy (from the cluster job metis_part_k32_64.sub).
#   tmux new -s ksweep ; bash run_ksweep_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=/unicorn/gnn
MO=$D/metis_ordered
PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero"
OUT=results_ksweep_gh200
mkdir -p $OUT logs_gh200
LOG=logs_gh200/run_ksweep.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; md5sum shard_infer.py build_metis_ordered_feat.py; df -h $D/data | tail -1; } | tee -a $LOG
for K in ${KS:-32 64}; do [ -f $D/metis/part_k$K.npy ] || { echo "missing $D/metis/part_k$K.npy" | tee -a $LOG; exit 1; }; done

# do not start while another GPU run (the k=16 swap run) is still going
while pgrep -f "shard_infer.py" > /dev/null; do sleep 120; done

for K in ${KS:-32 64}; do
  BPS=$((1024 / K))                      # batches per shard: 32 -> 32, 64 -> 16 (1,024 batches per cell)
  ORD="--feat $MO/full_feat_metis_k$K.npy --row-map $MO/rowmap_k$K.npy --chunk-bounds $MO/bounds_k$K.npy"
  if [ -f $MO/info_k$K.json ]; then
    echo "ordered table k=$K present" | tee -a $LOG
  else
    python build_metis_ordered_feat.py --feat $D/data/full_feat.npy --metis $D/metis --k $K --out $MO 2>&1 | tee logs_gh200/metis_order_k$K.log
    echo "build k=$K done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
    [ -f $MO/info_k$K.json ] || { echo "build k=$K failed, stopping" | tee -a $LOG; exit 1; }
  fi
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

  # resident
  numactl --membind=0 python shard_infer.py $PATHS $ORD --metis $D/metis --k $K --in-memory \
    --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 --variants fetch --batches-per-shard $BPS \
    --out $OUT --summary $OUT/summary_ksweep_gh200.csv --tag gh200_k${K}_resident 2>&1 | tee logs_gh200/ksweep_k${K}_resident.log
  echo "k=$K resident done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

  # shard load inside every batch
  numactl --membind=0 python shard_infer.py $PATHS $ORD --metis $D/metis --k $K --in-memory --swap-per-batch \
    --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 --variants fetch --batches-per-shard $BPS \
    --out $OUT --summary $OUT/summary_ksweep_gh200.csv --tag gh200_k${K}_swap 2>&1 | tee logs_gh200/ksweep_k${K}_swap.log
  echo "k=$K swap done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

  # shard load inside every batch, overlapped with the CPU steps
  numactl --membind=0 python shard_infer.py $PATHS $ORD --metis $D/metis --k $K --in-memory --swap-per-batch --swap-overlap \
    --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 --variants fetch --batches-per-shard $BPS \
    --out $OUT --summary $OUT/summary_ksweep_gh200.csv --tag gh200_k${K}_swapovl 2>&1 | tee logs_gh200/ksweep_k${K}_swapovl.log
  echo "k=$K swap overlapped done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
done
echo "all done $(date)" | tee -a $LOG
