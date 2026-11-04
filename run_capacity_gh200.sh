#!/bin/bash
# GH200, 3 Nov: latency vs capacity (MemoryForge Fig 3 style) for the GNN workload.
# Graph sizes = unions of METIS k=16 shards 0..n-1, built by build_subset.py into $D/subsets/s<n>.
# For every size and all six fanouts, 1,000 timed batches each, no accuracy pass:
#   CPU-only  : sage_mag240m.py --device cpu   (Grace CPU)
#   NVLink    : sage_mag240m.py --device cuda  (GraphSAGE / node filtering, table in RAM)
#   GPU-only  : shard_infer.py --k 1 (the whole subset held on the GPU), sizes 1-4 only;
#               one process per fanout, so a size that does not fit fails alone and is logged.
# The full graph (16 shards) is the existing 1,000-batch run in results_gh200/.
# Same seeds everywhere: default_rng(0) over the subset's papers (--unsorted-seeds for GPU-only).
# Before each GPU process the script waits until the GPU shows < 1 GB used (up to 20 min).
#   tmux new -s cap ; bash run_capacity_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
W=$D/results/h100het_378025/best.pt
OUT=results_capacity_gh200
SIZES="1 2 3 4 8"
FANOUTS="40/20 5/5 10/5 15/10 25/15 60/30"
SWEEP="1024:40/20:1000,1024:5/5:1000,1024:10/5:1000,1024:15/10:1000,1024:25/15:1000,1024:60/30:1000"
mkdir -p $OUT logs_gh200
LOG=logs_gh200/run_capacity.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; free -g | head -2;
  md5sum build_subset.py shard_infer.py sage_mag240m.py hetero_common.py $W; } | tee $LOG

wait_gpu () {   # wait up to 20 min for an empty GPU, then log what it shows
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null   # file cache shows as used GPU memory on the GH200
  for i in $(seq 1 20); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$U" -lt 1000 ] && break; sleep 60
  done
  echo "  GPU used before start: ${U} MiB  $(date)" | tee -a $LOG
}

# 1. build the subsets once (skipped if s8 already has its info file)
if [ ! -f $D/subsets/s8/subset_info.json ]; then
  python build_subset.py --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy \
    --metis $D/metis --k 16 --sizes 1,2,3,4,8 --out $D/subsets 2>&1 | tee logs_gh200/build_subset.log
  echo "build done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
fi

for N in $SIZES; do
  S=$D/subsets/s$N
  SP="--weights $W --root $D/data --csr $S/csr --hetero-dir $S/csr_hetero --feat $S/full_feat.npy"
  echo "== size s$N: $(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["nodes"], "nodes,", round(d["feat_gb"], 1), "GB")' $S/subset_info.json)" | tee -a $LOG

  # 2a. GPU-only (sizes 1 to 4): whole subset on the GPU, one fanout per process
  if [ $N -le 4 ]; then
    for F in $FANOUTS; do
      wait_gpu
      python shard_infer.py $SP --metis $S/metis --k 1 --in-memory --fanouts $F --variants drop \
        --batches-per-shard 1000 --no-val --unsorted-seeds \
        --out $OUT --summary $OUT/summary_capacity_gpuonly.csv --tag gh200cap_s${N}_gpuonly_f${F/\//x} \
        2>&1 | tee logs_gh200/cap_s${N}_gpuonly_f${F/\//x}.log
      RC=${PIPESTATUS[0]}
      echo "  s$N gpuonly $F exit $RC $(date)" | tee -a $LOG
      if [ $RC -ne 0 ] && grep -qi "out of memory" logs_gh200/cap_s${N}_gpuonly_f${F/\//x}.log; then
        echo "  s$N does not fit on the GPU: skipping its other GPU-only fanouts" | tee -a $LOG; break
      fi
    done
  fi

  # 2b. NVLink: GraphSAGE / node filtering, all fanouts in one process (table loaded once)
  wait_gpu
  python sage_mag240m.py --hetero $SP --device cuda --seeds 20000000 --seed-mode all --layers 2 \
    --fanout 40,20 --batch-size 1024 --hidden 256 --in-memory --no-val --sweep "$SWEEP" \
    --out $OUT --summary $OUT/summary_capacity_nvlink.csv --tag gh200cap_s${N}_nf \
    2>&1 | tee logs_gh200/cap_s${N}_nf.log
  echo "  s$N nvlink exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG

  # 2c. CPU-only on Grace, all fanouts in one process
  python sage_mag240m.py --hetero $SP --device cpu --seeds 20000000 --seed-mode all --layers 2 \
    --fanout 40,20 --batch-size 1024 --hidden 256 --in-memory --no-val --sweep "$SWEEP" \
    --out $OUT --summary $OUT/summary_capacity_cpu.csv --tag gh200cap_s${N}_cpu \
    2>&1 | tee logs_gh200/cap_s${N}_cpu.log
  echo "  s$N cpu exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
done
echo "all done $(date)" | tee -a $LOG
