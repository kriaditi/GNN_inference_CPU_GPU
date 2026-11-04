#!/bin/bash
# GH200, 4 Nov: latency vs capacity (MemoryForge Fig 3 style), density-ordered subsets d<n> in /unicorn/gnn/subsets_density.
# Same as run_capacity_gh200.sh, on the new subsets. Waits for run_density_build_gh200.sh (build + sampling check) to finish.
# For every size and all six fanouts, 1,000 timed batches each, no accuracy pass:
#   CPU-only  : sage_mag240m.py --device cpu   (Grace CPU)
#   NVLink    : sage_mag240m.py --device cuda  (GraphSAGE, table in RAM)
#   GPU-only  : shard_infer.py --k 1 (whole subset on the GPU), d1-d4 only; one process per fanout,
#               so a size that does not fit fails alone and is logged.
# The full graph (16 shards) is the same in any order: the existing 1,000-batch run in results_gh200/.
#   tmux new -s den2 ; bash run_capacity_density_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
W=$D/results/h100het_378025/best.pt
OUT=results_capacity_density_gh200
SIZES="1 2 3 4 6 8 12"
FANOUTS="40/20 5/5 10/5 15/10 25/15 60/30"
SWEEP="1024:40/20:1000,1024:5/5:1000,1024:10/5:1000,1024:15/10:1000,1024:25/15:1000,1024:60/30:1000"
mkdir -p $OUT logs_gh200
LOG=logs_gh200/run_capacity_density.log

# wait for the build + sampling check to finish
until grep -q "sample check exit" logs_gh200/run_density_build.log 2>/dev/null; do sleep 120; done
for N in $SIZES; do
  [ -f $D/subsets_density/d$N/subset_info.json ] || { echo "d$N missing: build did not finish, stopping" | tee -a $LOG; exit 1; }
done

{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; free -g | head -2;
  md5sum build_subset.py shard_infer.py sage_mag240m.py hetero_common.py $W results_capacity_order/order_density_k16.json; } | tee $LOG

wait_gpu () {   # drop the OS file cache (it shows up as used GPU memory on the GH200), then wait up to 20 min for an empty GPU
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null
  for i in $(seq 1 20); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$U" -lt 1000 ] && break; sleep 60
  done
  echo "  GPU used before start: ${U} MiB  $(date)" | tee -a $LOG
}

for N in $SIZES; do
  S=$D/subsets_density/d$N
  SP="--weights $W --root $D/data --csr $S/csr --hetero-dir $S/csr_hetero --feat $S/full_feat.npy"
  echo "== size d$N: $(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["shards"], d["nodes"], "nodes,", round(d["feat_gb"], 1), "GB")' $S/subset_info.json)" | tee -a $LOG

  # GPU-only (d1-d4): whole subset on the GPU, one fanout per process
  if [ $N -le 4 ]; then
    for F in $FANOUTS; do
      wait_gpu
      python shard_infer.py $SP --metis $S/metis --k 1 --in-memory --fanouts $F --variants drop \
        --batches-per-shard 1000 --no-val --unsorted-seeds \
        --out $OUT --summary $OUT/summary_capacity_density_gpuonly.csv --tag gh200den_d${N}_gpuonly_f${F/\//x} \
        2>&1 | tee logs_gh200/den_d${N}_gpuonly_f${F/\//x}.log
      RC=${PIPESTATUS[0]}
      echo "  d$N gpuonly $F exit $RC $(date)" | tee -a $LOG
      if [ $RC -ne 0 ] && grep -qi "out of memory" logs_gh200/den_d${N}_gpuonly_f${F/\//x}.log; then
        echo "  d$N does not fit on the GPU: skipping its other GPU-only fanouts" | tee -a $LOG; break
      fi
    done
  fi

  # NVLink: GraphSAGE, all fanouts in one process (table loaded once)
  wait_gpu
  python sage_mag240m.py --hetero $SP --device cuda --seeds 20000000 --seed-mode all --layers 2 \
    --fanout 40,20 --batch-size 1024 --hidden 256 --in-memory --no-val --sweep "$SWEEP" \
    --out $OUT --summary $OUT/summary_capacity_density_nvlink.csv --tag gh200den_d${N}_nf \
    2>&1 | tee logs_gh200/den_d${N}_nf.log
  echo "  d$N nvlink exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG

  # CPU-only on Grace, all fanouts in one process
  python sage_mag240m.py --hetero $SP --device cpu --seeds 20000000 --seed-mode all --layers 2 \
    --fanout 40,20 --batch-size 1024 --hidden 256 --in-memory --no-val --sweep "$SWEEP" \
    --out $OUT --summary $OUT/summary_capacity_density_cpu.csv --tag gh200den_d${N}_cpu \
    2>&1 | tee logs_gh200/den_d${N}_cpu.log
  echo "  d$N cpu exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
done
echo "all done $(date)" | tee -a $LOG
