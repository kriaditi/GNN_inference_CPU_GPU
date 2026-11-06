#!/bin/bash
# Lambda H100 (PCIe, 80 GB GPU, 221 GB RAM), 5 Nov: action item 5, largest size that fits on the H100 GPU.
# GPU-only line for the PCIe machine, same as the GH200 GPU-only runs in run_capacity_density_gh200.sh:
# whole subset held on the GPU (shard_infer.py --k 1), one process per fanout, 1,000 timed batches, no accuracy pass.
# Sizes d1 (23.4 GB), d2 (46.6 GB), d3 (70.0 GB); d4 (93.3 GB) cannot fit an 80 GB GPU. A size that does not fit
# fails alone, is logged, and its other fanouts are skipped.
# --in-memory reads the subset's table from disk into RAM BEFORE anything is timed (disk time ignored); the loads
# CSV then holds the RAM -> GPU time of each subset (load_gather_s, load_transfer_s): the overhead Michael asked for.
# Disk: d1-d3 need 140 GB; delete d8 (187 GB, its runs are done) and dalt3 (70 GB, gate failed) first.
#   tmux new -s gpuonly ; bash run_gpuonly_density_h100.sh ; (Ctrl-b d)

cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
W=$D/results/h100het_378025/best.pt
ORDER=results_capacity_order/order_density_k16.json
OUT=results_capacity_density_h100
FANOUTS="40/20 5/5 10/5 15/10 25/15 60/30"
mkdir -p $OUT logs_h100
LOG=logs_h100/run_gpuonly_density.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; free -g | head -2; df -h $D | tail -1
  md5sum build_subset.py shard_infer.py hetero_common.py $W $ORDER; } 2>&1 | tee $LOG

# 1. build d1-d3 (skipped if all three exist); needs about 150 GB free
if [ ! -f $D/subsets_density/d1/subset_info.json ] || [ ! -f $D/subsets_density/d2/subset_info.json ] || [ ! -f $D/subsets_density/d3/subset_info.json ]; then
  FREE=$(df -BG --output=avail $D | tail -1 | tr -dc 0-9)
  [ "$FREE" -ge 150 ] || { echo "only ${FREE} GB free, d1-d3 need about 150 GB: delete d8 and dalt3 first" | tee -a $LOG; exit 1; }
  python build_subset.py --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy \
    --metis $D/metis --k 16 --order $ORDER --name-prefix d --sizes 1,2,3 \
    --out $D/subsets_density 2>&1 | tee logs_h100/build_subset_density_gpuonly.log
  RC=${PIPESTATUS[0]}
  echo "build d1-d3 done $(date) exit $RC" | tee -a $LOG
  [ $RC -eq 0 ] || { echo "build failed, stopping" | tee -a $LOG; exit 1; }
fi

wait_gpu () {   # wait up to 20 min for an empty GPU
  for i in $(seq 1 20); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$U" -lt 1000 ] && break; sleep 60
  done
  echo "  GPU used before start: ${U} MiB  $(date)" | tee -a $LOG
}

# 2. GPU-only per size, one process per fanout
for N in 1 2 3; do
  S=$D/subsets_density/d$N
  SP="--weights $W --root $D/data --csr $S/csr --hetero-dir $S/csr_hetero --feat $S/full_feat.npy"
  echo "== size d$N: $(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["shards"], d["nodes"], "nodes,", round(d["feat_gb"], 1), "GB")' $S/subset_info.json)" | tee -a $LOG
  for F in $FANOUTS; do
    wait_gpu
    python shard_infer.py $SP --metis $S/metis --k 1 --in-memory --fanouts $F --variants drop \
      --batches-per-shard 1000 --no-val --unsorted-seeds \
      --out $OUT --summary $OUT/summary_capacity_density_h100_gpuonly.csv --tag h100den_d${N}_gpuonly_f${F/\//x} \
      2>&1 | tee logs_h100/den_d${N}_gpuonly_f${F/\//x}.log
    RC=${PIPESTATUS[0]}
    echo "  d$N gpuonly $F exit $RC $(date)" | tee -a $LOG
    if [ $RC -ne 0 ] && grep -qi "out of memory" logs_h100/den_d${N}_gpuonly_f${F/\//x}.log; then
      echo "  d$N does not fit on the GPU at $F: skipping its other fanouts" | tee -a $LOG; break
    fi
  done
done
echo "all done $(date)" | tee -a $LOG
