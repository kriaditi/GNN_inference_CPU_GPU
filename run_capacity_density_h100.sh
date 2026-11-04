#!/bin/bash
# Lambda H100 (PCIe, 80 GB GPU, 221 GB RAM, ~380 GB disk), 4 Nov: latency vs capacity, PCIe line,
# density-ordered subsets. Phase 1 of 2 (same pattern as run_capacity_h100.sh / _s8.sh).
#   1. build d1 d2 d3 d4 (23.4 + 46.6 + 70.0 + 93.3 = 233 GB of features) into $D/subsets_density,
#      the first n shards of results_capacity_order/order_density_k16.json (copied from the GH200)
#   2. GraphSAGE (node filtering) per subset: six fanouts x 1,000 batches, no accuracy pass.
#      --in-memory reads the table from disk into RAM BEFORE timing, so every timed batch is RAM -> GPU over PCIe.
# Phase 2 (run_capacity_density_h100_p2.sh) deletes d1-d4 and does d6, then d8.
#   tmux new -s den ; bash run_capacity_density_h100.sh ; (Ctrl-b d)

cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
W=$D/results/h100het_378025/best.pt
ORDER=results_capacity_order/order_density_k16.json
OUT=results_capacity_density_h100
SWEEP="1024:40/20:1000,1024:5/5:1000,1024:10/5:1000,1024:15/10:1000,1024:25/15:1000,1024:60/30:1000"
mkdir -p $OUT logs_h100
LOG=logs_h100/run_capacity_density.log
{ echo "phase 1 start $(date)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,pcie.link.gen.current,pcie.link.width.current --format=csv
  free -g | head -2; nproc; lscpu | grep "Model name"; df -h $D | tail -1
  md5sum build_subset.py sage_mag240m.py hetero_common.py $D/metis/part_k16.npy $W $ORDER; cat $ORDER; } 2>&1 | tee $LOG

[ -f $ORDER ] || { echo "$ORDER missing: copy it from the GH200 first" | tee -a $LOG; exit 1; }

# 1. build d1-d4 (skipped if already built); needs about 250 GB free
if [ ! -f $D/subsets_density/d4/subset_info.json ]; then
  FREE=$(df -BG --output=avail $D | tail -1 | tr -dc 0-9)
  [ "$FREE" -ge 250 ] || { echo "only ${FREE} GB free, d1-d4 need about 250 GB: free space first (old subsets?)" | tee -a $LOG; exit 1; }
  python build_subset.py --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy \
    --metis $D/metis --k 16 --order $ORDER --name-prefix d --sizes 1,2,3,4 \
    --out $D/subsets_density 2>&1 | tee logs_h100/build_subset_density_p1.log
  RC=${PIPESTATUS[0]}
  echo "build d1-d4 done $(date) exit $RC" | tee -a $LOG
  [ $RC -eq 0 ] || { echo "build failed, stopping" | tee -a $LOG; exit 1; }
fi

# 2. GraphSAGE per subset; wait for an empty GPU before each
for N in 1 2 3 4; do
  S=$D/subsets_density/d$N
  for i in $(seq 1 20); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$U" -lt 1000 ] && break; sleep 60; done
  echo "== d$N: GPU used ${U} MiB, free RAM $(free -g | awk '/Mem:/{print $7}') GB  $(date)" | tee -a $LOG
  python sage_mag240m.py --hetero --weights $W --root $D/data --csr $S/csr --hetero-dir $S/csr_hetero --feat $S/full_feat.npy \
    --device cuda --seeds 20000000 --seed-mode all --layers 2 --fanout 40,20 --batch-size 1024 --hidden 256 \
    --in-memory --no-val --sweep "$SWEEP" \
    --out $OUT --summary $OUT/summary_capacity_density_h100.csv --tag h100den_d${N}_nf 2>&1 | tee logs_h100/den_d${N}_nf.log
  echo "  d$N exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
done
echo "phase 1 done $(date)" | tee -a $LOG
