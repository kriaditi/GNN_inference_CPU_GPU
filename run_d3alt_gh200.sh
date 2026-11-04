#!/bin/bash
# GH200, 4 Nov: replacement d3 for the density curve: dalt3 = shards {2, 5, 0} (third shard 0 instead of 11).
# Waits until run_capacity_density_gh200.sh (den2) is completely done, so the build never overlaps a timed run.
#   1. order file with positions 3 and 4 swapped -> build dalt3 only (70 GB)
#   2. sampling check on dalt3 (CPU) -> gate_d3alt.py: must lie between d2 and d4 at every fanout
#   3. only if the gate passes: GPU-only (one process per fanout), GraphSAGE (NVLink) and CPU-only, 1,000 batches each
#   tmux new -s alt ; bash run_d3alt_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
W=$D/results/h100het_378025/best.pt
OUT=results_capacity_density_gh200
FANOUTS="40/20 5/5 10/5 15/10 25/15 60/30"
SWEEP="1024:40/20:1000,1024:5/5:1000,1024:10/5:1000,1024:15/10:1000,1024:25/15:1000,1024:60/30:1000"
ALT=results_capacity_order/order_density_alt_k16.json
LOG=logs_gh200/run_d3alt.log
mkdir -p logs_gh200

until grep -q "all done" logs_gh200/run_capacity_density.log 2>/dev/null; do sleep 120; done

python -c "import json; o=json.load(open('results_capacity_order/order_density_k16.json'))['order']; o[2],o[3]=o[3],o[2]; json.dump(dict(k=16, order=o, rule='density order with positions 3 and 4 swapped (d3 sampled more nodes per batch than d4)'), open('$ALT','w'), indent=1)"
{ echo "start $(date)"; cat $ALT; echo; md5sum build_subset.py sample_check.py gate_d3alt.py shard_infer.py sage_mag240m.py $ALT; df -h /unicorn | tail -1; } | tee $LOG

# 1. build dalt3 (skipped if already built)
if [ ! -f $D/subsets_density/dalt3/subset_info.json ]; then
  python build_subset.py --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy \
    --metis $D/metis --k 16 --order $ALT --name-prefix dalt --sizes 3 \
    --out $D/subsets_density 2>&1 | tee logs_gh200/build_subset_dalt3.log
  RC=${PIPESTATUS[0]}
  echo "build dalt3 done $(date) exit $RC" | tee -a $LOG
  [ $RC -eq 0 ] || { echo "build failed, stopping" | tee -a $LOG; exit 1; }
fi

# 2. sampling check + gate
python sample_check.py --subsets $D/subsets_density/dalt3 --out results_capacity_order/sample_check_dalt3_gh200.csv \
  2>&1 | tee logs_gh200/sample_check_dalt3.log
python gate_d3alt.py results_capacity_order/sample_check_dalt3_gh200.csv dalt3 2>&1 | tee -a $LOG
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "gate failed: no timed runs $(date)" | tee -a $LOG; exit 1; }

wait_gpu () {   # drop the OS file cache (shows up as used GPU memory on the GH200), then wait up to 20 min for an empty GPU
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null
  for i in $(seq 1 20); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$U" -lt 1000 ] && break; sleep 60
  done
  echo "  GPU used before start: ${U} MiB  $(date)" | tee -a $LOG
}

# 3. timed runs, same commands as run_capacity_density_gh200.sh
S=$D/subsets_density/dalt3
SP="--weights $W --root $D/data --csr $S/csr --hetero-dir $S/csr_hetero --feat $S/full_feat.npy"
for F in $FANOUTS; do
  wait_gpu
  python shard_infer.py $SP --metis $S/metis --k 1 --in-memory --fanouts $F --variants drop \
    --batches-per-shard 1000 --no-val --unsorted-seeds \
    --out $OUT --summary $OUT/summary_capacity_density_gpuonly.csv --tag gh200den_dalt3_gpuonly_f${F/\//x} \
    2>&1 | tee logs_gh200/den_dalt3_gpuonly_f${F/\//x}.log
  echo "  dalt3 gpuonly $F exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
done
wait_gpu
python sage_mag240m.py --hetero $SP --device cuda --seeds 20000000 --seed-mode all --layers 2 \
  --fanout 40,20 --batch-size 1024 --hidden 256 --in-memory --no-val --sweep "$SWEEP" \
  --out $OUT --summary $OUT/summary_capacity_density_nvlink.csv --tag gh200den_dalt3_nf \
  2>&1 | tee logs_gh200/den_dalt3_nf.log
echo "  dalt3 nvlink exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
python sage_mag240m.py --hetero $SP --device cpu --seeds 20000000 --seed-mode all --layers 2 \
  --fanout 40,20 --batch-size 1024 --hidden 256 --in-memory --no-val --sweep "$SWEEP" \
  --out $OUT --summary $OUT/summary_capacity_density_cpu.csv --tag gh200den_dalt3_cpu \
  2>&1 | tee logs_gh200/den_dalt3_cpu.log
echo "  dalt3 cpu exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
echo "all done $(date)" | tee -a $LOG
