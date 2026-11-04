#!/bin/bash
# Lambda H100, 4 Nov: replacement d3 for the density curve. The sampling check showed d3 {2,5,11} above d4 {2,5,11,0}
# at every fanout, so the third shard becomes 0 instead of 11: dalt3 = shards {2, 5, 0} (d4 stays the same set).
#   1. order file with positions 3 and 4 swapped -> build dalt3 only (70 GB) into $D/subsets_density/dalt3
#   2. sampling check on dalt3 (CPU) -> gate_d3alt.py: must lie between d2 and d4 at every fanout
#   3. only if the gate passes: GraphSAGE timed run, six fanouts x 1,000 batches (same as the other d's)
#   tmux new -s alt ; bash run_d3alt_h100.sh ; (Ctrl-b d)

cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
W=$D/results/h100het_378025/best.pt
OUT=results_capacity_density_h100
SWEEP="1024:40/20:1000,1024:5/5:1000,1024:10/5:1000,1024:15/10:1000,1024:25/15:1000,1024:60/30:1000"
ALT=results_capacity_order/order_density_alt_k16.json
LOG=logs_h100/run_d3alt.log
mkdir -p logs_h100

python -c "import json; o=json.load(open('results_capacity_order/order_density_k16.json'))['order']; o[2],o[3]=o[3],o[2]; json.dump(dict(k=16, order=o, rule='density order with positions 3 and 4 swapped (d3 sampled more nodes per batch than d4)'), open('$ALT','w'), indent=1)"
{ echo "start $(date)"; cat $ALT; echo; md5sum build_subset.py sample_check.py gate_d3alt.py $ALT; df -h $D | tail -1; } | tee $LOG

# 1. build dalt3 (skipped if already built)
if [ ! -f $D/subsets_density/dalt3/subset_info.json ]; then
  python build_subset.py --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy \
    --metis $D/metis --k 16 --order $ALT --name-prefix dalt --sizes 3 \
    --out $D/subsets_density 2>&1 | tee logs_h100/build_subset_dalt3.log
  RC=${PIPESTATUS[0]}
  echo "build dalt3 done $(date) exit $RC" | tee -a $LOG
  [ $RC -eq 0 ] || { echo "build failed, stopping" | tee -a $LOG; exit 1; }
fi

# 2. sampling check + gate
python sample_check.py --subsets $D/subsets_density/dalt3 --out results_capacity_order/sample_check_dalt3_h100.csv \
  2>&1 | tee logs_h100/sample_check_dalt3.log
python gate_d3alt.py results_capacity_order/sample_check_dalt3_h100.csv dalt3 2>&1 | tee -a $LOG
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "gate failed: no timed run $(date)" | tee -a $LOG; exit 1; }

# 3. GraphSAGE timed run
S=$D/subsets_density/dalt3
for i in $(seq 1 20); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$U" -lt 1000 ] && break; sleep 60; done
echo "== dalt3: GPU used ${U} MiB, free RAM $(free -g | awk '/Mem:/{print $7}') GB  $(date)" | tee -a $LOG
python sage_mag240m.py --hetero --weights $W --root $D/data --csr $S/csr --hetero-dir $S/csr_hetero --feat $S/full_feat.npy \
  --device cuda --seeds 20000000 --seed-mode all --layers 2 --fanout 40,20 --batch-size 1024 --hidden 256 \
  --in-memory --no-val --sweep "$SWEEP" \
  --out $OUT --summary $OUT/summary_capacity_density_h100.csv --tag h100den_dalt3_nf 2>&1 | tee logs_h100/den_dalt3_nf.log
echo "  dalt3 exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
