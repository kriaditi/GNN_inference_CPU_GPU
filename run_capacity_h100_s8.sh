#!/bin/bash
# Lambda H100, phase 2 of the capacity runs (Michael, 3 Nov: run some subsets, delete them, run the rest).
# The disk (351 GB free) cannot hold all five subsets at once, so after run_capacity_h100.sh has finished
# sizes 1-4 this script: deletes s1-s4, builds s8 alone (188 GB), and runs GraphSAGE on it.
# s8 needs about 188 GB of features + ~15 GB of graph in RAM (218 GB free): if it does not fit,
# the run fails alone and the log says so.
#   tmux new -s cap8 ; bash run_capacity_h100_s8.sh ; (Ctrl-b d)

cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
W=$D/results/h100het_378025/best.pt
OUT=results_capacity_h100
SWEEP="1024:40/20:1000,1024:5/5:1000,1024:10/5:1000,1024:15/10:1000,1024:25/15:1000,1024:60/30:1000"
LOG=logs_h100/run_capacity.log

# phase 1 must be complete: every size 1-4 has its six summary rows
for N in 1 2 3 4; do
  C=$(grep -c "^h100cap_s${N}_nf_" $OUT/summary_capacity_h100.csv 2>/dev/null)
  [ "$C" -ge 6 ] || { echo "s$N has ${C:-0} of 6 summary rows: phase 1 not finished, stopping"; exit 1; }
done
echo "phase 2 start $(date)" | tee -a $LOG

for N in 1 2 3 4; do cp $D/subsets/s$N/subset_info.json logs_h100/subset_info_s$N.json; done
rm -rf $D/subsets/s1 $D/subsets/s2 $D/subsets/s3 $D/subsets/s4
df -h $D | tail -1 | tee -a $LOG

python build_subset.py --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy \
  --metis $D/metis --k 16 --sizes 8 --out $D/subsets 2>&1 | tee logs_h100/build_subset_s8.log
echo "build s8 done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG

S=$D/subsets/s8
echo "== s8: GPU used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB, free RAM $(free -g | awk '/Mem:/{print $7}') GB  $(date)" | tee -a $LOG
python sage_mag240m.py --hetero --weights $W --root $D/data --csr $S/csr --hetero-dir $S/csr_hetero --feat $S/full_feat.npy \
  --device cuda --seeds 20000000 --seed-mode all --layers 2 --fanout 40,20 --batch-size 1024 --hidden 256 \
  --in-memory --no-val --sweep "$SWEEP" \
  --out $OUT --summary $OUT/summary_capacity_h100.csv --tag h100cap_s8_nf 2>&1 | tee logs_h100/cap_s8_nf.log
echo "  s8 exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
echo "phase 2 done $(date)" | tee -a $LOG
