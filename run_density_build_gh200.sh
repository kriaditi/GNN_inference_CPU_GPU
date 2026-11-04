#!/bin/bash
# GH200, 4 Nov: latency vs capacity, density-ordered subsets (step 1 and 2 of the agreed plan).
#   1. build d1, d2, d3, d4, d6, d8, d12 = the first n shards of results_capacity_order/order_density_k16.json
#      into /unicorn/gnn/subsets_density (about 840 GB; the old subsets in /unicorn/gnn/subsets are untouched)
#   2. sampling-only check (CPU, no GPU): nodes sampled per batch for the old s1 s2 s3 s4 s8 and the new d's,
#      all six fanouts, 1,000 batches each. The timed runs start only after we look at this together.
#   tmux new -s den ; bash run_density_build_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export PYTHONUNBUFFERED=1

D=/unicorn/gnn
ORDER=results_capacity_order/order_density_k16.json
SIZES="1 2 3 4 6 8 12"
mkdir -p logs_gh200
LOG=logs_gh200/run_density_build.log
{ echo "start $(date)"; df -h /unicorn | tail -1; free -g | head -2
  md5sum build_subset.py sample_check.py hetero_common.py $D/metis/part_k16.npy $ORDER; cat $ORDER; } 2>&1 | tee $LOG

# 1. build (skipped if d12 already has its info file)
if [ ! -f $D/subsets_density/d12/subset_info.json ]; then
  python build_subset.py --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy \
    --metis $D/metis --k 16 --order $ORDER --name-prefix d --sizes ${SIZES// /,} \
    --out $D/subsets_density 2>&1 | tee logs_gh200/build_subset_density.log
  RC=${PIPESTATUS[0]}
  echo "build done $(date) exit $RC" | tee -a $LOG
  [ $RC -eq 0 ] || { echo "build failed, stopping" | tee -a $LOG; exit 1; }
fi
df -h /unicorn | tail -1 | tee -a $LOG

# 2. sampling-only check, old subsets first (cross-check against the logged nodes per batch), then the new ones
LIST=""
for N in 1 2 3 4 8; do [ -f $D/subsets/s$N/subset_info.json ] && LIST="$LIST,$D/subsets/s$N"; done
for N in $SIZES; do LIST="$LIST,$D/subsets_density/d$N"; done
python sample_check.py --subsets "${LIST#,}" --out results_capacity_order/sample_check.csv \
  2>&1 | tee logs_gh200/sample_check.log
echo "sample check exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
