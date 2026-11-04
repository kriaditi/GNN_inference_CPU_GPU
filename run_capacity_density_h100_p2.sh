#!/bin/bash
# Lambda H100, 4 Nov: latency vs capacity, density-ordered subsets, phase 2 of 2.
# The disk cannot hold d6 (139.7 GB) and d8 (187.0 GB) next to d1-d4, so after phase 1 has finished this script:
#   deletes d1-d4 -> builds d6 alone, runs it -> deletes d6 -> builds d8 alone, runs it.
# d8 needs about 187 GB of features + the graph in RAM (old s8, 187.5 GB, fit); if it does not fit,
# the run fails alone and the log says so.
#   tmux new -s den2 ; bash run_capacity_density_h100_p2.sh ; (Ctrl-b d)

cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
W=$D/results/h100het_378025/best.pt
ORDER=results_capacity_order/order_density_k16.json
OUT=results_capacity_density_h100
SWEEP="1024:40/20:1000,1024:5/5:1000,1024:10/5:1000,1024:15/10:1000,1024:25/15:1000,1024:60/30:1000"
LOG=logs_h100/run_capacity_density.log

# phase 1 must be complete: every size 1-4 has its six summary rows
for N in 1 2 3 4; do
  C=$(grep -c "^h100den_d${N}_nf_" $OUT/summary_capacity_density_h100.csv 2>/dev/null)
  [ "${C:-0}" -ge 6 ] || { echo "d$N has ${C:-0} of 6 summary rows: phase 1 not finished, stopping"; exit 1; }
done
echo "phase 2 start $(date)" | tee -a $LOG

PREV="1 2 3 4"
for N in 6 8; do
  for P in $PREV; do
    [ -f $D/subsets_density/d$P/subset_info.json ] && cp $D/subsets_density/d$P/subset_info.json logs_h100/subset_info_d$P.json
    rm -rf $D/subsets_density/d$P
  done
  df -h $D | tail -1 | tee -a $LOG
  python build_subset.py --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy \
    --metis $D/metis --k 16 --order $ORDER --name-prefix d --sizes $N \
    --out $D/subsets_density 2>&1 | tee logs_h100/build_subset_density_d$N.log
  RC=${PIPESTATUS[0]}
  echo "build d$N done $(date) exit $RC" | tee -a $LOG
  [ $RC -eq 0 ] || { echo "build d$N failed, stopping" | tee -a $LOG; exit 1; }

  S=$D/subsets_density/d$N
  for i in $(seq 1 20); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$U" -lt 1000 ] && break; sleep 60; done
  echo "== d$N: GPU used ${U} MiB, free RAM $(free -g | awk '/Mem:/{print $7}') GB  $(date)" | tee -a $LOG
  python sage_mag240m.py --hetero --weights $W --root $D/data --csr $S/csr --hetero-dir $S/csr_hetero --feat $S/full_feat.npy \
    --device cuda --seeds 20000000 --seed-mode all --layers 2 --fanout 40,20 --batch-size 1024 --hidden 256 \
    --in-memory --no-val --sweep "$SWEEP" \
    --out $OUT --summary $OUT/summary_capacity_density_h100.csv --tag h100den_d${N}_nf 2>&1 | tee logs_h100/den_d${N}_nf.log
  echo "  d$N exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
  PREV=$N
done
cp $D/subsets_density/d8/subset_info.json logs_h100/subset_info_d8.json 2>/dev/null
echo "phase 2 done $(date)" | tee -a $LOG
