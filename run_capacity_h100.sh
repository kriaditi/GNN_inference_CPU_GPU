#!/bin/bash
# Lambda H100 (PCIe, 80 GB GPU, 221 GB RAM), 3 Nov: latency vs capacity, PCIe line.
# 1. Build the subsets here (unions of METIS k=16 shards 0..n-1), same code and inputs as the GH200.
# 2. For each subset: GraphSAGE (node filtering), all six fanouts x 1,000 batches, no accuracy pass.
#    --in-memory reads the subset's feature table from disk into RAM BEFORE timing starts, so the
#    timed batches never touch the disk: each batch is RAM -> GPU over PCIe.
#    Sizes 1-4 (23 to 94 GB) fit in RAM. Size 8 (188 GB + graph) is tried once; if it does not fit,
#    that process fails alone and the log says so.
#   tmux new -s cap ; bash run_capacity_h100.sh ; (Ctrl-b d)

cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
W=$D/results/h100het_378025/best.pt
OUT=results_capacity_h100
SWEEP="1024:40/20:1000,1024:5/5:1000,1024:10/5:1000,1024:15/10:1000,1024:25/15:1000,1024:60/30:1000"
mkdir -p $OUT logs_h100
LOG=logs_h100/run_capacity.log
{ echo "start $(date)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,pcie.link.gen.current,pcie.link.width.current --format=csv
  free -g | head -2; nproc; lscpu | grep "Model name"; df -h $D | tail -1
  md5sum build_subset.py sage_mag240m.py hetero_common.py $D/metis/part_k16.npy $W; } 2>&1 | tee $LOG

# 1. build (skipped if already built)
if [ ! -f $D/subsets/s4/subset_info.json ]; then
  python build_subset.py --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy \
    --metis $D/metis --k 16 --sizes 1,2,3,4 --out $D/subsets 2>&1 | tee logs_h100/build_subset.log
  echo "build done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
fi

# 2. GraphSAGE (node filtering) per subset; wait for an empty GPU before each
for N in 1 2 3 4; do
  S=$D/subsets/s$N
  for i in $(seq 1 20); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$U" -lt 1000 ] && break; sleep 60; done
  echo "== s$N: GPU used ${U} MiB, free RAM $(free -g | awk '/Mem:/{print $7}') GB  $(date)" | tee -a $LOG
  python sage_mag240m.py --hetero --weights $W --root $D/data --csr $S/csr --hetero-dir $S/csr_hetero --feat $S/full_feat.npy \
    --device cuda --seeds 20000000 --seed-mode all --layers 2 --fanout 40,20 --batch-size 1024 --hidden 256 \
    --in-memory --no-val --sweep "$SWEEP" \
    --out $OUT --summary $OUT/summary_capacity_h100.csv --tag h100cap_s${N}_nf 2>&1 | tee logs_h100/cap_s${N}_nf.log
  echo "  s$N exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
done
echo "all done $(date)" | tee -a $LOG
