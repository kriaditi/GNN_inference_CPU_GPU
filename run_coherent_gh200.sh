#!/bin/bash
# GH200, 9 Nov (Michael): per shard size k = 64, 32, 16, 8, ONE shard (shard 0), the same 1,000 batches through three paths:
#   cpuonly  : CPU gathers every row from RAM and runs the model on the CPU
#   fetch    : shard copied into HBM, GPU gathers the in-shard rows there, CPU fetches the out-of-shard rows   (what we have)
#   coherent : shard stays in host RAM, GPU gathers the in-shard rows in place over NVLink-C2C (no copy, no HBM),
#              CPU fetches the out-of-shard rows
# 6 fanouts, accuracy on shard 0's validation papers. Smaller k first (Michael: coherence should help small shards most).
# Uses the METIS-ordered tables built for the k sweep. Smoke first: k=64, 8 batches, no accuracy pass; it also checks
# that the coherent view returns the host rows bit-exact (assert in shard_infer.py).
#   tmux new -s coh ; bash run_coherent_gh200.sh ; (Ctrl-b d)        one k: KS=32 bash run_coherent_gh200.sh

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=/unicorn/gnn
MO=$D/metis_ordered
PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero"
OUT=results_coherent_gh200
mkdir -p $OUT logs_gh200
LOG=logs_gh200/run_coherent.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; md5sum shard_infer.py; } | tee -a $LOG
for K in ${KS:-64 32 16 8}; do [ -f $MO/info_k$K.json ] || { echo "ordered table k=$K missing" | tee -a $LOG; exit 1; }; done
while pgrep -f "shard_infer.py" > /dev/null; do sleep 120; done
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

# smoke: coherent path, k=64 shard 0, 8 batches, 40/20
ORD="--feat $MO/full_feat_metis_k64.npy --row-map $MO/rowmap_k64.npy --chunk-bounds $MO/bounds_k64.npy"
numactl --membind=0 python shard_infer.py $PATHS $ORD --metis $D/metis --k 64 --in-memory \
  --shards 0 --fanouts 40/20 --variants coherent,fetch --batches-per-shard 8 --no-val \
  --out $OUT/smoke --summary $OUT/smoke/summary_smoke.csv --tag gh200_coh_smoke 2>&1 | tee logs_gh200/coherent_smoke.log
echo "smoke done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
grep -q "^\[summary\] gh200_coh_smoke_k64_coherent" logs_gh200/coherent_smoke.log || { echo "smoke failed, stopping" | tee -a $LOG; exit 1; }

for K in ${KS:-64 32 16 8}; do
  ORD="--feat $MO/full_feat_metis_k$K.npy --row-map $MO/rowmap_k$K.npy --chunk-bounds $MO/bounds_k$K.npy"
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
  numactl --membind=0 python shard_infer.py $PATHS $ORD --metis $D/metis --k $K --in-memory \
    --shards 0 --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 --variants cpuonly,fetch,coherent --batches-per-shard 1000 \
    --out $OUT --summary $OUT/summary_coherent_gh200.csv --tag gh200_coh_k${K} 2>&1 | tee logs_gh200/coherent_k${K}.log
  echo "k=$K done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
done
echo "all done $(date)" | tee -a $LOG
