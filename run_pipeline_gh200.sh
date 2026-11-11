#!/bin/bash
# GH200, 10 Nov: every batch needs a different shard (shards rotate batch by batch, k=16, 23 GB shards).
# Three ways to bring the shard in, same batches:
#   series    copy the shard RAM -> GPU, then the batch                      (one GPU buffer)
#   overlap   copy on a side stream while the CPU samples, filters, fetches  (one GPU buffer)
#   pipeline  two GPU buffers: while batch i runs, the shard of batch i+1 is copied into the other buffer
# 63 rounds x 16 shards = 1,008 timed batches per (mode, fanout), 6 fanouts, accuracy pass per fanout.
# Uses the METIS-ordered k=16 table already on disk.
#   1. smoke  1 timed round (16 batches), 40/20, no accuracy pass   ~5 min
#   2. full                                                          ~1 h
#   tmux new -s pipe ; bash run_pipeline_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=/unicorn/gnn
MO=$D/metis_ordered
K=${K:-16}
ARGS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero --metis $D/metis --k $K \
  --feat $MO/full_feat_metis_k$K.npy --row-map $MO/rowmap_k$K.npy --chunk-bounds $MO/bounds_k$K.npy --in-memory"
OUT=results_pipeline_gh200
mkdir -p $OUT logs_gh200
LOG=logs_gh200/run_pipeline.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; md5sum shard_pipeline.py; } | tee -a $LOG
[ -f $MO/info_k$K.json ] || { echo "ordered table k=$K missing" | tee -a $LOG; exit 1; }
while pgrep -f "shard_infer.py|shard_pipeline.py" > /dev/null; do sleep 120; done
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

# 1. smoke
numactl --membind=0 python shard_pipeline.py $ARGS --modes series,overlap,pipeline --fanouts 40/20 --rounds 1 --no-val \
  --out $OUT/smoke --summary $OUT/smoke/summary_smoke.csv --tag gh200_pipe_smoke 2>&1 | tee logs_gh200/pipeline_smoke.log
echo "smoke done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
[ "$(grep -c '^\[summary\]' logs_gh200/pipeline_smoke.log)" -ge 3 ] || { echo "smoke failed, stopping" | tee -a $LOG; exit 1; }
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

# 2. full run
numactl --membind=0 python shard_pipeline.py $ARGS --modes series,overlap,pipeline --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 --rounds 63 \
  --out $OUT --summary $OUT/summary_pipeline_gh200.csv --tag gh200_pipe 2>&1 | tee logs_gh200/pipeline_k$K.log
echo "pipeline k=$K done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
echo "all done $(date)" | tee -a $LOG
