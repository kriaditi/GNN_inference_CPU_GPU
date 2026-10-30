#!/bin/bash
# GH200, 30 Oct: "naive" streaming per Michael: the table divided into a few large
# contiguous chunks (not METIS), each loaded to the GPU in turn. Same stream_infer.py,
# larger chunks: 47 GB = 8 chunks, 62.6 GB = 6 chunks, 75.01 GB = 5 chunks.
# Free GPU memory with the table in RAM is ~78.7 GB, so the 75 GB buffer is expected
# to fail; it runs last so the 8- and 6-chunk results are already saved.
# G = 1 only (one full pull of the table per batch); 1,000 timed batches; one process per fanout.
# Run after run_sustained_gh200.sh has finished:
#   tmux new -s naive ; bash run_naive_stream_gh200.sh ; (Ctrl-b d)

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1

D=/unicorn/gnn
PATHS="--weights $D/results/h100het_378025/best.pt --root $D/data --csr $D/csr --hetero-dir $D/csr_hetero --feat $D/data/full_feat.npy"
OUT=results_stream_gh200_naive
mkdir -p $OUT logs_gh200
LOG=logs_gh200/run_naive.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; } | tee $LOG

for F in 40/20 5/5 10/5 15/10 25/15 60/30; do
  python stream_infer.py $PATHS --in-memory --fanouts $F --batches-per-pass 1 --chunk-gb 47,62.6,75.01 --val \
    --batches 1000 --out $OUT --summary $OUT/summary_stream_gh200_naive.csv --tag gh200_naive \
    2>&1 | tee logs_gh200/naive_f${F/\//x}.log
  echo "naive $F done $(date) exit ${PIPESTATUS[0]}" | tee -a $LOG
done
echo "all done $(date)" | tee -a $LOG
