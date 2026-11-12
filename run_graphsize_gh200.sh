#!/bin/bash
# GH200, 12 Nov (Michael): GraphSAGE + METIS on smaller graphs.
# Graph sizes = the existing subsets of the full graph (unions of k=16 shards, built 3 Nov by build_subset.py):
#   s8 = 187 GB, s4 = 93 GB, s2 = 47 GB, s1 = 23 GB  (the full 375 GB graph is the existing k=16 run).
# For each size:
#   1. METIS k=16 on that graph (metis_partition.py --build, then --k 16), into $S/metis/part_k16.npy
#   2. shard_infer.py, variant fetch only: one shard resident on the GPU, GPU gathers the in-shard rows,
#      CPU samples / filters / fetches the out-of-shard rows; all 16 shards, 64 batches per shard = 1,024 batches,
#      6 fanouts, timing only (the subsets renumber nodes, so no accuracy pass; same as the capacity runs).
# Needs libmetis (64-bit) in $HOME/local and /unicorn/code/metis_partition.py.
#   tmux new -s gsz ; bash run_graphsize_gh200.sh ; (Ctrl-b d)
#   subset of sizes:  SIZES="1 2" bash run_graphsize_gh200.sh

cd /unicorn/code
eval "$(conda shell.bash hook)" && conda activate gnn || { echo "conda activate failed"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=4
export METIS_HOME=$HOME/local LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH OMP_NUM_THREADS=16

D=/unicorn/gnn
W=$D/results/h100het_378025/best.pt
OUT=results_graphsize_gh200
K=16
mkdir -p $OUT logs_gh200
LOG=logs_gh200/run_graphsize.log
{ echo "start $(date)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; free -g | head -2; df -h $D | tail -1;
  md5sum shard_infer.py metis_partition.py $W; ls $HOME/local/lib | grep -i metis; } | tee -a $LOG
[ -f $HOME/local/lib/libmetis.so ] || { echo "libmetis.so missing in $HOME/local/lib" | tee -a $LOG; exit 1; }
while pgrep -f "shard_infer.py|shard_pipeline.py|metis_partition.py" > /dev/null; do sleep 120; done

for N in ${SIZES:-1 2 4 8}; do
  S=$D/subsets/s$N
  [ -f $S/subset_info.json ] || { echo "missing $S" | tee -a $LOG; exit 1; }
  echo "== s$N: $(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["nodes"], "nodes,", round(d["feat_gb"], 1), "GB")' $S/subset_info.json)  $(date)" | tee -a $LOG

  # 1. METIS k=16 on this graph (skipped if part_k16.npy already exists)
  if [ ! -f $S/metis/part_k$K.npy ]; then
    python metis_partition.py --build --csr $S/csr --hetero $S/csr_hetero --out $S/metis 2>&1 | tee logs_gh200/metis_s${N}_build.log
    echo "  s$N metis build exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
    python metis_partition.py --k $K --csr $S/csr --hetero $S/csr_hetero --out $S/metis 2>&1 | tee logs_gh200/metis_s${N}_k$K.log
    echo "  s$N metis k=$K exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
    [ -f $S/metis/part_k$K.npy ] || { echo "  s$N: no part_k$K.npy, stopping" | tee -a $LOG; exit 1; }
    rm -f $S/metis/graph_xadj.npy $S/metis/graph_adjncy.npy     # the saved graph is only needed for partitioning
    head -30 $S/metis/part_k$K.json | tee -a $LOG
  fi

  # 2. GraphSAGE + METIS, shard resident, all 16 shards, 1,024 batches, 6 fanouts
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
  numactl --membind=0 python shard_infer.py --weights $W --root $D/data --csr $S/csr --hetero-dir $S/csr_hetero \
    --feat $S/full_feat.npy --metis $S/metis --k $K --in-memory \
    --fanouts 40/20,5/5,10/5,15/10,25/15,60/30 --variants fetch,coherent --batches-per-shard $((1024 / K)) --warmup 1 --no-val \
    --out $OUT --summary $OUT/summary_graphsize_gh200.csv --tag gh200_gsz_s${N} 2>&1 | tee logs_gh200/graphsize_s${N}.log
  echo "  s$N fetch exit ${PIPESTATUS[0]} $(date)" | tee -a $LOG
done
echo "all done $(date)" | tee -a $LOG
