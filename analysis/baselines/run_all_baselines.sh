#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN="conda run -n plant-mechinterp-rev python analysis/baselines/train_baselines.py"

# 3 seeds for speed; all four models; 15 epochs with early stopping.
for ds in region_type splice tss promoter; do
  out="data/real/results/baselines/${ds}.json"
  if [[ -f "$out" ]]; then
    echo "[skip] $out exists"
    continue
  fi
  echo "[run ] $ds"
  $RUN --dataset "$ds" \
       --models cnn1 cnn3 bilstm kmer_mlp \
       --seeds 0 1 2 \
       --epochs 15 \
       --batch 32 \
       > "data/real/logs/baselines_${ds}.log" 2>&1
done
echo DONE
