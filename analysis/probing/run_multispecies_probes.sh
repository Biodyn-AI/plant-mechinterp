#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN="conda run -n plant-mechinterp-rev python -W ignore analysis/probing/run_probing.py"

# Runs in sequence after the first multispecies probe finishes.
# (Relies on the earlier --dataset multispecies run having completed.)

for ds in multispecies_gc_matched multispecies_heldout; do
  out="data/real/results/probing/${ds}.json"
  if [[ -f "$out" ]]; then
    echo "[skip] $out exists"
    continue
  fi
  echo "[run ] $ds"
  $RUN --dataset "$ds" > "data/real/logs/probe_${ds}.log" 2>&1
done
echo DONE
