#!/usr/bin/env bash
# Sequential activation extraction for all datasets, both trained and random
# variants. Runs inside the plant-mechinterp-rev conda env.
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN="conda run -n plant-mechinterp-rev python analysis/probing/extract_activations.py"

# region_type (14 999) already has trained; add random.
$RUN --dataset region_type --tag random --batch 16

# splice (9 000)
$RUN --dataset splice --tag trained --batch 16
$RUN --dataset splice --tag random --batch 16

# tss (balanced subsample 20k)
$RUN --dataset tss --tag trained --batch 16 --max-seq 20000
$RUN --dataset tss --tag random --batch 16 --max-seq 20000

# promoter (balanced subsample 20k)
$RUN --dataset promoter --tag trained --batch 16 --max-seq 20000
$RUN --dataset promoter --tag random --batch 16 --max-seq 20000

# multispecies natural-GC (12 000)
$RUN --dataset multispecies --tag trained --batch 16 --label-col species
$RUN --dataset multispecies --tag random --batch 16 --label-col species

# multispecies GC-matched (~3 140)
$RUN --dataset multispecies_gc_matched --tag trained --batch 16 --label-col species
$RUN --dataset multispecies_gc_matched --tag random --batch 16 --label-col species

# multispecies held-out (4 000)
$RUN --dataset multispecies_heldout --tag trained --batch 16 --label-col species
$RUN --dataset multispecies_heldout --tag random --batch 16 --label-col species

echo DONE
