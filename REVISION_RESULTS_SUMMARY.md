# Revision Results Summary

This document records the outcome of executing the REVIEWER_RESPONSE_PLAN.md
autonomously on real plant genomic data. All numbers are measured, not
paraphrased. Paper draft: `paper/main_v2.tex` / `paper/main_v2.pdf`.

## 1. Real data benchmarks built (§1 of the plan)

All from TAIR10 + Araport11 r58 + EPDnew + Ensembl Plants r58:

| Dataset | Classes | N | Window | Chrom split |
|---|---|---|---|---|
| AraReg-RegionType | 5 (exon/intron/utr5/utr3/intergenic) | 14 999 | 1 024 nt | chr1-3 train, chr4 val, chr5 test |
| AraReg-Splice | 3 (donor/acceptor/dinuc-shuf) | 9 000 | 1 024 nt | same |
| AraReg-TSS | 2 (EPDnew-TSS/shuffled) | 45 402 → 20 000 | 1 024 nt | same |
| AraReg-Promoter | 2 (promoter/intergenic) | 40 701 → 20 000 | 1 024 nt | same |
| Multi-Species (natural GC) | 6 species | 12 000 | 1 024 nt | per-species chrom split |
| Multi-Species-GCmatched | 4 species | 3 140 | 1 024 nt | GC ≈ 0.40 ± 0.01 |
| Multi-Species-Heldout | tomato + soybean | 4 000 | 1 024 nt | clade-level 1-NN |

## 2. Real-data probing (reviewer Major 1)

5-fold CV accuracy with standard deviation, held-out-chromosome test:

| Task | PlantDG best layer | CV acc | Test acc | Random init best | Best k-mer | GC only |
|---|---|---|---|---|---|---|
| RegionType | L11 | 0.642 ± 0.008 | **0.637** | 0.446 (L1) | 0.541 (4-mer) | 0.264 |
| Splice     | L11 | 0.665 ± 0.013 | **0.668** | 0.508 (L0) | 0.625 (4-mer) | 0.333 |
| TSS        | L8  | 0.992 ± 0.002 | 0.993 | 0.842 (L0) | 0.504 (4-mer) | 0.493 |
| Promoter   | L11 | 0.927 ± 0.007 | 0.933 | 0.800 (L0) | 0.521 (4-mer) | 0.521 |
| Multispecies (6-way) | L11 | 0.690 ± 0.007 | — | 0.409 (L0) | 0.569 (4-mer) | 0.268 |
| **Multispecies GC-matched (4-way)** | L11 | **0.680 ± 0.026** | — | 0.380 (L2) | 0.563 | 0.257 |
| Multispecies Heldout (2-way) | L11 | 0.791 ± 0.029 | — | 0.611 (L0) | 0.715 | 0.517 |

**Label-shuffle control**: ≈ chance on every task (21% 5-way, 34% 3-way, 50%
binary). Pipeline has no label leak.

## 3. Learned-from-scratch baselines (reviewer Major 3)

Held-out-chromosome test set accuracy, mean ± 1 SD over 3 seeds:

| Task | PlantDG | CNN1 | CNN3 | BiLSTM | k-mer MLP | PlantDG gap |
|---|---|---|---|---|---|---|
| RegionType | **0.637** | 0.482 ± .005 | 0.556 ± .049 | 0.305 ± .053 | 0.568 ± .004 | **+6.9 pp** |
| Splice     | **0.668** | 0.575 ± .009 | 0.625 ± .011 | 0.436 ± .030 | 0.634 ± .017 | **+3.4 pp** |
| TSS        | **0.993** | 0.957 ± .001 | 0.989 ± .000 | 0.985 ± .001 | 0.985 ± .001 | +0.4 pp |
| Promoter   | **0.933** | 0.874 ± .003 | 0.929 ± .002 | 0.748 ± .090 | 0.917 ± .000 | +0.4 pp |

Honest finding: on binary tasks with strong local composition (TSS,
promoter), a 3-layer CNN trained from scratch effectively matches
Plant-DnaGemma. The model's real advantage is on multi-class tasks.

## 4. Cross-species v2 (reviewer Major 4)

- **GC-matched 4-way species classification**: trained 0.680, random init 0.380,
  4-mer 0.563, GC-only 0.257 (chance = 0.25). Plant-DnaGemma gives +11.7 pp
  over 4-mer and +42.3 pp over GC alone even when GC is forced constant.
- **Partial Mantel** (representational distance ~ phylogenetic distance,
  controlling for GC distance): r = **0.59, p = 0.019** (5 000 permutations).
  Simple rep~phy correlation: 0.21; GC~phy correlation: 0.84 (GC confound).
  After removing the GC confound, the phylogenetic signal emerges significant.
- **Held-out clade 1-NN** (tomato and soybean, both eudicots, classified by
  nearest neighbor among the 4 in-distribution species: Arabidopsis [eudicot],
  rice, maize, Brachypodium [monocots]):
  - Tomato → Arabidopsis: **58.3%** (25.7% maize, 10.3% rice, 5.7% Brachypodium)
  - Soybean → Arabidopsis: **52.6%** (19.8% rice, 17.3% maize, 10.3% Brachypodium)
  Both eudicots cluster with the only in-distribution eudicot despite lower
  GC than the monocots — a clade-level phylogenetic signal not explainable
  by composition.

## 5. SAE with biological grounding (reviewer Major 5)

12 SAEs trained (L1 & TopK × 4× and 16× expansion × 3 seeds) on real layer-7
activations:

| Variant | Var. explained | L0 | Dead | Sparsity |
|---|---|---|---|---|
| L1 ×4    | 0.985 | 1568 | 0.0% | 0.490 |
| L1 ×16   | 0.977 | 3291 | 0.0% | 0.732 |
| TopK ×4  | 0.863 | 32   | 0.5% | 0.990 |
| **TopK ×16** | **0.918** | **32** | 8.5% | **0.997** |

**Motif enrichment** on TopK-16× seed 0 (300 top features, 805 JASPAR 2024
plantae motifs, Mann–Whitney U with BH FDR):

- **72/300 features (24%)** have a significantly enriched best-matching TF
  motif at q < 0.05.
- Top TF families matched: **BPC** (BPC1/5/6, GAGA-repeat binders),
  **DOF** (DOF3.6, DOF5.8), **MYB**, **NAC**, **ERF** (ERF115), **LEC2**,
  **JUB1**, **AHL20**.
- L1-16× seed 0 gives 234/300 (78%) significant features — less sparse, so
  more permissive threshold. TopK result is the headline number.

This directly addresses the reviewer concern that SAE analyses were
"purely statistical". The features ground to well-documented plant TFs.

## 6. Mechanistic causal evidence (reviewer Major 2)

### Activation patching v3 on splice sites (96 sequences)

| Corruption mode | Direction | Peak layer | Peak effect (normalized) |
|---|---|---|---|
| Canonical GT→CC | noising  | L2 | +59.8 |
| Canonical GT→CC | denoising | L0 | +12.2 |
| Dinuc-shuffled (±20 nt window) | noising | L2 | −23.0 |
| Dinuc-shuffled | denoising | L4 | −4.2 |

Splice-site signal is localized to early-middle layers (0–5) and the
canonical GT/AG motif bases are causally critical: corrupting them from
layers 0–3 drops the classifier logit by ~60 normalized units.

### Per-component ablations on splice (splice classifier baseline 64.2%)

- **Attention heads are largely redundant**: mean drop across 144 heads 0.6%,
  max 5.0% (L6H11). Only 10/144 heads exceed a 3% drop.
- **MLPs are causally critical**: L6 MLP ablation drops 28.3 pp, L5 MLP 27.5,
  L0 MLP 25.0, L4 MLP 15.0. MLPs at layers 7–11 have no effect because the
  probe reads out at L7.

Interpretation: splice processing is MLP-dominated with redundant attention.
This is a properly mechanistic result (not just decodability) — it shows
*which components matter* for the classifier's decision.

## 7. Paper update

- `paper/main_v2.tex` (12 pages) — full rewrite with real-data results.
- `paper/figures_real/` — 9 regenerated figures (probing per task, baselines,
  cross-species, SAE, patching, head ablations).
- References added for Gao et al. 2024 (TopK SAE), Legendre & Legendre
  (partial Mantel), Rauluseviciute et al. 2024 (JASPAR), Meister et al. 2004
  (BPC), Yanagisawa 2004 (DOF).

## 8. What a reviewer would now see

- Every result on real data with proper statistical reporting.
- Learned CNN/LSTM/k-mer-MLP baselines on every task.
- GC-matched cross-species controls with partial Mantel and held-out clade.
- SAE features mapped to named plant TFs with q-values.
- Causal localization of splice processing to early-middle MLPs.
- Honest negative-style findings: CNN3 matches Plant-DnaGemma on binary
  tasks; attention heads are redundant; BiLSTM under-trains on 1024 nt.

All reviewer concerns from Majors 1–5 and Minor reporting issues are directly
addressed with new experiments, not reframing.
