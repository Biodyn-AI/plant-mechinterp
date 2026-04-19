# Cover letter — Plant Methods submission

**Manuscript title:** Mechanistic Interpretability of a Plant Genomic Foundation Model: Data Probes, Causal Interventions, and Motif-Grounded Features

**Corresponding author:** Ihor Kendiukhov, Department of Computer Science,
University of Tübingen, Tübingen, Germany.
E-mail: <ihor.kendiukhov@student.uni-tuebingen.de>.
ORCID: [0000-0001-5342-1499](https://orcid.org/0000-0001-5342-1499).

**Co-authors:** none (sole author).

---

Dear Editor,

We are pleased to submit our manuscript, *"Mechanistic Interpretability of a
Plant Genomic Foundation Model: Data Probes, Causal Interventions, and
Motif-Grounded Features,"* for consideration as a **Methodology** article in
*Plant Methods*.

Foundation models trained on plant DNA (AgroNT, Plant-DnaGemma, PlantCAD2,
PlantRNA-FM) are being rapidly adopted across plant-biology pipelines, yet
the community currently has no standard protocol for asking *what* these
models learn or *whether* their learned features correspond to known
biology. Existing claims in the field rely on attention visualizations,
small synthetic evaluation sets, or single-number benchmark gains, and are
vulnerable to GC-content and composition confounds. This gap is the
methodological problem our manuscript addresses.

## What we contribute

We present a reproducible evaluation and interpretation protocol for plant
genomic foundation models, instantiated on Plant-DnaGemma (152M parameters)
and built around six annotated benchmarks drawn from TAIR10/Araport11,
EPDnew, and Ensembl Plants release 58 (110 000+ windows across 6
angiosperm species). The protocol comprises:

1. **Per-layer linear probing** with 5-fold cross-validation and held-out-
   chromosome evaluation on four Arabidopsis tasks (region-type,
   splice-site, TSS, promoter).
2. **Learned-from-scratch baselines** — CNN, BiLSTM, *k*-mer MLP — so that
   foundation-model gains can be meaningfully quantified against methods of
   comparable task supervision.
3. **GC-matched composition controls** on both AraReg tasks and the
   6-species analysis, preventing the widespread confound that plant species
   differ systematically in GC content.
4. **Phylogenetic evaluation** with partial-Mantel residualization against
   GC and 1-nearest-neighbor clade analysis on held-out tomato and soybean.
5. **Sparse-autoencoder feature decomposition** (Top-K 16×, $L_0{=}32$) on
   layer-7 activations, with motif enrichment against JASPAR 2024 plantae
   and cross-referencing against PlantTFDB 5.0 Arabidopsis TF families.
6. **Causal interventions** — activation patching with canonical-base and
   dinucleotide-preserving corruptions, negative-control patching at
   non-motif positions, and per-component (attention head / MLP) ablations.

## Principal findings

- Plant-DnaGemma outperforms *k*-mer, CNN, BiLSTM, and *k*-mer-MLP baselines
  on multi-class tasks (+6.9 pp on 5-way region-type, +3.4 pp on 3-way
  splice). On binary tasks with strong compositional signal, a 3-layer CNN
  effectively matches the foundation model (within 0.4 pp) — an honest
  negative-style finding that clarifies where pretraining adds value.
- The model captures non-compositional cross-species information (68 %
  accuracy on a GC-matched 4-species probe, chance 25 %; partial Mantel
  r = 0.55, p = 0.019 between representational and phylogenetic distance
  after residualizing GC).
- **72 of 300 analyzed SAE features (24 %) are significantly enriched** for
  a JASPAR 2024 plantae transcription-factor motif (BH q < 0.05), with
  **47 of the 72 mapping to named PlantTFDB Arabidopsis TF families**
  (BBR-BPC, NAC, MYB, AT-hook, WRKY, Dof, GRF, ERF, and others). A
  parallel region-class enrichment test finds 270 of 300 features (90 %)
  significantly enriched for a specific genomic region class.
- Per-component ablations on splice-site processing show that the causal
  load is carried by MLPs (L0/L4/L5/L6 MLP ablations cause 15–28 pp
  accuracy drops), not by individual attention heads (max 5 pp, 10 of 144
  heads > 3 pp) — a genuinely mechanistic rather than decodability-based
  finding.

## Why Plant Methods

Our manuscript is fundamentally a **methodology paper**: it defines how to
rigorously evaluate and interpret a plant genomic foundation model. The
protocol integrates standard plant bioinformatics resources
(TAIR10/Araport11, EPDnew, PlantTFDB, JASPAR plantae) with modern
interpretability tooling (sparse autoencoders, activation patching, partial
Mantel, GC-matched subsetting) in a way that is directly reusable by other
groups on their own plant genomic models. We believe this fits the
*Plant Methods* remit for "new and improved methods in plant sciences,
plant biotechnology, and plant bioinformatics" better than any other
specialist venue.

## Reproducibility

All code, dataset-building scripts, frozen data splits, and
figure-generation scripts are available at
**<https://github.com/Biodyn-AI/plant-mechinterp>**. A tagged snapshot
of the repository together with trained sparse-autoencoder weights,
pre-computed motif-score matrices, and built parquet datasets is
permanently archived at Zenodo with DOI
**<https://doi.org/10.5281/zenodo.18665058>**. Data sources are pinned
to specific versions (TAIR10, Araport11, Ensembl Plants release 58,
EPDnew v1, JASPAR 2024 CORE plantae, PlantTFDB 5.0). Stochastic
pipelines use fixed seeds (0, 1, 2). Figures are regenerated
deterministically from the committed result JSONs.

## Declarations

- **This work has not been published elsewhere and is not under
  consideration by any other journal.**
- **Competing interests:** the author declares no competing interests.
- **Ethics:** this study uses only publicly available genomic data; no
  animal or human subjects were involved.
- **Funding:** no specific grant from any public, commercial, or
  not-for-profit funding agency.
- **Author contributions:** Ihor Kendiukhov (sole author) conceived the
  study, designed and ran all experiments, analyzed the results, and
  wrote the manuscript.

## Suggested reviewers

The author does not suggest specific reviewers and defers to the editorial
office to identify suitable referees among active researchers in plant
genomic foundation models, genomic interpretability, and plant
transcription-factor biology (PlantTFDB / JASPAR).

## Non-preferred reviewers

None.

Thank you for considering this manuscript. I look forward to the
reviewers' feedback.

Sincerely,

Ihor Kendiukhov
Department of Computer Science, University of Tübingen.
