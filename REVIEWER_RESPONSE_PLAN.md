# Reviewer Response Plan — "Mechanistic Interpretability of a Plant Genomic Foundation Model"

This document turns each reviewer concern into a concrete work plan. The guiding rule: **whenever a comment can be addressed either by reframing or by running new/additional experiments, we run the experiments (and reframe only where needed as a follow-up).** Every section below names the deliverable, the concrete data/method to use, the metrics, and where the paper has to change.

All paths below are relative to the repository root `plant-mechinterp/`.

---

## 0. Summary of Reviewer Concerns → Response Strategy

| # | Concern | Strategy | Primary New Experiments |
|---|---------|----------|-------------------------|
| 1 | Synthetic data → weak biological validity | **Replace synthetic eval with real plant genomic data** for every core result | §1: Real Genomic Benchmark (TAIR10 + Ensembl Plants) |
| 2 | "Mechanistic" claims overstated (descriptive only) | **Add causal mechanistic evidence**: path patching, head ablations, attention-attribution, stronger patching perturbations, circuit discovery | §2: Mechanistic Evidence Upgrade |
| 3 | Baselines insufficient / partly misleading | **Add learned baselines** (CNN, LSTM, shallow MLP on one-hot and k-mers) and **redesign tasks** so GC/k-mer do not trivially solve them | §3: Baselines & Task Redesign |
| 4 | Cross-species analysis underpowered | **Scale up** (thousands of sequences, ≥1 kb length), **phylogenetic control**, **GC-matched real sequences**, **held-out species** | §4: Cross-Species v2 |
| 5 | SAE lacks biological interpretation | **Map SAE features to known biology**: motif enrichment against JASPAR/PlantTFDB/PlantCARE, annotation overlap, causal feature ablation, replicate across seeds | §5: SAE Biological Grounding |
| 6 | Statistical reporting & figure quantification | Fix reporting conventions throughout; add quantification panels/tables; label every error bar clearly | §6: Reporting Polish |

Sections 1–5 are *new experimental work*. Section 6 is editorial but depends on rerun data.

---

## 1. Real Genomic Benchmark (addresses Major Comment 1 and partially 3, 4, 5)

### 1.1 New datasets (all real, all annotated)

Build a versioned real-data benchmark at `data/real/`. Download scripts live under `scripts/data/`.

| Dataset | Source | What we pull | Intended use |
|---------|--------|--------------|--------------|
| **AraReg-Promoter** | TAIR10 + Araport11 GFF3 | 5′ flanking regions (−1000, +200) of all protein-coding genes; curate a clean positive set using EPDnew *A. thaliana* promoters | Promoter vs. non-promoter classification |
| **AraReg-Exon/Intron/Intergenic** | TAIR10 + Araport11 GFF3 | Internal exons (CDS, ≥100 nt), introns (≥100 nt, canonical GT–AG), random intergenic regions ≥5 kb from any gene | 4-way region-type classification (replaces synthetic §3.2) |
| **AraUTR-5/3** | Araport11 | Annotated 5′UTR and 3′UTR regions | Auxiliary region task |
| **TSS-Pos** | EPDnew *A. thaliana* (v1, 22 128 TSSs) | ±500 nt around TSS | TSS vs. shuffled-control binary task; finer than region-type |
| **Splice-Sites** | Araport11 GFF3 | 5′ and 3′ splice sites; dinucleotide-shuffled negatives | Splice donor/acceptor/non-site task |
| **TFBS-Motif** | PlantTFDB 5.0 + JASPAR CORE *plantae* 2024 | Windows centered on called TF binding sites for ≥20 top TFs; GC-matched negatives | Used in §2 for activation patching and §5 for SAE mapping |
| **Multi-Species-Real** | Ensembl Plants release 58 (*A. thaliana*, *O. sativa*, *Z. mays*, *S. lycopersicum*, *G. max*, *B. distachyon*) | 10 000 random 1 kb windows per species from non-overlapping chromosomes | Cross-species analysis (§4) |
| **Ortho-Matched** | OrthoFinder run on above species' proteomes | Orthologous gene promoters and CDS triples/quadruples | Phylogenetic control subset (§4) |

Each dataset is exported as (a) FASTA, (b) a parquet table with `sequence_id, label, gc, length, chrom, start, end, strand, source, split`, and (c) a **fixed stratified train/val/test split** (70/15/15) frozen under `data/real/splits/`. All downloads are driven by `scripts/data/fetch_real_data.py` and verified with checksums.

**Sample sizes.** ≥ 5 000 sequences per class for region-type tasks, ≥ 10 000 per species for cross-species, ≥ 2 000 windows per TF. All orders of magnitude above the current synthetic 100 per class.

**Length.** Default window 1 024 nt (within Plant-DnaGemma's 2 048 context); additional 200 nt control window to match current paper.

### 1.2 Re-run every existing analysis on real data

Every table and figure in the current manuscript must have a real-data twin. Concretely:

- **Probing (§Results 4.1, 4.2).** Re-run 5-fold CV probing on `AraReg-Exon/Intron/Intergenic` (4-way), `TSS-Pos` (binary), `Splice-Sites` (3-way). Report accuracy **and** macro F1 **and** per-class F1.
- **Layer-wise table.** Regenerate Table 1 on real data; keep synthetic table in the Appendix for transparency.
- **Activation patching (§Results 4.4).** Run on `TFBS-Motif` windows where the patched position is a real TF binding site, not a placed motif. See §2.
- **SAE (§Results 4.5).** Train on real layer-7 activations (§5).
- **Cross-species (§Results 4.6).** Replaced entirely by §4.

The current synthetic experiments stay in the paper **only in a supplement**, clearly labeled as sanity checks with controlled ground truth, not as primary results.

### 1.3 Pitfalls to guard against

- **Label leakage via length or GC.** All splits are stratified by class *and* length bin *and* GC bin. Report class-wise GC distributions.
- **Chromosome leakage.** Train/test split by chromosome for each species to prevent genomic-proximity leakage.
- **Overlap with pretraining.** Plant-DnaGemma was trained on plant genomes including Arabidopsis. We cannot remove that overlap, but we (a) report this as a limitation, (b) add a held-out species (*S. lycopersicum* or *G. max*) that stresses generalization, (c) include a *random-shuffle-within-class* negative control to show learned signal isn't a pretraining memorization artifact.

### 1.4 Paper changes

- Replace Methods §3.2 "Datasets" — synthetic paragraph becomes a short supplementary note; a new "Real genomic benchmark" subsection describes §1.1.
- Replace every main-text result figure with its real-data counterpart; move synthetic versions to the supplement.
- Update Abstract, Introduction, and Limitations to reflect real-data findings.

---

## 2. Mechanistic Evidence Upgrade (addresses Major Comment 2)

The reviewer is right that linear probing is decodability, attention visualization is descriptive, and the activation-patching null result was not followed up. We add genuinely mechanistic (i.e., causal) evidence.

### 2.1 Stronger and more informative activation patching

New script `analysis/mechanistic/activation_patching_v3.py`. For each task in `TFBS-Motif` and `Splice-Sites`:

1. **Clean/corrupt pairs** constructed by (a) replacing the motif with a di-nucleotide–shuffled surrogate (preserves GC and base composition, destroys the motif), and (b) mutating only the canonical bases (e.g., GT → CA at donor). This answers the reviewer's worry that prior corruption was "insufficient."
2. **Denoising direction** (restore clean activations into corrupted run) *and* **noising direction** (inject corrupted activations into clean run) — report both.
3. **Granularity sweep**: per residual-stream position × per layer × per component (attn-out vs MLP-out). Produces the standard 2D heatmap à la Meng et al. (2022).
4. **Metric**: logit difference on a downstream classifier head probed at the final layer (trained once, frozen). Report effect size with bootstrap CIs.
5. **Negative controls**: patch random non-motif positions; expect null effect.

Deliverable: Figure "Causal localization of TFBS/splice-site information" with the denoising heatmap + noising heatmap side-by-side and a bar chart of the per-layer effect.

### 2.2 Path patching and head ablation

New script `analysis/mechanistic/path_patching.py` using the `TransformerLens`-style hook framework already present in `analysis/`.

- **Head ablation.** Zero/mean-ablate each of the 144 heads on `TSS-Pos` and `Splice-Sites`; rank heads by accuracy drop. Identify the top-*k* heads whose removal collapses the task.
- **Path patching** from motif-position residual stream into each head's value and query inputs; isolate which heads read motif information.
- **MLP ablation** per layer on the same tasks.

Expected deliverable: a ranked list of "critical heads" and an accuracy-vs-top-*k*-ablated curve. A successful result gives a circuit diagram figure; a null result across all heads gets reported honestly (and supports distributed encoding).

### 2.3 Attention attribution, not just visualization

Replace the current qualitative attention figure with:

- **Attention-roll-out / attention flow** (Abnar & Zuidema 2020) attributed to motif positions.
- **Attention-norm attribution** (Kobayashi et al. 2020) — ||αᵢⱼ · V(xⱼ)|| — which fixes the well-known weakness that raw attention weights overstate importance.
- **Quantitative head characterization.** For each head compute: attention entropy, previous-token score, motif-to-CLS attention, motif-to-motif attention. Cluster heads by this vector (k-means, k chosen by gap statistic). Report cluster sizes and composition by layer.

Output replaces Figure 2. Report one summary table (Appendix) with per-head scores on all metrics.

### 2.4 Logit-lens / tuned-lens on genomic tokens

Add `analysis/mechanistic/logit_lens.py`. Project each layer's residual stream through the unembedding to track how nucleotide/k-mer predictions evolve. For a TATA-containing prompt and a control, plot per-layer top-*k* tokens and token-probability trajectories. This gives readers a second mechanistic lens that is not just decodability.

### 2.5 Paper changes

- Rename Results §4.4 "Distributed motif encoding" → "Causal localization of regulatory information" and present the denoise+noise heatmaps.
- Add a new subsection "Critical attention heads and circuit sketch" with the ablation ranking and path-patching result.
- Replace attention figure with the quantitative attribution + head-cluster figure.
- In the Introduction and Discussion, drop the word "mechanistic" in places where we only have probing evidence; use it only where we have causal interventions.

---

## 3. Baselines and Task Redesign (addresses Major Comment 3 and part of 1)

### 3.1 Add learned baselines

In addition to k-mer logistic regression and the random-weight model, train from scratch on the *same* real tasks:

1. **1-layer CNN** on one-hot DNA — 64 filters × width 8 + global max-pool + linear. ~50 k parameters.
2. **3-layer CNN** — the canonical DanQ/Basset-style stack (CNN → max-pool → CNN → pool → MLP).
3. **Bi-LSTM** on one-hot (1 layer, 128 hidden).
4. **MLP on k-mer counts** (k=3,4,5,6) with tuned regularization — a stronger k-mer baseline than current logistic regression.
5. **DNABERT-2 frozen + linear probe** — external foundation-model baseline to show Plant-DnaGemma isn't uniquely good just because it's a transformer.

All baselines trained with identical splits, early stopping, and 5 seeds. Report mean ± 95 % bootstrap CI.

### 3.2 Redesign tasks so composition ≠ label

The reviewer's critique that "GC baseline dominates species classification" means the *tasks themselves* are bad, not just the baselines. We fix this:

- **GC-matched contrasts.** For every classification task, construct a GC-matched variant where negatives/positives have identical GC distributions (matched via propensity stratification). Plot model and baseline performance on matched vs. unmatched sets.
- **Shuffle-composition controls.** For each positive sequence, a dinucleotide-shuffled version becomes a hard negative. A k-mer baseline collapses to chance on dinuc-shuffled pairs, so any remaining signal is non-compositional.
- **Harder region-type task.** 4-way among exon / intron / 5′UTR / 3′UTR with length- and GC-matched sampling.
- **Reverse-complement invariance test.** Evaluate on RC sequences; a model that learned composition only should be trivially RC-invariant on top of k-mer; a model that learned directional motifs (e.g., TATA) should not.

### 3.3 Better reporting of the +7.5 % gain

The reviewer dislikes the "+7.5 % over k-mer" framing because it's on easy synthetic data. On the redesigned real tasks:

- Report the Plant-DnaGemma gain on the **GC-matched** and **dinuc-shuffled** contrasts — the only contrasts where a gain means something non-trivial.
- Report a **scaling-with-training-data** curve (10 %, 50 %, 100 % of train set); pretraining should help most in low-data regimes.
- Report calibration (ECE) and ROC-AUC, not just accuracy.

### 3.4 Paper changes

- Methods §3.4 expands with subsection "Learned baselines" listing CNN/LSTM/MLP/DNABERT-2.
- Results §4.1 is rewritten around the matched and shuffled contrasts; replace the single "+7.5 %" number with a small table comparing all methods on all contrasts.
- Honest framing: where Plant-DnaGemma matches but does not clearly beat a 3-layer CNN, say so.

---

## 4. Cross-Species v2 (addresses Major Comment 4)

### 4.1 Scale-up

From 100 × 3 species × 128 nt → **10 000 × 6 species × 1 024 nt**, using `Multi-Species-Real` from §1.

### 4.2 Phylogenetic control

- Build a species tree from Ensembl Plants release notes and single-copy orthologs via OrthoFinder.
- For each species-pair, compute the phylogenetic distance and the representational distance (CKA and RSA on mean-pooled layer-*l* embeddings).
- Test whether representational distance correlates with phylogenetic distance *after* partialling out mean-GC distance (partial Mantel test). This is the right way to ask "does the model encode phylogeny beyond composition."

### 4.3 GC-matched real sequences

From the 10 000 × 6 pool, select a GC-matched subset (kernel density matching on mean GC with bandwidth 0.01) yielding ≥1 000 sequences per species with overlapping GC distributions. Rerun species classification on this subset; this is the critical test the reviewer is asking for.

### 4.4 Held-out species generalization

Train a probe on 4 species, test on 2 held-out species embedded in the same space (nearest-neighbor classification in representation space). A model that learned *phylogeny* should place held-out species near their sister taxa; a model that learned *just GC* should not.

### 4.5 Orthologous-gene analysis

For N orthologous gene families present in all 6 species, extract promoters, run them through the model, and test:

- whether orthologs cluster together across species in representation space (silhouette score);
- whether the same SAE features activate on orthologs (§5.3).

### 4.6 Paper changes

- Results §4.6 is rewritten around §4.1–4.5, with the GC-matched result as the headline.
- The Discussion's "GC confound" paragraph is rewritten to reflect what the scaled-up experiment actually shows (likely: model does encode *some* non-compositional species signal, but less than one would hope — or more, depending on results).

---

## 5. SAE Biological Grounding (addresses Major Comment 5)

### 5.1 Train SAE on real data, replicate, scale

- Retrain the SAE on layer-7 activations from `AraReg-Exon/Intron/Intergenic` and `Multi-Species-Real` sequences — not synthetic.
- **Scale sweep**: 4× (current), 16×, 64× expansion. Expect finer-grained features at larger widths.
- **TopK-SAE variant** (Gao et al. 2024) in addition to L1 — better feature quality, no shrinkage bias.
- **Replicate across 3 seeds**; report feature-matching rate between seeds (Hungarian matching on decoder-direction cosine).
- Evaluation metrics: reconstruction MSE, fraction of variance explained, dead-feature rate, L0, feature-reuse heavy-tail statistic.

### 5.2 Map features to known biological motifs

For each learned feature *f*:

1. Collect the top-*k* sequences (k = 100) that maximally activate *f*.
2. Extract the top-*k* token/window positions within those sequences.
3. Run motif discovery with **STREME** (MEME suite) on the activating windows vs. shuffled controls.
4. Compare the discovered motif to **JASPAR CORE plantae 2024** and **PlantTFDB 5.0** via TomTom; record the best match, E-value, and TF name.
5. Cross-check against **PlantCARE** for cis-regulatory element names.
6. For each feature with a significant motif match (E < 0.05 after multiple-test correction), record: TF family, biological function, GO terms.

Output: a table of "feature → motif → TF → biology" with ≥1 row per statistically matched feature. This is the concrete biological grounding the reviewer asked for.

### 5.3 Map features to genomic annotations

Beyond motifs, test whether features fire preferentially on specific annotated regions:

- **Enrichment of feature activation in** exon / intron / UTR / promoter / transposable-element / repeat regions using Araport11 and RepeatMasker tracks. Fisher's exact test with BH correction.
- **Chromatin / expression correlates** where available: AtGenExpress expression deciles per gene; ATAC-seq / DNase accessibility from ePlant if time permits.

### 5.4 Causal validation of feature interpretations

For the ≥5 top biologically-matched features:

- **Feature ablation**: zero the feature's activation, decode via SAE → residual stream, run the task classifier; measure the accuracy drop specifically on sequences that contain the feature's supposed motif.
- **Feature steering**: add scalar × decoder direction to a clean sequence; measure whether the motif-related classifier prediction shifts in the predicted direction.

If ablating a "TATA-box feature" selectively hurts promoter classification on TATA-containing sequences, we have causal evidence for the interpretation. If not, we report the discrepancy.

### 5.5 Paper changes

- Results §4.5 is rewritten to report (a) real-data SAE, (b) scale sweep, (c) motif and annotation enrichment table, (d) causal ablation result.
- A new supplementary table lists every significantly-matched feature with its TF/motif/biology.
- Discussion paragraph on SAEs is rewritten to claim *biological grounding* only where we have the motif-match + causal-ablation evidence.

---

## 6. Reporting Polish (addresses Minor Concerns)

### 6.1 Statistical reporting conventions (standardized across the paper)

- **Always state the quantity**: "5-fold CV mean ± 1 SD" or "mean with 95 % bootstrap CI (10 000 resamples)". Never mix.
- **Significance tests** where we compare methods: paired bootstrap over test sequences; report *p* and effect size. Replace "significantly outperforms" with the actual numbers.
- **Multiple-test correction** wherever applicable (per-layer tests, per-feature motif tests): Benjamini–Hochberg, q-value reported.
- **Effect sizes** (Cohen's d or equivalent) alongside p-values.
- **Error bars labelled in every caption** (currently Fig. 1 caption says "95 % CI" but Table 1 says "std"; pick one and be explicit per figure).

### 6.2 Figure quantification

Every main figure gets a small adjacent panel with a quantitative summary:

- Probing figure: table inset with best layer, accuracy, advantage over each baseline, *p*-value.
- Attention figure: panel with per-layer entropy / locality distribution — not just heatmaps.
- SAE figure: panel with fraction-of-variance-explained, L0, dead-feature rate numerically.
- Species figure: panel reporting matched-GC accuracy next to unmatched.
- Patching figure: panel with per-layer peak effect and bootstrap CIs.

### 6.3 Paper changes

- One-paragraph "Statistical conventions" subsection in Methods listing the above rules.
- Rewrite every figure caption to the new convention.

---

## 7. Implementation Plan and File Layout

### 7.1 New directory layout

```
plant-mechinterp/
├── data/
│   └── real/
│       ├── arareg/                  # §1.1
│       ├── splice/
│       ├── tfbs/
│       ├── multispecies/
│       └── splits/
├── scripts/
│   └── data/
│       ├── fetch_real_data.py       # §1.1
│       ├── build_splits.py
│       └── gc_matching.py           # §3.2, §4.3
├── analysis/
│   ├── baselines/
│   │   ├── cnn_baseline.py          # §3.1
│   │   ├── lstm_baseline.py
│   │   ├── kmer_mlp.py
│   │   └── dnabert2_probe.py
│   ├── probing/
│   │   └── probe_real.py            # §1.2
│   ├── mechanistic/
│   │   ├── activation_patching_v3.py  # §2.1
│   │   ├── path_patching.py           # §2.2
│   │   ├── attention_attribution.py   # §2.3
│   │   └── logit_lens.py              # §2.4
│   ├── cross_species/
│   │   ├── cross_species_v2.py        # §4
│   │   └── phylogeny.py
│   └── sae/
│       ├── train_sae_real.py          # §5.1
│       ├── motif_enrichment.py        # §5.2
│       ├── annotation_enrichment.py   # §5.3
│       └── feature_causal.py          # §5.4
└── REVIEWER_RESPONSE_PLAN.md         # this file
```

### 7.2 Execution order (dependencies)

1. `fetch_real_data.py` + `build_splits.py` → real-data benchmark.
2. Learned baselines (§3.1) in parallel with Plant-DnaGemma probing (§1.2) on real data.
3. Task-redesign experiments (§3.2).
4. Mechanistic upgrades (§2) — depend on real data and on a trained end-to-end probe from §1.2.
5. Cross-species v2 (§4) — depends on real data.
6. SAE (§5) — depends on real-data layer-7 activations.
7. Reporting polish (§6) — last, once numbers are stable.

Steps 2–6 can proceed in parallel once Step 1 is done.

### 7.3 Compute budget sanity check

Plant-DnaGemma is 152 M params and already runs on an RTX 2060. Scaling from 400 → ~100 000 sequences at 1 024 nt is still feasible:

- Embedding extraction: one forward pass, batch size 8, ≈ 0.5 s/sample → ~14 h for 100 k sequences. Cache to disk.
- Probing / baselines: minutes, runs on CPU or single GPU.
- Activation patching at per-position × per-layer granularity: 12 layers × 12 heads × ~500 sequences × (clean+corrupt+patched) ≈ 2–3 days on the same GPU — manageable.
- SAE training: 100 k × 768 activations × 16× expansion fits comfortably in 6 GB VRAM.

If the RTX 2060 becomes a blocker, renting an A10/A100 for 1–2 days (§2 patching sweep, §5 large SAE) is the only likely cloud cost.

### 7.4 Reproducibility

- Pin every external DB version (TAIR10, Araport11, Ensembl Plants release 58, PlantTFDB 5.0, JASPAR 2024 CORE plantae, EPDnew v1).
- Fixed seeds (0, 1, 2) for every stochastic step.
- Split files are committed as CSVs with stable hashes.
- All runs output a single JSON with metrics; figures are regenerated from JSON via deterministic plotting scripts.

---

## 8. Manuscript-Level Rewrite Checklist

Once experiments from §§1–5 are complete:

- [ ] **Abstract**: replace "74.5 %" number with the real-data headline; add one sentence on causal evidence; add one sentence on SAE biological grounding.
- [ ] **Introduction**: drop overstated "mechanistic" claims where unsupported; state upfront that evaluation is on real plant genomic data.
- [ ] **Contributions list**: rewrite to name real-data probing, causal circuit evidence, SAE-to-motif mapping, scaled phylogenetic cross-species evaluation.
- [ ] **Methods**: new "Real genomic benchmark" subsection; new "Learned baselines" subsection; new "Causal mechanistic analyses" subsection; statistical-conventions paragraph.
- [ ] **Results**: every subsection gets the real-data version; synthetic results move to supplement.
- [ ] **Discussion**: rewrite "Distributed encoding" around causal results; rewrite "GC confound" around the GC-matched and held-out-species results; rewrite SAE paragraph around the motif and annotation enrichment findings.
- [ ] **Limitations**: shrink synthetic-data caveat (no longer primary); add new caveats (pretraining overlap with evaluation genomes, scale of SAE, number of species).
- [ ] **Figures**: 6 main figures regenerated; 3–4 new supplementary figures (synthetic controls, GC-matched panels, per-feature motif logos, circuit sketch).
- [ ] **Tables**: Layer-wise table rerun on real data; new baselines table; new SAE motif-match table.

---

## 9. Honest Negative-Result Policy

Several of the new experiments may produce null or negative results (e.g., head ablations that identify no single critical head; SAE features that do not match any known motif; species representations that *still* collapse to GC after scaling). The plan commits to reporting these honestly: the paper's value is in the **methodological rigor** demonstrated, and a null result after proper controls is a stronger contribution than an inflated claim on synthetic data. Every section above includes a "what a null result looks like and what we write" clause implicitly — it will be made explicit in each subsection's write-up.

---

*End of plan.*
