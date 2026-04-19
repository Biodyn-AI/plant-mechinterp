# Mechanistic Interpretability of Plant Foundation Models

This repository contains a systematic mechanistic-interpretability study of plant DNA foundation models, focused on Plant-DnaGemma. It includes an end-to-end reproducible pipeline: benchmark construction from TAIR10/Araport11/EPDnew/Ensembl Plants, per-layer probing with learned baselines, GC-matched composition controls, cross-species analysis with phylogenetic residualization, sparse-autoencoder feature decomposition with JASPAR/PlantTFDB motif grounding, and causal per-component ablations.

**Archival DOI (tagged snapshot + trained SAE weights + built datasets):** [10.5281/zenodo.18665058](https://doi.org/10.5281/zenodo.18665058)

## Project Overview

- **Models**: PlantCAD2-Small (24 layers, 768d, 676M parameters)
- **Techniques**: Attention analysis, activation patching, sparse autoencoders, probing classifiers
- **Species**: Focus on Arabidopsis thaliana with cross-species validation
- **Hardware**: RTX 2060, 16GB RAM

## Quick Start

```bash
# Set up environment
set PYTHONIOENCODING=utf-8
C:\Users\Agent\miniconda3\envs\bioinfo\python.exe

# Load model (after download)
python scripts/load_model.py

# Run attention analysis
python analysis/attention_patterns.py
```

## Directory Structure

```
├── data/                  # Plant genomic datasets
├── models/                # Downloaded foundation models
├── analysis/              # Analysis scripts and notebooks
│   └── results/          # Generated figures and outputs
├── paper/                 # Manuscript and figures
├── literature/           # Relevant papers and references
├── scripts/              # Utility scripts
└── RESEARCH_PLAN.md      # Detailed research plan
```

## Key Research Questions

1. What genomic features do plant DNA models learn?
2. How do models represent species-specific vs. conserved elements?
3. What computational circuits emerge for gene regulatory networks?
4. How does information flow through transformer layers?

## Timeline

- **Weeks 1-2**: Attention pattern analysis
- **Weeks 3-4**: Activation patching and causal tracing  
- **Weeks 5-6**: Probing classifier experiments
- **Weeks 7-10**: Sparse autoencoder feature discovery
- **Weeks 11-12**: Integration and validation

## Contributing

This is an active research project. See `RESEARCH_PLAN.md` for detailed methodology and objectives.

## Citation

```bibtex
@misc{plantmechinterp2026,
  title={Mechanistic Interpretability of Plant Foundation Models},
  author={Ihor Kendiukhov},
  year={2026},
  note={In preparation}
}
```

## References

- PlantRNA-FM: Nature Machine Intelligence (2024) - baseline attention analysis
- PlantCAD2: HuggingFace kuleshov-group/PlantCAD2-Small-l24-d0768  
- AgroNT: HuggingFace zhangtaolab/agront-1b

---
