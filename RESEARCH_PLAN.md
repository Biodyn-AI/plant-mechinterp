# Mechanistic Interpretability of Plant Foundation Models

## Executive Summary

This project represents the first systematic application of deep mechanistic interpretability techniques to plant foundation models. While PlantRNA-FM (2024, Nature Machine Intelligence) performed surface-level attention analysis on plant RNA sequences, no prior work has applied advanced mechanistic interpretability methods (activation patching, sparse autoencoders, causal tracing) to understand what plant DNA foundation models learn at a mechanistic level.

## 1. Research Questions

### Primary Research Questions

**Q1: What genomic features do plant DNA foundation models learn to recognize?**
- Do models develop specialized circuits for identifying promoters, enhancers, TATA boxes, and transcription factor binding sites?
- How do models represent different types of transposable elements and their regulatory impacts?
- Can we identify feature detectors for splice sites, UTR regions, and coding sequences?

**Q2: How do plant foundation models represent evolutionary and species-specific information?**
- Do different layers specialize in conserved vs. species-specific regulatory elements?
- How do models handle orthologous genes across the 65 angiosperm species in PlantCAD2's training data?
- Can we extract phylogenetic relationships from internal representations?

**Q3: What computational circuits emerge for gene regulatory network inference?**
- Do attention heads learn to connect transcription factors with their target genes?
- How do models represent long-range chromatin interactions and enhancer-promoter loops?
- Can we identify circuits that compute gene expression predictions based on regulatory context?

**Q4: How does information flow through the transformer architecture?**
- Which layers focus on local features (motifs, short regulatory elements) vs. global context (gene neighborhoods)?
- How do attention patterns change from early to late layers?
- What information is compressed or transformed at each layer?

### Secondary Research Questions

**Q5: Can mechanistic understanding improve model performance?**
- Can we use circuit analysis to improve few-shot learning for new plant species?
- How do different attention head types contribute to downstream tasks?
- Can we identify and remove spurious correlations learned from training data biases?

## 2. Models for Analysis

### Primary Target: PlantCAD2-Small
- **Architecture**: 24-layer transformer, 768-dimensional embeddings
- **Training data**: 65 angiosperm genomes
- **Parameters**: 676M (fits comfortably on RTX 2060 6GB VRAM)
- **HuggingFace**: kuleshov-group/PlantCAD2-Small-l24-d0768
- **Rationale**: Manageable size for detailed analysis, well-documented architecture

### Secondary Target: AgroNT (if computational resources permit)
- **Architecture**: 1B parameters
- **HuggingFace**: zhangtaolab/agront-1b
- **Rationale**: Larger model for comparison, different training methodology

### Baseline Comparisons
- Random embeddings and attention patterns
- PlantRNA-FM attention patterns (for methodological comparison)
- Simple CNN baselines on same datasets

## 3. Mechanistic Interpretability Techniques

### Phase 1: Attention Pattern Analysis (Weeks 1-2)
**Building on PlantRNA-FM methodology but going deeper:**
- Extract attention patterns from all 24 layers × 12 heads = 288 attention heads
- Cluster attention heads by their attention patterns on diverse genomic regions
- Identify heads that attend to specific motifs, structural features, or positional patterns
- Compare attention patterns across different species and gene families

**Novel contributions beyond PlantRNA-FM:**
- Systematic analysis across all layers (not just selected examples)
- Quantitative clustering and classification of attention head types
- Species-comparative analysis of attention patterns

### Phase 2: Activation Patching & Causal Tracing (Weeks 3-4)
**Techniques from transformer mechanistic interpretability:**
- **Activation Patching**: Replace activations from specific components with corrupted/baseline activations to test causal importance
- **Path Patching**: Trace information flow through specific computational paths
- **Head Patching**: Isolate the causal contribution of individual attention heads
- **Layer Patching**: Determine which layers are most important for different genomic predictions

**Applications to plant genomics:**
- Which attention heads are causally important for promoter prediction?
- How do models use transcription factor binding site information to predict expression?
- What happens when we ablate species-specific vs. conserved information?

### Phase 3: Probing Classifiers (Weeks 5-6)
**Linear probing to extract learned representations:**
- Train linear classifiers on internal activations to predict:
  - Gene ontology annotations
  - Transcription factor binding sites
  - Expression levels across tissues/conditions
  - Evolutionary conservation scores
  - Species identity

**Analysis across layers:**
- Track how different types of information emerge and transform through the network
- Identify when local features get combined into higher-order representations
- Measure information compression and disentanglement

### Phase 4: Sparse Autoencoders (Weeks 7-10)
**Feature discovery and neuron interpretability:**
- Train sparse autoencoders on activations from key layers to identify monosemantic features
- Discover what individual "neurons" represent in the context of plant genomics
- Map features to biological concepts (e.g., specific transcription factor motifs, chromatin states)

**Technical approach:**
- Use techniques from Anthropic's sparse autoencoder work adapted for genomic data
- Focus on layers where probing suggests rich feature representations
- Validate discovered features against known plant biology

### Phase 5: Logit Lens & Vocabulary Projection (Weeks 11-12)
**Understanding output representations:**
- Project internal activations through the final layer to understand what the model "thinks" at each layer
- Analyze how nucleotide predictions evolve through the forward pass
- Identify where biological concepts get translated into sequence-level predictions

## 4. Data Requirements

### Core Datasets

**Arabidopsis thaliana Reference (TAIR10)**
- Complete genome sequence (~125 MB)
- Gene annotations with GO terms
- Known regulatory elements from TAIR database
- Expression data from AtGenExpress
- **Justification**: Best-annotated plant genome, compact size, standard model organism

**Regulatory Element Databases**
- PlantRegMap: Transcription factor binding sites across plant species
- PlantPAN 3.0: Promoter analysis database
- AthaMap: Arabidopsis transcription factor binding sites
- **Size limit**: <1GB total for initial analysis

**Additional Plant Genomes (for comparative analysis)**
- Oryza sativa (rice) - major crop, different plant family
- Solanum lycopersicum (tomato) - eudicot comparison
- **Total size**: <500MB each, selected chromosome regions

### Benchmark Datasets
- DREAM5 plant gene regulatory network challenge data
- Plant expression quantitative trait loci (eQTL) datasets
- Cross-species conservation scores for regulatory elements

## 5. Timeline and Milestones

### Week 1: Infrastructure and Baseline Analysis
- **Days 1-2**: Complete model setup, verify inference capabilities
- **Days 3-5**: Basic attention pattern extraction and visualization
- **Days 6-7**: Literature review integration, methodology refinement

**Deliverable**: Working analysis pipeline, initial attention visualizations

### Week 2: Attention Pattern Deep Dive
- **Days 8-10**: Systematic attention head clustering and characterization
- **Days 11-12**: Species-comparative attention analysis
- **Days 13-14**: Motif discovery in attention patterns

**Deliverable**: Complete attention pattern atlas, head taxonomy

### Weeks 3-4: Causal Interventions
- **Week 3**: Implement activation patching framework
- **Week 4**: Run causal experiments on key biological questions
**Deliverable**: Causal circuit diagrams for key plant genomic functions

### Weeks 5-6: Probing Experiments
- **Week 5**: Train probing classifiers across layers and tasks
- **Week 6**: Analyze information flow and representation learning
**Deliverable**: Layer-wise information analysis, representation quality metrics

### Weeks 7-10: Sparse Autoencoder Feature Discovery
- **Weeks 7-8**: Train sparse autoencoders on key layers
- **Weeks 9-10**: Interpret and validate discovered features
**Deliverable**: Monosemantic feature catalog with biological validation

### Weeks 11-12: Integration and Validation
- **Week 11**: Logit lens analysis, integration of all techniques
- **Week 12**: Validation against held-out biological datasets
**Deliverable**: Comprehensive mechanistic model of PlantCAD2-Small

### Month 3: Extension and Publication
- Compare with AgroNT (if computationally feasible)
- Develop mechanistic interpretability tools for plant genomics
- Write and submit research paper

## 6. Expected Outputs

### Research Papers
**Primary Paper**: "Mechanistic Interpretability of Plant Foundation Models"
- Target venue: Nature Machine Intelligence, Cell Systems, or ICML/NeurIPS
- Novel contributions: First mechanistic analysis of plant foundation models
- Biological insights: Circuit-level understanding of plant genomic feature learning

**Secondary Papers**:
- "Attention Patterns in Plant DNA Transformers: A Systematic Analysis"
- "Sparse Feature Discovery in Plant Foundation Model Representations"

### Software and Tools
**PlantMechInterp Toolkit**
- Open-source Python package for mechanistic analysis of genomic foundation models
- Jupyter notebooks demonstrating key techniques
- Integration with HuggingFace ecosystem

**Attention Visualization Platform**
- Interactive web tool for exploring attention patterns in plant sequences
- Compare across species, genes, and regulatory regions
- Educational resource for plant biology and ML communities

### Datasets and Resources
**Plant Foundation Model Interpretability Benchmark**
- Curated datasets for evaluating mechanistic understanding
- Ground truth annotations for attention pattern validation
- Cross-species comparative analysis datasets

**Feature Atlas**
- Comprehensive catalog of learned features in plant foundation models
- Biological validation and interpretation of discovered circuits
- Integration with existing plant genomics databases

## 7. Novelty and Significance

### Technical Novelty
**First application of modern mechanistic interpretability to plant genomics:**
- PlantRNA-FM (2024) only performed surface-level attention visualization
- No prior work has applied activation patching, sparse autoencoders, or causal tracing to plant foundation models
- Bridging cutting-edge AI interpretability with computational plant biology

### Methodological Contributions
**Adapting mechanistic interpretability for genomic sequences:**
- Develop techniques for handling variable-length sequences and positional information
- Create biological validation frameworks for discovered features
- Establish best practices for interpretability in genomic foundation models

### Biological Significance
**Understanding how AI models learn plant biology:**
- Discover which aspects of plant regulatory networks are learnable from sequence alone
- Identify biases and limitations in current foundation model approaches
- Guide development of more biologically-informed architectures

**Applications to plant breeding and biotechnology:**
- Use mechanistic insights to improve few-shot learning for crop species
- Identify sequence patterns most important for specific agronomic traits
- Develop interpretable models for regulatory element design

### Broader Impact
**Advancing AI safety and interpretability:**
- Demonstrate mechanistic interpretability techniques in a new scientific domain
- Develop methods for validating AI representations against ground truth biological knowledge
- Create frameworks for scientist-in-the-loop interpretability

## 8. Risk Assessment and Mitigation

### Technical Risks
**Risk**: Model too large for detailed analysis on RTX 2060
- **Mitigation**: Focus on PlantCAD2-Small, use gradient checkpointing, analyze subset of layers

**Risk**: Sparse autoencoders may not find interpretable features
- **Mitigation**: Start with attention head analysis, validate on known motifs, adjust sparsity

### Biological Risks
**Risk**: Discovered features may not correspond to known biology
- **Mitigation**: Validate against multiple plant biology databases, collaborate with plant biologists

**Risk**: Model representations may be too abstract for biological interpretation
- **Mitigation**: Start with known regulatory elements, work from simple to complex features

### Resource Risks
**Risk**: Computational requirements exceed hardware capabilities
- **Mitigation**: Implement efficient analysis pipelines, use cloud resources for large experiments

## 9. Success Metrics

### Technical Metrics
- Successfully extract and cluster 288 attention heads from PlantCAD2-Small
- Achieve >80% accuracy in probing classifiers for known biological features
- Discover >50 interpretable features through sparse autoencoders
- Complete causal tracing for >10 key biological functions

### Biological Metrics
- Validate >70% of discovered features against known plant biology
- Identify previously unknown regulatory patterns in ≥3 species
- Demonstrate biological relevance through expression prediction improvements

### Impact Metrics
- Submit research to top-tier venue within 3 months
- Open-source toolkit with ≥100 GitHub stars within 6 months
- Adoption by plant biology research groups

## 10. Future Directions

### Short-term Extensions (Months 4-6)
- Analysis of fine-tuned models for specific plant tasks
- Cross-domain comparison with human/animal genomic foundation models
- Development of mechanistic-informed architectures

### Long-term Vision (Years 1-3)
- Mechanistic interpretability for multi-modal plant foundation models (DNA + RNA + protein)
- Real-time interpretability tools for plant breeding applications
- Foundation model architectures designed for biological interpretability

## Conclusion

This project represents a groundbreaking application of mechanistic interpretability to plant genomics. By systematically analyzing how PlantCAD2-Small learns and represents plant biological concepts, we will advance both AI interpretability science and computational plant biology. The techniques developed will establish a new paradigm for understanding foundation models in biological domains, with immediate applications to plant breeding, biotechnology, and biological discovery.

---

*Project initiated: February 13, 2026*  
*Lead: OpenClaw Agent*  
*Hardware: RTX 2060, 16GB RAM*  
*Estimated duration: 3 months*  
*Budget: Computational only (local resources)*