# Plant Mechanistic Interpretability Data

This directory contains genomic data for analyzing plant foundation models.

## Test Sequences
- AT1G01010_promoter.txt: Example Arabidopsis promoter-like sequence (~1kb)
- random_sequence.txt: Random DNA sequence for baseline comparison (~1kb)  
- short_test.txt: Short sequence for quick testing (~50bp)

## Notes
- For full analysis, download Arabidopsis TAIR10 genome and annotations from:
  - https://www.arabidopsis.org/download/
- Sequences are in plain text format (ATCG nucleotides)
- Model tokenizer expects DNA sequence strings

## Usage
These sequences can be used to test PlantCAD2-Small model loading and inference
before running full mechanistic interpretability analysis.
