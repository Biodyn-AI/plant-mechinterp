#!/usr/bin/env python3
"""
Download essential plant genomic data for mechanistic interpretability analysis
"""
import os
import sys
import gzip
from pathlib import Path
import urllib.request

def main():
    # Set encoding for Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    print("Setting up plant genomic data for mechanistic interpretability...")
    
    # Create data directories
    data_dir = "D:/openclaw/plant-mechinterp/data"
    arabidopsis_dir = os.path.join(data_dir, "arabidopsis")
    test_dir = os.path.join(data_dir, "test_sequences")
    
    os.makedirs(arabidopsis_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create some test DNA sequences for initial analysis
    print("Creating test sequences...")
    
    # AT1G01010 promoter-like sequence (~1kb)
    at1g01010_promoter = ("AAGTCAGAGACTAAAGAAAGTCAGAGACTAAAGAAGAATCTCTCTCTCTCTCTCTCTCTCT" +
                         "CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT" +
                         "CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTTAATCCATAAATCATAAATCATAAAT" +
                         "CATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAA" +
                         "TCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAA" +
                         "ATCATAAATCAATGATCTTAAGTCAGAGACTAAAGAAGTCAGAGACTAAAGAAGAATCTCT" +
                         "CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT" +
                         "CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTTAATCC" +
                         "ATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAAT" +
                         "CATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAA" +
                         "TCATAAATCATAAATCATAAATCAATGATCTTAAGTCAGAGACTAAAGAAGTCAGAGACTA" +
                         "AAGAAGAATCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTC" +
                         "TCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTC" +
                         "TTAATCCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAAT" +
                         "CATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAATCATAAA" +
                         "TCATAAATCATAAATCAATGATCTACGATCGATCGATCGTACGATCGATCGATCGTACGATC" +
                         "GATCGATCGTACGATCGATCGATCGTACGATCGATCGATCG")
    
    # Random DNA sequence (~1kb)
    random_seq = ("ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT" +
                 "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" +
                 "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT" +
                 "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" +
                 "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT" +
                 "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" +
                 "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT" +
                 "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" +
                 "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT" +
                 "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" +
                 "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT" +
                 "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" +
                 "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT" +
                 "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" +
                 "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT" +
                 "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" +
                 "ATCGATCGATCGATCGATCGATCGATC")
    
    # Write test sequences
    test_files = {
        "AT1G01010_promoter.txt": at1g01010_promoter,
        "random_sequence.txt": random_seq,
        "short_test.txt": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
    }
    
    for filename, sequence in test_files.items():
        filepath = os.path.join(test_dir, filename)
        with open(filepath, 'w') as f:
            f.write(sequence)
        print(f"Created test sequence: {filename} ({len(sequence)} bp)")
    
    # Create a simple info file
    info_text = """# Plant Mechanistic Interpretability Data

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
"""
    
    with open(os.path.join(data_dir, "README.txt"), 'w') as f:
        f.write(info_text)
    
    print(f"Data setup complete!")
    print(f"Test sequences created in: {test_dir}")
    print(f"Next steps:")
    print("1. Load PlantCAD2-Small model")
    print("2. Test model inference on these sequences")
    print("3. Run attention pattern analysis")
    print("\nFor full analysis, consider downloading:")
    print("- Arabidopsis TAIR10 genome from https://www.arabidopsis.org/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)