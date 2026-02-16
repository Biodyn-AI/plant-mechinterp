#!/usr/bin/env python3
"""
Debug version to understand why activation differences are zero
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
from pathlib import Path
import random

def debug_tokenization():
    """Debug how sequences are being tokenized"""
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_name = 'zhangtaolab/plant-dnagemma-BPE'
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()
        
        print(f"Model loaded on {device}")
        
        # Test sequences
        sequence_file = Path("data/test_sequences/arabidopsis_AT1G01010_real.txt")
        with open(sequence_file, 'r') as f:
            original_sequence = f.read().strip()
        
        # Focus on a shorter region containing TATA and CAAT motifs
        # Find the TATA motif position
        tata_pos = original_sequence.upper().find('TATAAA')
        caat_pos = original_sequence.upper().find('CAAT')
        
        print(f"TATA motif at position: {tata_pos}")
        print(f"CAAT motif at position: {caat_pos}")
        
        # Extract region around these motifs (200bp window)
        start_pos = max(0, min(tata_pos, caat_pos) - 100)
        end_pos = min(len(original_sequence), max(tata_pos, caat_pos) + 100)
        
        clean_seq = original_sequence[start_pos:end_pos]
        print(f"\nFocused sequence length: {len(clean_seq)} bp")
        print(f"Clean sequence: {clean_seq}")
        
        # Create corrupted version - replace motifs with more dramatic changes
        corrupted_seq = clean_seq.replace('TATAAA', 'GCGCGC')  # Complete change
        corrupted_seq = corrupted_seq.replace('CAAT', 'TTGG')  # Complete change
        
        print(f"\nCorrupted sequence: {corrupted_seq}")
        print(f"Differences: {sum(1 for a, b in zip(clean_seq, corrupted_seq) if a != b)} positions")
        
        # Tokenize both
        clean_tokens = tokenizer(clean_seq, return_tensors='pt', 
                                truncation=True, max_length=128).to(device)
        corrupted_tokens = tokenizer(corrupted_seq, return_tensors='pt',
                                   truncation=True, max_length=128).to(device)
        
        print(f"\nClean tokens shape: {clean_tokens['input_ids'].shape}")
        print(f"Corrupted tokens shape: {corrupted_tokens['input_ids'].shape}")
        
        # Compare token IDs
        clean_ids = clean_tokens['input_ids'][0].cpu().numpy()
        corrupted_ids = corrupted_tokens['input_ids'][0].cpu().numpy()
        
        print(f"\nToken comparison:")
        print(f"Clean token IDs: {clean_ids}")
        print(f"Corrupted token IDs: {corrupted_ids}")
        print(f"Token differences: {np.sum(clean_ids != corrupted_ids)} out of {len(clean_ids)}")
        
        # Convert back to check actual tokens
        clean_tokens_text = tokenizer.convert_ids_to_tokens(clean_ids)
        corrupted_tokens_text = tokenizer.convert_ids_to_tokens(corrupted_ids)
        
        print(f"\nClean tokens: {clean_tokens_text}")
        print(f"Corrupted tokens: {corrupted_tokens_text}")
        
        # Run model on both
        with torch.no_grad():
            clean_outputs = model(**clean_tokens, output_hidden_states=True)
            corrupted_outputs = model(**corrupted_tokens, output_hidden_states=True)
            
        print(f"\nModel outputs:")
        print(f"Clean final hidden state shape: {clean_outputs.hidden_states[-1].shape}")
        print(f"Corrupted final hidden state shape: {corrupted_outputs.hidden_states[-1].shape}")
        
        # Check if identical
        final_clean = clean_outputs.hidden_states[-1]
        final_corrupted = corrupted_outputs.hidden_states[-1]
        
        are_identical = torch.allclose(final_clean, final_corrupted, atol=1e-6)
        print(f"Final outputs identical? {are_identical}")
        
        if not are_identical:
            diff = torch.norm(final_clean - final_corrupted).item()
            print(f"L2 difference: {diff}")
        
        # Check layer by layer
        print(f"\nLayer-by-layer differences:")
        for i, (clean_hidden, corrupted_hidden) in enumerate(zip(clean_outputs.hidden_states, corrupted_outputs.hidden_states)):
            diff = torch.norm(clean_hidden - corrupted_hidden).item()
            print(f"Layer {i}: {diff:.6f}")
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_tokenization()