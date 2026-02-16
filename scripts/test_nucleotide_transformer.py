#!/usr/bin/env python3
"""Test script for InstaDeepAI nucleotide transformer"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("Starting nucleotide transformer test...")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    model_name = 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species'
    print(f"Loading model: {model_name}")
    
    # Download and cache (trust remote code for custom model)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    print("Model loaded successfully!")
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Test inference  
    test_seq = "ATCGATCGATCGAAATTTCCCGGG"
    print(f"Test sequence: {test_seq}")
    
    inputs = tokenizer(test_seq, return_tensors='pt')
    print(f"Tokenized input shape: {inputs['input_ids'].shape}")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("Test inference successful!")
    print(f"Output shape: {outputs.last_hidden_state.shape}")
    print(f"Hidden size: {outputs.last_hidden_state.shape[-1]}")
    print(f"Sequence length: {outputs.last_hidden_state.shape[1]}")
    
    # Check if we can move to GPU
    if torch.cuda.is_available():
        print(f"CUDA available! GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    else:
        print("CUDA not available, using CPU")
    
    print("\nReady for mechanistic interpretability analysis!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()