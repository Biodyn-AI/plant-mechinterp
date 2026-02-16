#!/usr/bin/env python3
"""Test script for zhangtaolab plant models"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("Testing zhangtaolab plant models...")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    # Try different models in order of preference
    models_to_try = [
        'zhangtaolab/plant-nucleotide-transformer-BPE',
        'zhangtaolab/plant-dnagemma-BPE',
        'zhangtaolab/plant-dnabert-BPE'
    ]
    
    for model_name in models_to_try:
        print(f"\nTrying model: {model_name}")
        try:
            # Download and cache
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            print("Loading model...")
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
            
            print(f"\n*** SUCCESS: {model_name} is working! ***")
            
            # Save successful model name for analysis scripts
            with open("models/successful_model.txt", "w") as f:
                f.write(model_name)
            
            break  # Success, stop trying other models
            
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
    
    else:
        print("\nAll models failed to load!")
    
except Exception as e:
    print(f"Critical error: {e}")
    import traceback
    traceback.print_exc()