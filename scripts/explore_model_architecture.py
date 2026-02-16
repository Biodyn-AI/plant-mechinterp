#!/usr/bin/env python3
"""Explore the architecture of our working model"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

def explore_model():
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        # Load our working model
        model_name = 'zhangtaolab/plant-dnagemma-BPE'
        print(f"Exploring model architecture: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"\nModel type: {type(model)}")
        print(f"Model class: {model.__class__.__name__}")
        
        # Check model architecture
        print(f"\nModel architecture details:")
        print(f"- Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        
        # Print model structure
        print(f"\nModel layers:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                print(f"  {name}: {type(module).__name__}")
        
        # Check what outputs the model provides
        test_seq = "ATCGATCGATCG"
        inputs = tokenizer(test_seq, return_tensors='pt')
        
        print(f"\nTesting model outputs...")
        with torch.no_grad():
            # Try different output options
            try:
                outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
                print("✓ Supports output_attentions and output_hidden_states")
                
                print(f"Available output attributes:")
                for attr in dir(outputs):
                    if not attr.startswith('_'):
                        value = getattr(outputs, attr)
                        if torch.is_tensor(value):
                            print(f"  {attr}: {value.shape}")
                        elif value is not None:
                            print(f"  {attr}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
                        else:
                            print(f"  {attr}: None")
                            
            except Exception as e:
                print(f"Standard outputs failed: {e}")
                
                # Try basic output
                outputs = model(**inputs)
                print(f"Basic output available:")
                for attr in dir(outputs):
                    if not attr.startswith('_'):
                        value = getattr(outputs, attr)
                        if torch.is_tensor(value):
                            print(f"  {attr}: {value.shape}")
                        elif value is not None and hasattr(value, '__len__'):
                            print(f"  {attr}: {type(value)} (length: {len(value)})")
        
        # Check if it's actually a Gemma/LLM model
        print(f"\nModel config:")
        if hasattr(model, 'config'):
            config = model.config
            print(f"- Model type: {getattr(config, 'model_type', 'Unknown')}")
            print(f"- Architecture: {getattr(config, 'architectures', 'Unknown')}")
            print(f"- Hidden size: {getattr(config, 'hidden_size', 'Unknown')}")
            print(f"- Num layers: {getattr(config, 'num_hidden_layers', 'Unknown')}")
            print(f"- Num attention heads: {getattr(config, 'num_attention_heads', 'Unknown')}")
            print(f"- Vocab size: {getattr(config, 'vocab_size', 'Unknown')}")
            
        return True
        
    except Exception as e:
        print(f"Error exploring model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    explore_model()