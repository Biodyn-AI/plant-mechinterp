#!/usr/bin/env python3
"""Simple test to verify model loading and basic functionality"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

def test_model():
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        model_name = 'zhangtaolab/plant-dnagemma-BPE'
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()
        
        print(f"Model loaded on {device}")
        print(f"Model type: {type(model)}")
        
        # Test with simple sequence
        test_seq = "ATCGATCGATCG"
        inputs = tokenizer(test_seq, return_tensors='pt').to(device)
        
        print(f"Input shape: {inputs['input_ids'].shape}")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        print(f"Output type: {type(outputs)}")
        print(f"Number of hidden states: {len(outputs.hidden_states)}")
        print(f"Hidden state shape: {outputs.hidden_states[0].shape}")
        
        print("✅ Model working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model()