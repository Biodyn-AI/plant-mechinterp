#!/usr/bin/env python3
"""
Download PlantCAD2-Small model from HuggingFace
"""
import os
import sys
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download

def main():
    # Set encoding for Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Model details
    model_name = "kuleshov-group/PlantCAD2-Small-l24-d0768"
    local_dir = "D:/openclaw/plant-mechinterp/models/PlantCAD2-Small"
    
    print("Starting download of PlantCAD2-Small...")
    print(f"Model: {model_name}")
    print(f"Destination: {local_dir}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download model files using snapshot_download
        print("Downloading model files...")
        downloaded_path = snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Copy files instead of symlinks on Windows
            cache_dir=None  # Use default cache
        )
        
        print(f"Model downloaded successfully to: {downloaded_path}")
        
        # Test loading the model
        print("Testing model loading...")
        
        try:
            # Try loading tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
            print("Tokenizer loaded successfully")
            print(f"Vocabulary size: {tokenizer.vocab_size}")
            
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            print("This might be expected if model uses custom tokenization")
        
        try:
            # Try loading model
            model = AutoModel.from_pretrained(local_dir)
            print("Model loaded successfully")
            print(f"Model architecture: {model.config.architectures}")
            print(f"Hidden size: {model.config.hidden_size}")
            print(f"Number of layers: {model.config.num_hidden_layers}")
            print(f"Number of attention heads: {model.config.num_attention_heads}")
            
            # Get model size
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"Warning: Could not load model for testing: {e}")
            print("Model files downloaded but may need custom loading code")
        
        print("\nDownload completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)