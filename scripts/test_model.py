#!/usr/bin/env python3
"""
Test PlantCAD2-Small model loading and run proof-of-concept analysis
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

def main():
    # Set encoding for Windows  
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    print("Testing PlantCAD2-Small model...")
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name())
    
    # Model paths
    model_path = "D:/openclaw/plant-mechinterp/models/PlantCAD2-Small"
    test_seq_path = "D:/openclaw/plant-mechinterp/data/test_sequences/AT1G01010_promoter.txt"
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        
        print("Loading model...")
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        print(f"Model loaded successfully!")
        print(f"Model config: {model.config}")
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Load test sequence
        print(f"Loading test sequence from {test_seq_path}")
        with open(test_seq_path, 'r') as f:
            test_sequence = f.read().strip()
        
        print(f"Test sequence length: {len(test_sequence)} bp")
        print(f"First 100 bp: {test_sequence[:100]}")
        
        # Tokenize sequence
        print("Tokenizing sequence...")
        inputs = tokenizer(test_sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        print(f"Input shape: {inputs['input_ids'].shape}")
        
        # Run inference
        print("Running model inference...")
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        
        print("Inference complete!")
        
        # Analyze outputs
        if hasattr(outputs, 'last_hidden_state'):
            print(f"Hidden states shape: {outputs.last_hidden_state.shape}")
        
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            print(f"Number of attention layers: {len(outputs.attentions)}")
            print(f"Attention shape (layer 0): {outputs.attentions[0].shape}")
            
            # Extract attention from first layer, first head
            attention_layer_0 = outputs.attentions[0]  # [batch, heads, seq_len, seq_len]
            attention_head_0 = attention_layer_0[0, 0].cpu().numpy()  # First head
            
            print(f"Attention head 0 shape: {attention_head_0.shape}")
            
            # Simple attention analysis
            print("Running basic attention analysis...")
            
            # Save results directory
            results_dir = "D:/openclaw/plant-mechinterp/analysis/results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Plot attention heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(attention_head_0, cmap='Blues', aspect='auto')
            plt.colorbar(label='Attention Weight')
            plt.title('Attention Pattern - Layer 0, Head 0\nPlantCAD2-Small on AT1G01010 Promoter')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            
            # Save figure
            attention_plot_path = os.path.join(results_dir, "attention_layer0_head0.png")
            plt.savefig(attention_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Attention heatmap saved: {attention_plot_path}")
            
            # Analyze attention patterns
            seq_len = attention_head_0.shape[0]
            
            # Self-attention strength (diagonal)
            self_attention = np.diag(attention_head_0)
            avg_self_attention = np.mean(self_attention)
            
            # Long-range vs short-range attention
            attention_distances = []
            attention_weights = []
            
            for i in range(seq_len):
                for j in range(seq_len):
                    distance = abs(i - j)
                    weight = attention_head_0[i, j]
                    attention_distances.append(distance)
                    attention_weights.append(weight)
            
            # Plot attention vs distance
            plt.figure(figsize=(10, 6))
            plt.scatter(attention_distances, attention_weights, alpha=0.1, s=1)
            plt.xlabel('Position Distance')
            plt.ylabel('Attention Weight')
            plt.title('Attention Weight vs Position Distance\nPlantCAD2-Small - Layer 0, Head 0')
            
            distance_plot_path = os.path.join(results_dir, "attention_vs_distance.png")
            plt.savefig(distance_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Distance plot saved: {distance_plot_path}")
            
            # Summary statistics
            print("\n=== Attention Analysis Summary ===")
            print(f"Average self-attention: {avg_self_attention:.4f}")
            print(f"Max attention weight: {np.max(attention_head_0):.4f}")
            print(f"Min attention weight: {np.min(attention_head_0):.4f}")
            print(f"Attention sparsity (% < 0.01): {np.mean(attention_head_0 < 0.01)*100:.1f}%")
            
            # Check for position-specific patterns
            query_attention_sums = np.sum(attention_head_0, axis=1)
            key_attention_sums = np.sum(attention_head_0, axis=0)
            
            print(f"Most attended-to position (query): {np.argmax(query_attention_sums)}")
            print(f"Most attended-to position (key): {np.argmax(key_attention_sums)}")
            
        else:
            print("Warning: No attention outputs available")
        
        # Save analysis results
        results_summary = f"""
Plant Foundation Model Analysis Results
=====================================

Model: PlantCAD2-Small
Test Sequence: AT1G01010_promoter.txt ({len(test_sequence)} bp)
Date: 2026-02-13

Model Info:
- Total parameters: {total_params:,}
- Vocab size: {tokenizer.vocab_size}
- Input shape: {inputs['input_ids'].shape}

Attention Analysis (Layer 0, Head 0):
- Average self-attention: {avg_self_attention:.4f}
- Max attention weight: {np.max(attention_head_0):.4f}
- Min attention weight: {np.min(attention_head_0):.4f}
- Attention sparsity (% < 0.01): {np.mean(attention_head_0 < 0.01)*100:.1f}%

Generated Files:
- attention_layer0_head0.png: Attention heatmap
- attention_vs_distance.png: Distance analysis

Next Steps:
1. Analyze all 24 layers x 12 heads = 288 attention heads
2. Compare attention patterns across different sequences
3. Run activation patching experiments
4. Apply sparse autoencoders for feature discovery
"""
        
        summary_path = os.path.join(results_dir, "analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(results_summary)
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {results_dir}")
        print(f"Summary: {summary_path}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)