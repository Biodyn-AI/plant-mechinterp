#!/usr/bin/env python3
"""Comprehensive mechanistic interpretability analysis for Plant-DnaGemma model"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_layer_representations():
    """Analyze representations across all 12 layers of the Gemma model"""
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_name = 'zhangtaolab/plant-dnagemma-BPE'
        print(f"Loading Plant-DnaGemma model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()
        
        print(f"Model loaded on {device}")
        print(f"Architecture: Gemma with 12 layers, 12 heads, 768 hidden dims")
        
        # Load test sequences
        sequences = {}
        sequence_dir = Path("data/test_sequences")
        
        for seq_file in sequence_dir.glob("*.txt"):
            with open(seq_file, 'r') as f:
                sequence = f.read().strip()
                sequences[seq_file.stem] = sequence
        
        print(f"Loaded {len(sequences)} test sequences")
        
        # Analyze layer-wise representations
        layer_analysis = {}
        
        for seq_name, sequence in sequences.items():
            print(f"\nAnalyzing {seq_name}...")
            
            # Truncate for memory efficiency
            seq_truncated = sequence[:400]
            inputs = tokenizer(seq_truncated, return_tensors='pt', 
                              truncation=True, max_length=128).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Extract hidden states from all layers
                hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_size)
                
                print(f"  Got hidden states from {len(hidden_states)} layers")
                print(f"  Input shape: {inputs['input_ids'].shape}")
                print(f"  Hidden state shape: {hidden_states[0].shape}")
                
                # Analyze each layer
                layer_data = {}
                for layer_idx, layer_hidden in enumerate(hidden_states):
                    layer_hidden_cpu = layer_hidden.cpu().numpy()[0]  # Remove batch dim
                    
                    # Compute statistics
                    mean_activation = np.mean(layer_hidden_cpu, axis=0)
                    std_per_position = np.std(layer_hidden_cpu, axis=1)
                    l2_norm_per_position = np.linalg.norm(layer_hidden_cpu, axis=1)
                    
                    layer_data[layer_idx] = {
                        'hidden_states': layer_hidden_cpu,
                        'mean_activation': mean_activation,
                        'std_per_position': std_per_position,
                        'l2_norm_per_position': l2_norm_per_position,
                        'mean_l2_norm': np.mean(l2_norm_per_position),
                        'activation_sparsity': np.mean(np.abs(mean_activation) < 0.01),
                        'max_activation': np.max(np.abs(layer_hidden_cpu))
                    }
                
                layer_analysis[seq_name] = {
                    'layers': layer_data,
                    'tokens': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
                    'sequence': seq_truncated,
                    'input_ids': inputs['input_ids'].cpu().numpy()[0]
                }
        
        return layer_analysis
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {}

def create_layer_analysis_plots(analysis_data, save_dir="analysis/results"):
    """Create comprehensive layer analysis visualizations"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not analysis_data:
        print("No analysis data available")
        return
    
    # 1. Layer-wise activation magnitude evolution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot activation magnitudes across layers
    ax = axes[0, 0]
    for seq_name, data in analysis_data.items():
        layer_means = [data['layers'][i]['mean_l2_norm'] for i in range(13)]  # 13 layers total
        ax.plot(range(13), layer_means, 'o-', label=seq_name, linewidth=2, markersize=4)
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Mean L2 Norm of Activations')
    ax.set_title('Activation Magnitude Evolution Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot activation sparsity across layers  
    ax = axes[0, 1]
    for seq_name, data in analysis_data.items():
        sparsities = [data['layers'][i]['activation_sparsity'] for i in range(13)]
        ax.plot(range(13), sparsities, 's-', label=seq_name, linewidth=2, markersize=4)
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Activation Sparsity (fraction near zero)')
    ax.set_title('Representation Sparsity Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot max activations (saturation check)
    ax = axes[1, 0]
    for seq_name, data in analysis_data.items():
        max_acts = [data['layers'][i]['max_activation'] for i in range(13)]
        ax.plot(range(13), max_acts, '^-', label=seq_name, linewidth=2, markersize=4)
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Maximum Absolute Activation')
    ax.set_title('Maximum Activations Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Token-level analysis for first sequence
    ax = axes[1, 1]
    if analysis_data:
        seq_name = list(analysis_data.keys())[0]
        data = analysis_data[seq_name]
        
        # Show L2 norms across positions for different layers
        layers_to_show = [0, 3, 6, 9, 12]  # Sample of layers
        for layer_idx in layers_to_show:
            if layer_idx in data['layers']:
                l2_norms = data['layers'][layer_idx]['l2_norm_per_position']
                ax.plot(l2_norms, label=f'Layer {layer_idx}', alpha=0.7)
        
        ax.set_xlabel('Token Position')  
        ax.set_ylabel('L2 Norm of Hidden State')
        ax.set_title(f'Token Activations by Layer: {seq_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'layer_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved layer analysis: layer_analysis.png")

def analyze_regulatory_motif_activations(analysis_data, save_dir="analysis/results"):
    """Analyze how different layers respond to regulatory motifs"""
    save_dir = Path(save_dir)
    
    motifs = {
        'TATA': 'TATAAA',
        'CAAT': 'CAAT',
        'GC': 'GGGCGG'
    }
    
    motif_layer_analysis = {}
    
    for seq_name, data in analysis_data.items():
        sequence = data['sequence']
        tokens = data['tokens']
        
        print(f"\nMotif analysis for {seq_name}:")
        
        for motif_name, motif_seq in motifs.items():
            # Find motif positions in sequence
            motif_positions = []
            start = 0
            while True:
                pos = sequence.upper().find(motif_seq, start)
                if pos == -1:
                    break
                motif_positions.append(pos)
                start = pos + 1
            
            if not motif_positions:
                print(f"  {motif_name}: Not found")
                continue
                
            print(f"  {motif_name}: {len(motif_positions)} occurrences at {motif_positions}")
            
            # Analyze activations at motif positions across layers
            motif_activations_by_layer = []
            
            for layer_idx in range(13):
                layer_activations = []
                hidden = data['layers'][layer_idx]['hidden_states']
                
                for seq_pos in motif_positions:
                    # Map sequence position to token position (approximate)
                    token_pos = min(seq_pos // 4, len(hidden) - 1)  # Rough mapping
                    if token_pos < len(hidden):
                        activation = np.linalg.norm(hidden[token_pos])
                        layer_activations.append(activation)
                
                if layer_activations:
                    motif_activations_by_layer.append(np.mean(layer_activations))
                else:
                    motif_activations_by_layer.append(0.0)
            
            if seq_name not in motif_layer_analysis:
                motif_layer_analysis[seq_name] = {}
            motif_layer_analysis[seq_name][motif_name] = motif_activations_by_layer
    
    # Create motif analysis plot
    if motif_layer_analysis:
        fig, axes = plt.subplots(1, len(motifs), figsize=(15, 5))
        if len(motifs) == 1:
            axes = [axes]
        
        for motif_idx, motif_name in enumerate(motifs.keys()):
            ax = axes[motif_idx]
            
            for seq_name in motif_layer_analysis.keys():
                if motif_name in motif_layer_analysis[seq_name]:
                    activations = motif_layer_analysis[seq_name][motif_name]
                    ax.plot(range(13), activations, 'o-', label=seq_name, linewidth=2)
            
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Mean Activation at Motif')
            ax.set_title(f'{motif_name} Motif Responses')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'motif_layer_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved motif analysis: motif_layer_analysis.png")

def create_representation_heatmap(analysis_data, save_dir="analysis/results"):
    """Create heatmaps of hidden representations"""
    save_dir = Path(save_dir)
    
    if not analysis_data:
        return
    
    # Select first sequence for detailed visualization
    seq_name = list(analysis_data.keys())[0]
    data = analysis_data[seq_name]
    
    # Create heatmap of activations across layers and positions
    n_layers_to_show = 6  # Show subset for clarity
    layer_indices = [0, 2, 4, 6, 10, 12]
    
    fig, axes = plt.subplots(n_layers_to_show, 1, figsize=(15, 12))
    
    for i, layer_idx in enumerate(layer_indices):
        if layer_idx in data['layers']:
            hidden = data['layers'][layer_idx]['hidden_states']
            
            # Show first 50 hidden dimensions for clarity
            hidden_subset = hidden[:, :50]
            
            im = axes[i].imshow(hidden_subset.T, aspect='auto', cmap='RdBu_r', 
                               vmin=-2, vmax=2)
            axes[i].set_title(f'Layer {layer_idx} Activations')
            axes[i].set_ylabel('Hidden Dimension')
            
            if i == len(layer_indices) - 1:
                axes[i].set_xlabel('Token Position')
    
    plt.colorbar(im, ax=axes, label='Activation Value', fraction=0.02)
    plt.tight_layout()
    plt.savefig(save_dir / f'activation_heatmap_{seq_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved activation heatmap: activation_heatmap_{seq_name}.png")

def main():
    """Main mechanistic interpretability analysis"""
    print("=== Plant-DnaGemma Mechanistic Interpretability Analysis ===")
    
    # Analyze layer representations
    analysis_data = analyze_layer_representations()
    
    if analysis_data:
        print(f"\n✓ Successfully analyzed {len(analysis_data)} sequences across 13 layers")
        
        # Create visualizations
        create_layer_analysis_plots(analysis_data)
        analyze_regulatory_motif_activations(analysis_data)
        create_representation_heatmap(analysis_data)
        
        # Save summary
        results_dir = Path("analysis/results")
        
        with open(results_dir / "gemma_analysis_summary.txt", "w") as f:
            f.write("Plant-DnaGemma Mechanistic Interpretability Analysis Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: zhangtaolab/plant-dnagemma-BPE\n")
            f.write(f"Architecture: Gemma (12 layers, 12 heads, 768 dims)\n")
            f.write(f"Sequences analyzed: {len(analysis_data)}\n\n")
            
            for seq_name, data in analysis_data.items():
                f.write(f"Sequence: {seq_name}\n")
                f.write(f"  Length: {len(data['sequence'])} bp\n")
                f.write(f"  Tokens: {len(data['tokens'])}\n")
                
                # Report layer statistics
                final_layer = data['layers'][12]
                f.write(f"  Final layer mean activation: {final_layer['mean_l2_norm']:.3f}\n")
                f.write(f"  Final layer sparsity: {final_layer['activation_sparsity']:.3f}\n")
                f.write("\n")
        
        print(f"\n🎉 MECHANISTIC INTERPRETABILITY ANALYSIS COMPLETE!")
        print(f"Generated visualizations:")
        print(f"  - layer_analysis.png")
        print(f"  - motif_layer_analysis.png") 
        print(f"  - activation_heatmap_*.png")
        print(f"  - gemma_analysis_summary.txt")
        
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main()