#!/usr/bin/env python3
"""Fixed analysis that works with any model architecture"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_hidden_representations():
    """Analyze hidden representations from our working model"""
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Load our working model
        model_name = 'zhangtaolab/plant-dnagemma-BPE'
        print(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Move to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()
        
        print(f"Model loaded on {device}")
        
        # Load test sequences
        test_sequences = {}
        sequence_dir = Path("data/test_sequences")
        
        for seq_file in sequence_dir.glob("*.txt"):
            with open(seq_file, 'r') as f:
                sequence = f.read().strip()
                test_sequences[seq_file.stem] = sequence
        
        print(f"Loaded {len(test_sequences)} sequences")
        
        # Analyze each sequence
        sequence_representations = {}
        
        for seq_name, sequence in test_sequences.items():
            print(f"\nAnalyzing {seq_name}...")
            
            # Tokenize (limit length for memory)
            inputs = tokenizer(sequence[:500], return_tensors='pt', 
                              truncation=True, max_length=256).to(device)
            
            with torch.no_grad():
                # Get model outputs
                outputs = model(**inputs)
                
                # Extract hidden states (last layer)
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states[-1]  # Last layer
                elif hasattr(outputs, 'logits'):
                    # For generative models, get representations before final layer
                    hidden_states = None
                    print(f"  Model has logits output: {outputs.logits.shape}")
                else:
                    print(f"  Couldn't find hidden states in output")
                    continue
                
                if hidden_states is not None:
                    print(f"  Hidden states shape: {hidden_states.shape}")
                    
                    # Move to CPU and store
                    hidden_cpu = hidden_states.cpu().numpy()
                    sequence_representations[seq_name] = {
                        'hidden_states': hidden_cpu,
                        'sequence': sequence[:500],
                        'tokens': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
                        'input_ids': inputs['input_ids'].cpu().numpy()
                    }
                    
                    # Basic analysis
                    mean_activation = np.mean(hidden_cpu[0], axis=0)
                    std_activation = np.std(hidden_cpu[0], axis=0)
                    
                    print(f"  Mean activation: {np.mean(mean_activation):.3f}")
                    print(f"  Std activation: {np.std(mean_activation):.3f}")
                    print(f"  Active units (>0.1): {np.sum(np.abs(mean_activation) > 0.1)}")
        
        return sequence_representations
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}

def create_representation_visualizations(representations, save_dir="analysis/results"):
    """Create visualizations of hidden representations"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not representations:
        print("No representations to visualize")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Hidden state magnitudes by sequence
    ax = axes[0, 0]
    seq_names = list(representations.keys())
    magnitudes = []
    
    for seq_name in seq_names:
        hidden = representations[seq_name]['hidden_states'][0]  # First batch
        magnitude = np.mean(np.abs(hidden), axis=1)  # Mean across hidden dims
        magnitudes.append(magnitude)
    
    for i, (seq_name, mag) in enumerate(zip(seq_names, magnitudes)):
        ax.plot(mag, label=seq_name, alpha=0.7)
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Mean Activation Magnitude')
    ax.set_title('Hidden State Activations by Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Activation distribution
    ax = axes[0, 1]
    all_activations = []
    
    for seq_name, data in representations.items():
        hidden = data['hidden_states'][0]
        all_activations.extend(hidden.flatten())
    
    ax.hist(all_activations, bins=50, alpha=0.7, density=True)
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of All Activations')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Sequence length vs representation diversity
    ax = axes[1, 0]
    lengths = []
    diversities = []
    
    for seq_name, data in representations.items():
        length = len(data['sequence'])
        hidden = data['hidden_states'][0]
        diversity = np.std(hidden, axis=0).mean()  # Diversity across positions
        
        lengths.append(length)
        diversities.append(diversity)
        ax.scatter(length, diversity, label=seq_name, s=60, alpha=0.7)
    
    ax.set_xlabel('Sequence Length (bp)')
    ax.set_ylabel('Representation Diversity')
    ax.set_title('Sequence Length vs Representation Diversity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Token-level attention (approximation)
    ax = axes[1, 1]
    
    # Use first sequence for detailed view
    if seq_names:
        seq_name = seq_names[0]
        hidden = representations[seq_name]['hidden_states'][0]
        tokens = representations[seq_name]['tokens']
        
        # Compute token importance (L2 norm of hidden state)
        token_importance = np.linalg.norm(hidden, axis=1)
        
        positions = range(len(token_importance))
        bars = ax.bar(positions, token_importance, alpha=0.7)
        
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Representation Magnitude')
        ax.set_title(f'Token Importance: {seq_name}')
        
        # Add token labels if not too many
        if len(tokens) <= 20:
            ax.set_xticks(positions)
            ax.set_xticklabels(tokens, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'hidden_representation_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved representation analysis: {save_dir / 'hidden_representation_analysis.png'}")

def create_motif_analysis(representations, save_dir="analysis/results"):
    """Analyze representation patterns around known motifs"""
    save_dir = Path(save_dir)
    
    motifs = {
        'TATA': 'TATAAA',
        'CAAT': 'CAAT', 
        'GC_box': 'GGGCGG'
    }
    
    motif_activations = {motif: [] for motif in motifs}
    
    for seq_name, data in representations.items():
        sequence = data['sequence']
        hidden = data['hidden_states'][0]
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
            
            if motif_positions:
                print(f"  {motif_name}: {len(motif_positions)} occurrences")
                
                # Try to map sequence positions to token positions
                # This is approximate since tokenization may not align perfectly
                for seq_pos in motif_positions:
                    # Find approximate token position
                    token_pos = min(seq_pos // 4, len(hidden) - 1)  # Rough approximation
                    
                    if token_pos < len(hidden):
                        activation = np.linalg.norm(hidden[token_pos])
                        motif_activations[motif_name].append(activation)
            else:
                print(f"  {motif_name}: Not found")
    
    # Create motif activation plot
    if any(motif_activations.values()):
        plt.figure(figsize=(10, 6))
        
        motif_names = []
        motif_means = []
        motif_stds = []
        
        for motif_name, activations in motif_activations.items():
            if activations:
                motif_names.append(motif_name)
                motif_means.append(np.mean(activations))
                motif_stds.append(np.std(activations))
        
        if motif_names:
            plt.bar(motif_names, motif_means, yerr=motif_stds, 
                   alpha=0.7, capsize=5)
            plt.xlabel('Regulatory Motif')
            plt.ylabel('Mean Representation Magnitude')
            plt.title('Model Activations at Regulatory Motifs')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'motif_activation_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved motif analysis: {save_dir / 'motif_activation_analysis.png'}")

def main():
    """Main analysis pipeline"""
    print("=== FIXED Plant Mechanistic Interpretability Analysis ===")
    
    # Analyze hidden representations
    representations = analyze_hidden_representations()
    
    if representations:
        print(f"\nSuccessfully analyzed {len(representations)} sequences")
        
        # Create visualizations
        create_representation_visualizations(representations)
        create_motif_analysis(representations)
        
        # Save raw data
        results_dir = Path("analysis/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy arrays (more reliable than pickle)
        for seq_name, data in representations.items():
            np.save(results_dir / f"{seq_name}_hidden_states.npy", 
                   data['hidden_states'])
        
        print(f"\nAnalysis complete! Results saved to analysis/results/")
        print("Generated visualizations:")
        print("- hidden_representation_analysis.png")
        print("- motif_activation_analysis.png")
        print("- Raw hidden states saved as .npy files")
        
    else:
        print("No representations were successfully extracted")

if __name__ == "__main__":
    main()