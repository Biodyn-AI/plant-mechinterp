#!/usr/bin/env python3
"""Attention pattern analysis for plant genomic sequences"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def load_test_sequences():
    """Load test sequences from data directory"""
    data_dir = Path("data/test_sequences")
    sequences = {}
    
    for seq_file in data_dir.glob("*.txt"):
        with open(seq_file, 'r') as f:
            sequences[seq_file.stem] = f.read().strip()
    
    return sequences

def extract_attention_patterns(model, tokenizer, sequences, device='cpu'):
    """Extract attention patterns from model for given sequences"""
    model.to(device)
    model.eval()
    
    attention_data = {}
    
    for seq_name, sequence in sequences.items():
        print(f"Processing {seq_name}: {len(sequence)} bp")
        
        # Tokenize
        inputs = tokenizer(sequence, return_tensors='pt').to(device)
        
        # Forward pass with attention output
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Extract attention weights
        attention_weights = outputs.attentions  # Tuple of (batch, heads, seq, seq)
        
        attention_data[seq_name] = {
            'sequence': sequence,
            'tokens': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'attention_weights': [att.cpu().numpy() for att in attention_weights],
            'hidden_states': outputs.last_hidden_state.cpu().numpy()
        }
    
    return attention_data

def analyze_regulatory_motifs(attention_data, motifs=None):
    """Analyze attention patterns for known regulatory motifs"""
    if motifs is None:
        motifs = {
            'TATA': 'TATAAA',
            'CAAT': 'CAAT',
            'GC_box': 'GGGCGG',
            'AT_rich': 'AAAA',
            'CpG': 'CG'
        }
    
    motif_attention = {}
    
    for seq_name, data in attention_data.items():
        sequence = data['sequence']
        tokens = data['tokens']
        
        # Find motif positions
        motif_positions = {}
        for motif_name, motif_seq in motifs.items():
            positions = []
            start = 0
            while True:
                pos = sequence.find(motif_seq, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            motif_positions[motif_name] = positions
        
        # Analyze attention to motifs across layers and heads
        layer_head_motif_attention = []
        for layer_idx, layer_att in enumerate(data['attention_weights']):
            # layer_att shape: (batch=1, heads, seq_len, seq_len)
            layer_att = layer_att[0]  # Remove batch dimension
            
            head_motif_attention = []
            for head_idx in range(layer_att.shape[0]):
                head_att = layer_att[head_idx]
                
                motif_att_scores = {}
                for motif_name, positions in motif_positions.items():
                    scores = []
                    for pos in positions:
                        # Average attention TO motif positions
                        motif_score = np.mean(head_att[:, pos:pos+len(motifs[motif_name])])
                        scores.append(motif_score)
                    motif_att_scores[motif_name] = np.mean(scores) if scores else 0.0
                
                head_motif_attention.append(motif_att_scores)
            layer_head_motif_attention.append(head_motif_attention)
        
        motif_attention[seq_name] = {
            'motif_positions': motif_positions,
            'layer_head_attention': layer_head_motif_attention
        }
    
    return motif_attention

def create_attention_heatmaps(attention_data, save_dir="analysis/results"):
    """Create attention heatmaps for visualization"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for seq_name, data in attention_data.items():
        # Create subplot for each layer's attention pattern
        n_layers = len(data['attention_weights'])
        n_heads = data['attention_weights'][0].shape[1]
        
        fig, axes = plt.subplots(n_layers, 1, figsize=(15, 3*n_layers))
        if n_layers == 1:
            axes = [axes]
        
        for layer_idx, layer_att in enumerate(data['attention_weights']):
            # Average across heads for visualization
            avg_attention = np.mean(layer_att[0], axis=0)
            
            sns.heatmap(avg_attention, ax=axes[layer_idx], 
                       cmap='viridis', cbar=True)
            axes[layer_idx].set_title(f"Layer {layer_idx+1} - Average Attention")
            axes[layer_idx].set_xlabel("Token Position")
            axes[layer_idx].set_ylabel("Token Position")
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{seq_name}_attention_heatmap.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved attention heatmap: {seq_name}_attention_heatmap.png")

def plot_motif_attention_summary(motif_attention, save_dir="analysis/results"):
    """Plot summary of attention to regulatory motifs"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate data across sequences
    all_motifs = set()
    for seq_data in motif_attention.values():
        for layer_data in seq_data['layer_head_attention']:
            for head_data in layer_data:
                all_motifs.update(head_data.keys())
    
    motifs_list = sorted(list(all_motifs))
    
    # Create summary plot
    fig, axes = plt.subplots(1, len(motifs_list), figsize=(4*len(motifs_list), 6))
    if len(motifs_list) == 1:
        axes = [axes]
    
    for motif_idx, motif in enumerate(motifs_list):
        motif_scores = []
        layer_labels = []
        head_labels = []
        
        for seq_name, seq_data in motif_attention.items():
            for layer_idx, layer_data in enumerate(seq_data['layer_head_attention']):
                for head_idx, head_data in enumerate(layer_data):
                    score = head_data.get(motif, 0.0)
                    motif_scores.append(score)
                    layer_labels.append(layer_idx)
                    head_labels.append(head_idx)
        
        # Create scatter plot
        axes[motif_idx].scatter(layer_labels, motif_scores, alpha=0.6)
        axes[motif_idx].set_title(f"Attention to {motif}")
        axes[motif_idx].set_xlabel("Layer")
        axes[motif_idx].set_ylabel("Attention Score")
    
    plt.tight_layout()
    plt.savefig(save_dir / "motif_attention_summary.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved motif attention summary: motif_attention_summary.png")

def main():
    """Main analysis pipeline"""
    print("Starting attention pattern analysis...")
    
    # Load successful model
    with open("models/successful_model.txt", "r") as f:
        model_name = f.read().strip()
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load test sequences
    sequences = load_test_sequences()
    print(f"Loaded {len(sequences)} test sequences")
    
    # Extract attention patterns
    print("Extracting attention patterns...")
    attention_data = extract_attention_patterns(model, tokenizer, sequences, device)
    
    # Analyze regulatory motifs
    print("Analyzing regulatory motifs...")
    motif_attention = analyze_regulatory_motifs(attention_data)
    
    # Create visualizations
    print("Creating visualizations...")
    create_attention_heatmaps(attention_data)
    plot_motif_attention_summary(motif_attention)
    
    # Save analysis data
    results_dir = Path("analysis/results")
    torch.save({
        'attention_data': attention_data,
        'motif_attention': motif_attention
    }, results_dir / "attention_analysis_results.pt")
    
    print("Analysis complete! Results saved to analysis/results/")

if __name__ == "__main__":
    main()