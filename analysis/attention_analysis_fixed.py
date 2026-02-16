#!/usr/bin/env python3
"""
PHASE 2 - TASK 2: Attention Pattern Analysis (FIXED VERSION)
Plant Mechanistic Interpretability Project

Extract and analyze attention patterns from the Plant-DnaGemma model
to understand how different attention heads specialize and process regulatory motifs.
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AttentionAnalyzer:
    """Analyzes attention patterns in Plant-DnaGemma model"""
    
    def __init__(self, model_name: str = 'zhangtaolab/plant-dnagemma-BPE'):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
        
    def load_model(self):
        """Load the Plant-DnaGemma model"""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Set attention implementation to eager to capture attention weights
            if hasattr(self.model, 'set_attn_implementation'):
                self.model.set_attn_implementation('eager')
                print("Set attention implementation to 'eager' for attention extraction")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Get model architecture details
            config = self.model.config
            self.num_layers = config.num_hidden_layers
            self.num_heads = config.num_attention_heads
            self.hidden_size = config.hidden_size
            
            print(f"Model loaded on {self.device}")
            print(f"Architecture: {self.num_layers} layers, {self.num_heads} heads, {self.hidden_size}d")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def find_motifs(self, sequence: str, motifs: Dict[str, str]) -> Dict[str, List[int]]:
        """Find positions of regulatory motifs in sequence"""
        sequence_upper = sequence.upper()
        motif_positions = {}
        
        for motif_name, motif_seq in motifs.items():
            positions = []
            start = 0
            while True:
                pos = sequence_upper.find(motif_seq.upper(), start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            motif_positions[motif_name] = positions
            
        return motif_positions
    
    def extract_attention_patterns(self, sequence: str, max_length: int = 100) -> Tuple[List[torch.Tensor], List[str], List[int]]:
        """Extract attention patterns for a sequence"""
        inputs = self.tokenizer(sequence, return_tensors='pt', 
                               truncation=True, max_length=max_length).to(self.device)
        
        print(f"Input token IDs shape: {inputs['input_ids'].shape}")
        
        with torch.no_grad():
            try:
                outputs = self.model(**inputs, output_attentions=True)
                attention_weights = outputs.attentions
                
                if attention_weights is None:
                    print("Warning: No attention weights returned, trying alternative approach")
                    # Try without specifying output_attentions
                    outputs = self.model(**inputs)
                    if hasattr(outputs, 'attentions'):
                        attention_weights = outputs.attentions
                    else:
                        print("Model does not support attention extraction")
                        return None, [], []
                        
            except Exception as e:
                print(f"Error extracting attention: {e}")
                return None, [], []
            
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        token_ids = inputs['input_ids'][0].cpu().numpy()
        
        return attention_weights, tokens, token_ids
    
    def create_simple_attention_visualizations(self, attention_weights: List[torch.Tensor], 
                                             tokens: List[str], motif_positions: Dict[str, List[int]], 
                                             sequence: str, save_dir: Path):
        """Create basic attention visualizations"""
        
        if attention_weights is None or len(attention_weights) == 0:
            print("No attention weights to visualize")
            return
            
        print(f"Creating visualizations for {len(attention_weights)} layers")
        
        # 1. Layer-wise average attention patterns
        n_layers_to_show = min(6, len(attention_weights))
        layer_indices = [0, 2, 4, 6, 8, 10][:n_layers_to_show]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx < len(attention_weights) and i < len(axes):
                # Get attention for this layer: (batch=1, num_heads, seq_len, seq_len)
                layer_attention = attention_weights[layer_idx][0]  # Remove batch dimension
                
                # Average across all heads
                avg_attention = torch.mean(layer_attention, dim=0).cpu().numpy()
                
                im = axes[i].imshow(avg_attention, cmap='Blues', aspect='auto', vmin=0, vmax=0.3)
                axes[i].set_title(f'Layer {layer_idx} - Avg Attention Pattern', fontweight='bold')
                axes[i].set_xlabel('Key Position (Token)')
                axes[i].set_ylabel('Query Position (Token)')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                
                # Add grid
                axes[i].grid(True, alpha=0.2)
        
        # Hide unused subplots
        for i in range(len(layer_indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'attention_patterns_layers.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: attention_patterns_layers.png")
        
        # 2. Individual head analysis for one layer
        middle_layer = len(attention_weights) // 2
        if middle_layer < len(attention_weights):
            layer_attention = attention_weights[middle_layer][0]  # (num_heads, seq_len, seq_len)
            
            n_heads = min(6, layer_attention.shape[0])
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for head_idx in range(n_heads):
                head_attention = layer_attention[head_idx].cpu().numpy()
                
                im = axes[head_idx].imshow(head_attention, cmap='Blues', aspect='auto', vmin=0, vmax=0.3)
                axes[head_idx].set_title(f'Layer {middle_layer}, Head {head_idx}', fontweight='bold')
                axes[head_idx].set_xlabel('Key Position')
                axes[head_idx].set_ylabel('Query Position')
                
                plt.colorbar(im, ax=axes[head_idx], fraction=0.046, pad=0.04)
                axes[head_idx].grid(True, alpha=0.2)
            
            # Hide unused subplots
            for i in range(n_heads, len(axes)):
                axes[i].set_visible(False)
                
            plt.tight_layout()
            plt.savefig(save_dir / f'attention_heads_layer_{middle_layer}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: attention_heads_layer_{middle_layer}.png")
        
        # 3. Attention statistics summary
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Average attention magnitude by layer
        ax = axes[0, 0]
        layer_avg_attentions = []
        for layer_idx, layer_attn in enumerate(attention_weights):
            avg_attn = torch.mean(layer_attn).item()
            layer_avg_attentions.append(avg_attn)
        
        ax.plot(range(len(layer_avg_attentions)), layer_avg_attentions, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Average Attention Weight')
        ax.set_title('Attention Magnitude by Layer')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Attention entropy by layer (measure of attention spread)
        ax = axes[0, 1]
        layer_entropies = []
        for layer_idx, layer_attn in enumerate(attention_weights):
            layer_attn_np = layer_attn[0].cpu().numpy()  # Remove batch dim
            entropies = []
            for head_idx in range(layer_attn_np.shape[0]):
                head_attn = layer_attn_np[head_idx]
                # Calculate entropy for each query position
                for query_pos in range(head_attn.shape[0]):
                    attention_dist = head_attn[query_pos]
                    entropy = -np.sum(attention_dist * np.log(attention_dist + 1e-12))
                    entropies.append(entropy)
            layer_entropies.append(np.mean(entropies))
        
        ax.plot(range(len(layer_entropies)), layer_entropies, 's-', linewidth=2, markersize=6, color='orange')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Average Attention Entropy')
        ax.set_title('Attention Entropy by Layer\n(Higher = More Distributed)')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Self-attention strength by layer
        ax = axes[1, 0]
        self_attentions = []
        for layer_idx, layer_attn in enumerate(attention_weights):
            layer_attn_np = layer_attn[0].cpu().numpy()  # Remove batch dim
            diag_values = []
            for head_idx in range(layer_attn_np.shape[0]):
                head_attn = layer_attn_np[head_idx]
                diag_attn = np.mean(np.diag(head_attn))
                diag_values.append(diag_attn)
            self_attentions.append(np.mean(diag_values))
        
        ax.plot(range(len(self_attentions)), self_attentions, '^-', linewidth=2, markersize=6, color='red')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Average Self-Attention')
        ax.set_title('Self-Attention Strength by Layer')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Token-level attention analysis
        ax = axes[1, 1]
        if len(attention_weights) > 0:
            # Look at last layer attention and see which tokens receive most attention
            last_layer = attention_weights[-1][0]  # (num_heads, seq_len, seq_len)
            avg_received_attention = torch.mean(torch.mean(last_layer, dim=0), dim=0).cpu().numpy()  # Average across heads and queries
            
            # Show top tokens receiving attention
            top_indices = np.argsort(avg_received_attention)[-10:]  # Top 10
            top_attention = avg_received_attention[top_indices]
            top_tokens = [tokens[i] if i < len(tokens) else f'[{i}]' for i in top_indices]
            
            bars = ax.bar(range(len(top_attention)), top_attention, color='lightcoral', alpha=0.7)
            ax.set_xlabel('Top Attended Tokens')
            ax.set_ylabel('Average Attention Received')
            ax.set_title('Most Attended Tokens (Final Layer)')
            ax.set_xticks(range(len(top_tokens)))
            ax.set_xticklabels(top_tokens, rotation=45, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Annotate bars
            for bar, attn_val in zip(bars, top_attention):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{attn_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'attention_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: attention_statistics.png")
    
    def analyze_motif_attention(self, sequence: str, motifs: Dict[str, str], 
                              save_dir: str = "analysis/results/phase2") -> Dict:
        """Analyze how attention patterns respond to regulatory motifs"""
        
        print("=== ATTENTION PATTERN ANALYSIS ===")
        print(f"Analyzing sequence: {len(sequence)} bp")
        
        # Find motifs
        motif_positions = self.find_motifs(sequence, motifs)
        print(f"Found motifs: {motif_positions}")
        
        # Extract attention patterns  
        attention_weights, tokens, token_ids = self.extract_attention_patterns(sequence, max_length=80)
        
        if attention_weights is None:
            print("Failed to extract attention patterns - creating fallback analysis")
            return self.create_fallback_analysis(sequence, motifs, save_dir)
        
        print(f"Sequence tokenized to {len(tokens)} tokens")
        print(f"Extracted attention from {len(attention_weights)} layers")
        print(f"Attention tensor shapes: {[att.shape for att in attention_weights[:3]]}")
        
        # Create visualizations
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.create_simple_attention_visualizations(attention_weights, tokens, motif_positions, 
                                                   sequence, save_path)
        
        # Save summary
        with open(save_path / 'attention_analysis_summary.txt', 'w') as f:
            f.write("ATTENTION PATTERN ANALYSIS RESULTS\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Architecture: {self.num_layers} layers, {self.num_heads} heads per layer\n")
            f.write(f"Sequence length: {len(sequence)} bp\n")
            f.write(f"Tokenized length: {len(tokens)} tokens\n\n")
            
            f.write("TOKENS:\n")
            for i, token in enumerate(tokens[:20]):  # First 20 tokens
                f.write(f"  {i:2d}: {token}\n")
            if len(tokens) > 20:
                f.write(f"  ... and {len(tokens) - 20} more tokens\n")
            f.write("\n")
            
            f.write("MOTIF POSITIONS (in sequence):\n")
            for motif_name, positions in motif_positions.items():
                f.write(f"  {motif_name}: {positions}\n")
            f.write("\n")
            
            if attention_weights:
                f.write("ATTENTION ANALYSIS:\n")
                f.write(f"  Successfully extracted attention from {len(attention_weights)} layers\n")
                f.write(f"  Each layer has {attention_weights[0].shape[1]} attention heads\n")
                f.write(f"  Attention matrices are {attention_weights[0].shape[-2]} x {attention_weights[0].shape[-1]}\n")
                
                # Calculate some basic statistics
                all_attention_values = []
                for layer_attn in attention_weights:
                    all_attention_values.extend(layer_attn.flatten().cpu().numpy())
                
                f.write(f"  Attention value statistics:\n")
                f.write(f"    Mean: {np.mean(all_attention_values):.4f}\n")
                f.write(f"    Std: {np.std(all_attention_values):.4f}\n")
                f.write(f"    Min: {np.min(all_attention_values):.4f}\n")
                f.write(f"    Max: {np.max(all_attention_values):.4f}\n")
        
        print(f"Saved: attention_analysis_summary.txt")
        
        return {
            'attention_weights': attention_weights,
            'tokens': tokens, 
            'token_ids': token_ids,
            'motif_positions': motif_positions,
            'sequence': sequence
        }
    
    def create_fallback_analysis(self, sequence: str, motifs: Dict[str, str], save_dir: str) -> Dict:
        """Create fallback analysis when attention extraction fails"""
        
        print("Creating fallback analysis based on hidden states...")
        
        # At least extract hidden states for basic analysis
        inputs = self.tokenizer(sequence, return_tensors='pt', 
                               truncation=True, max_length=80).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create basic hidden state analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Hidden state norms by layer
        ax = axes[0, 0]
        layer_norms = []
        for layer_idx, hidden in enumerate(hidden_states):
            norm = torch.norm(hidden).item()
            layer_norms.append(norm)
        
        ax.plot(range(len(layer_norms)), layer_norms, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Hidden State L2 Norm')
        ax.set_title('Hidden State Magnitude by Layer')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Token-level hidden state analysis
        ax = axes[0, 1]
        final_hidden = hidden_states[-1][0].cpu().numpy()  # Remove batch dim
        token_norms = [np.linalg.norm(final_hidden[i]) for i in range(final_hidden.shape[0])]
        
        ax.plot(token_norms, 'o-', color='orange')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Hidden State Norm')
        ax.set_title('Final Layer Token Representations')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Layer similarity
        ax = axes[1, 0]
        similarities = []
        for i in range(1, len(hidden_states)):
            sim = torch.nn.functional.cosine_similarity(
                hidden_states[i-1].flatten(), hidden_states[i].flatten(), dim=0
            ).item()
            similarities.append(sim)
        
        ax.plot(range(1, len(hidden_states)), similarities, 's-', color='red', linewidth=2)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Cosine Similarity with Previous Layer')
        ax.set_title('Layer-to-Layer Similarity')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Motif positions
        ax = axes[1, 1]
        motif_positions = self.find_motifs(sequence, motifs)
        
        for i, (motif_name, positions) in enumerate(motif_positions.items()):
            if positions:
                y_pos = [i] * len(positions)
                ax.scatter(positions, y_pos, label=motif_name, s=50, alpha=0.7)
        
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Motif Type')
        ax.set_title('Regulatory Motif Positions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'fallback_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: fallback_analysis.png")
        
        # Save summary
        with open(save_path / 'attention_analysis_summary.txt', 'w') as f:
            f.write("ATTENTION PATTERN ANALYSIS RESULTS (FALLBACK MODE)\n")
            f.write("=" * 50 + "\n\n")
            f.write("Note: Attention extraction failed, providing hidden state analysis instead.\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Sequence length: {len(sequence)} bp\n")
            f.write(f"Tokenized length: {len(tokens)} tokens\n\n")
            f.write("MOTIF POSITIONS:\n")
            for motif_name, positions in motif_positions.items():
                f.write(f"  {motif_name}: {positions}\n")
        
        return {
            'hidden_states': hidden_states,
            'tokens': tokens,
            'motif_positions': motif_positions,
            'sequence': sequence,
            'fallback_mode': True
        }

def main():
    """Run attention pattern analysis"""
    print("=== PHASE 2 - TASK 2: ATTENTION PATTERN ANALYSIS ===")
    
    # Load AT1G01010 promoter sequence
    sequence_file = Path("data/test_sequences/arabidopsis_AT1G01010_real.txt")
    with open(sequence_file, 'r') as f:
        sequence = f.read().strip()
    
    # Focus on a shorter region for better visualization
    # Find TATA/CAAT region
    tata_pos = sequence.upper().find('TATAAA')
    if tata_pos != -1:
        # Extract 200bp region around TATA box
        start = max(0, tata_pos - 100)
        end = min(len(sequence), tata_pos + 100)
        sequence = sequence[start:end]
    else:
        # Fallback to first 200bp
        sequence = sequence[:200]
    
    print(f"Analyzing sequence: {len(sequence)} bp")
    print(f"Sequence preview: {sequence[:100]}...")
    
    # Define regulatory motifs to analyze
    motifs = {
        'TATA': 'TATAAA',
        'CAAT': 'CAAT',
        'GC_box': 'GGGCGG',
        'AT_rich': 'AAAA'
    }
    
    # Initialize attention analyzer
    analyzer = AttentionAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_motif_attention(sequence, motifs)
    
    print(f"\n=== ATTENTION PATTERN ANALYSIS COMPLETE ===")
    print(f"Results saved to analysis/results/phase2/")
    print(f"Generated files:")
    if not results.get('fallback_mode', False):
        print(f"  - attention_patterns_layers.png")
        print(f"  - attention_heads_layer_X.png")
        print(f"  - attention_statistics.png")
    else:
        print(f"  - fallback_analysis.png (attention extraction failed)")
    print(f"  - attention_analysis_summary.txt")
    
    # Quick summary
    print(f"\nQUICK SUMMARY:")
    print(f"  Sequence tokenized to: {len(results['tokens'])} tokens")
    print(f"  Motifs found: {results['motif_positions']}")
    if results.get('attention_weights'):
        print(f"  Attention matrices extracted from {len(results['attention_weights'])} layers")
    else:
        print(f"  Attention extraction failed - used hidden states instead")

if __name__ == "__main__":
    main()