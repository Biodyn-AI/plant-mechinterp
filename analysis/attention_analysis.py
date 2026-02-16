#!/usr/bin/env python3
"""
PHASE 2 - TASK 2: Attention Pattern Analysis
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
    
    def map_sequence_to_tokens(self, sequence: str, max_length: int = 128) -> Tuple[List[str], List[int], str]:
        """Map sequence positions to token positions"""
        inputs = self.tokenizer(sequence, return_tensors='pt', 
                               truncation=True, max_length=max_length)
        
        token_ids = inputs['input_ids'][0].numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # Get the actual processed sequence (might be truncated)
        processed_sequence = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        return tokens, token_ids, processed_sequence
    
    def extract_attention_patterns(self, sequence: str, max_length: int = 128) -> Tuple[torch.Tensor, List[str], List[int]]:
        """Extract attention patterns for a sequence"""
        inputs = self.tokenizer(sequence, return_tensors='pt', 
                               truncation=True, max_length=max_length).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        # outputs.attentions is a tuple of (batch_size, num_heads, seq_len, seq_len)
        attention_weights = outputs.attentions
        tokens, token_ids, processed_sequence = self.map_sequence_to_tokens(sequence, max_length)
        
        return attention_weights, tokens, token_ids
    
    def analyze_attention_head_specialization(self, attention_weights: torch.Tensor, tokens: List[str]) -> Dict:
        """Analyze how different attention heads specialize"""
        
        # attention_weights is tuple of tensors: (layer, batch, head, seq_len, seq_len)
        results = {
            'layer_head_patterns': [],
            'head_specialization': {},
            'attention_statistics': {}
        }
        
        seq_len = attention_weights[0].shape[-1]
        
        for layer_idx, layer_attention in enumerate(attention_weights):
            # layer_attention shape: (batch=1, num_heads, seq_len, seq_len)
            layer_attention = layer_attention[0]  # Remove batch dimension
            
            layer_results = []
            
            for head_idx in range(self.num_heads):
                head_attention = layer_attention[head_idx]  # Shape: (seq_len, seq_len)
                
                # Analyze this head's attention pattern
                head_analysis = self.analyze_single_head(head_attention, tokens, layer_idx, head_idx)
                layer_results.append(head_analysis)
            
            results['layer_head_patterns'].append(layer_results)
        
        return results
    
    def analyze_single_head(self, attention_matrix: torch.Tensor, tokens: List[str], 
                           layer_idx: int, head_idx: int) -> Dict:
        """Analyze a single attention head's behavior"""
        
        # Convert to numpy for analysis
        attn_np = attention_matrix.cpu().numpy()
        
        # Calculate attention statistics
        max_attention_per_token = np.max(attn_np, axis=1)  # What each token attends to most
        attention_entropy = -np.sum(attn_np * np.log(attn_np + 1e-12), axis=1)  # Attention spread
        
        # Find dominant attention patterns
        avg_attention = np.mean(attn_np, axis=0)  # Average attention received by each position
        
        # Identify attention type (local vs global vs specific patterns)
        diagonal_attention = np.mean(np.diag(attn_np))  # Self-attention
        
        # Local vs global attention
        local_window = 5
        local_attention = 0
        for i in range(len(attn_np)):
            for j in range(max(0, i-local_window), min(len(attn_np), i+local_window+1)):
                local_attention += attn_np[i, j]
        local_attention /= (len(attn_np) * len(attn_np))
        
        global_attention = np.sum(attn_np) - local_attention * len(attn_np) * len(attn_np)
        global_attention /= (len(attn_np) * len(attn_np))
        
        return {
            'layer': layer_idx,
            'head': head_idx,
            'attention_matrix': attn_np,
            'max_attention_per_token': max_attention_per_token,
            'attention_entropy': attention_entropy,
            'diagonal_attention': diagonal_attention,
            'local_attention_ratio': local_attention,
            'global_attention_ratio': global_attention,
            'avg_attention_received': avg_attention
        }
    
    def analyze_motif_attention(self, sequence: str, motifs: Dict[str, str], 
                              save_dir: str = "analysis/results/phase2") -> Dict:
        """Analyze how attention patterns respond to regulatory motifs"""
        
        print("=== ATTENTION PATTERN ANALYSIS ===")
        print(f"Analyzing sequence: {len(sequence)} bp")
        
        # Find motifs
        motif_positions = self.find_motifs(sequence, motifs)
        print(f"Found motifs: {motif_positions}")
        
        # Extract attention patterns
        attention_weights, tokens, token_ids = self.extract_attention_patterns(sequence)
        
        print(f"Sequence tokenized to {len(tokens)} tokens")
        print(f"Attention shape: {[att.shape for att in attention_weights[:3]]}...")  # Show first 3 layers
        
        # Analyze attention head specialization
        head_analysis = self.analyze_attention_head_specialization(attention_weights, tokens)
        
        # Create visualizations
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.create_attention_visualizations(attention_weights, tokens, motif_positions, 
                                           sequence, head_analysis, save_path)
        
        return {
            'attention_weights': attention_weights,
            'tokens': tokens, 
            'token_ids': token_ids,
            'motif_positions': motif_positions,
            'head_analysis': head_analysis,
            'sequence': sequence
        }
    
    def create_attention_visualizations(self, attention_weights: List[torch.Tensor], tokens: List[str],
                                      motif_positions: Dict[str, List[int]], sequence: str,
                                      head_analysis: Dict, save_dir: Path):
        """Create comprehensive attention visualizations"""
        
        # 1. Overall attention pattern heatmap for selected layers
        selected_layers = [0, 3, 6, 9, 11]  # Sample across depth
        fig, axes = plt.subplots(len(selected_layers), 1, figsize=(15, 3 * len(selected_layers)))
        
        if len(selected_layers) == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(selected_layers):
            if layer_idx < len(attention_weights):
                # Average attention across all heads in this layer
                layer_attn = attention_weights[layer_idx][0]  # Remove batch dim
                avg_attention = torch.mean(layer_attn, dim=0).cpu().numpy()  # Average across heads
                
                im = axes[i].imshow(avg_attention, cmap='Blues', aspect='auto')
                axes[i].set_title(f'Layer {layer_idx} - Average Attention Pattern', fontweight='bold')
                axes[i].set_xlabel('Key Position')
                axes[i].set_ylabel('Query Position')
                
                # Add token labels (sample every few tokens for readability)
                step = max(1, len(tokens) // 10)
                tick_positions = list(range(0, len(tokens), step))
                tick_labels = [tokens[j] if j < len(tokens) else '' for j in tick_positions]
                axes[i].set_xticks(tick_positions)
                axes[i].set_xticklabels(tick_labels, rotation=45, fontsize=8)
                axes[i].set_yticks(tick_positions)
                axes[i].set_yticklabels(tick_labels, fontsize=8)
                
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'attention_patterns_by_layer.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: attention_patterns_by_layer.png")
        
        # 2. Attention head specialization analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Attention entropy by head (measure of attention spread)
        ax = axes[0, 0]
        layer_head_entropies = []
        
        for layer_idx, layer_heads in enumerate(head_analysis['layer_head_patterns']):
            for head_data in layer_heads:
                avg_entropy = np.mean(head_data['attention_entropy'])
                layer_head_entropies.append({
                    'layer': layer_idx,
                    'head': head_data['head'],
                    'entropy': avg_entropy
                })
        
        entropies = [h['entropy'] for h in layer_head_entropies]
        layers = [h['layer'] for h in layer_head_entropies]
        
        scatter = ax.scatter(layers, entropies, alpha=0.6, s=30)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Average Attention Entropy')
        ax.set_title('Attention Head Specialization\n(Higher entropy = more distributed attention)')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Local vs Global attention ratios
        ax = axes[0, 1]
        local_ratios = [h['local_attention_ratio'] for layer_heads in head_analysis['layer_head_patterns'] for h in layer_heads]
        global_ratios = [h['global_attention_ratio'] for layer_heads in head_analysis['layer_head_patterns'] for h in layer_heads]
        
        ax.scatter(local_ratios, global_ratios, alpha=0.6)
        ax.set_xlabel('Local Attention Ratio')
        ax.set_ylabel('Global Attention Ratio')
        ax.set_title('Local vs Global Attention Patterns')
        ax.grid(True, alpha=0.3)
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        
        # Plot 3: Self-attention (diagonal) by layer and head
        ax = axes[1, 0]
        diagonal_attentions = []
        layer_labels = []
        
        for layer_idx, layer_heads in enumerate(head_analysis['layer_head_patterns']):
            for head_data in layer_heads:
                diagonal_attentions.append(head_data['diagonal_attention'])
                layer_labels.append(f"L{layer_idx}H{head_data['head']}")
        
        # Group by layer for boxplot
        layer_diagonals = []
        layer_nums = []
        for layer_idx, layer_heads in enumerate(head_analysis['layer_head_patterns']):
            layer_vals = [h['diagonal_attention'] for h in layer_heads]
            layer_diagonals.extend(layer_vals)
            layer_nums.extend([layer_idx] * len(layer_vals))
        
        # Create boxplot by layer
        layers_unique = sorted(set(layer_nums))
        layer_data = [[] for _ in layers_unique]
        for layer_num, diag_val in zip(layer_nums, layer_diagonals):
            layer_data[layer_num].append(diag_val)
        
        ax.boxplot(layer_data, labels=[f'L{i}' for i in layers_unique])
        ax.set_xlabel('Layer')
        ax.set_ylabel('Self-Attention Strength')
        ax.set_title('Self-Attention by Layer')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Attention pattern diversity
        ax = axes[1, 1]
        
        # For each layer, show the diversity of attention patterns across heads
        layer_diversities = []
        for layer_idx, layer_heads in enumerate(head_analysis['layer_head_patterns']):
            # Calculate diversity as std of attention patterns
            attention_matrices = [h['attention_matrix'] for h in layer_heads]
            if attention_matrices:
                # Flatten each attention matrix and compute pairwise differences
                flattened = [matrix.flatten() for matrix in attention_matrices]
                diversities = []
                for i in range(len(flattened)):
                    for j in range(i+1, len(flattened)):
                        diff = np.linalg.norm(flattened[i] - flattened[j])
                        diversities.append(diff)
                avg_diversity = np.mean(diversities) if diversities else 0
                layer_diversities.append(avg_diversity)
            else:
                layer_diversities.append(0)
        
        ax.plot(range(len(layer_diversities)), layer_diversities, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Average Head Pattern Diversity')
        ax.set_title('Attention Head Diversity by Layer')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'attention_head_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: attention_head_analysis.png")
        
        # 3. Individual attention head samples (most interesting heads)
        # Find heads with extreme characteristics
        most_local_head = None
        most_global_head = None
        highest_entropy_head = None
        
        max_local = 0
        max_global = 0
        max_entropy = 0
        
        for layer_heads in head_analysis['layer_head_patterns']:
            for head_data in layer_heads:
                if head_data['local_attention_ratio'] > max_local:
                    max_local = head_data['local_attention_ratio']
                    most_local_head = head_data
                    
                if head_data['global_attention_ratio'] > max_global:
                    max_global = head_data['global_attention_ratio']  
                    most_global_head = head_data
                    
                avg_entropy = np.mean(head_data['attention_entropy'])
                if avg_entropy > max_entropy:
                    max_entropy = avg_entropy
                    highest_entropy_head = head_data
        
        # Plot these interesting heads
        interesting_heads = [
            (most_local_head, "Most Local Attention"),
            (most_global_head, "Most Global Attention"), 
            (highest_entropy_head, "Highest Entropy (Most Distributed)")
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, (head_data, title) in enumerate(interesting_heads):
            if head_data:
                im = axes[i].imshow(head_data['attention_matrix'], cmap='Blues', aspect='auto')
                axes[i].set_title(f"{title}\nLayer {head_data['layer']}, Head {head_data['head']}")
                axes[i].set_xlabel('Key Position')
                axes[i].set_ylabel('Query Position')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'interesting_attention_heads.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: interesting_attention_heads.png")
        
        # 4. Save summary analysis
        with open(save_dir / 'attention_analysis_summary.txt', 'w') as f:
            f.write("ATTENTION PATTERN ANALYSIS RESULTS\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Architecture: {self.num_layers} layers, {self.num_heads} heads per layer\n")
            f.write(f"Sequence length: {len(sequence)} bp\n")
            f.write(f"Tokenized length: {len(tokens)} tokens\n\n")
            
            f.write("MOTIF POSITIONS:\n")
            for motif_name, positions in motif_positions.items():
                f.write(f"  {motif_name}: {positions}\n")
            f.write("\n")
            
            f.write("ATTENTION HEAD SPECIALIZATION ANALYSIS:\n")
            
            # Most specialized heads
            if most_local_head:
                f.write(f"Most local attention head: Layer {most_local_head['layer']}, Head {most_local_head['head']}\n")
                f.write(f"  Local attention ratio: {most_local_head['local_attention_ratio']:.3f}\n")
                
            if most_global_head:
                f.write(f"Most global attention head: Layer {most_global_head['layer']}, Head {most_global_head['head']}\n")
                f.write(f"  Global attention ratio: {most_global_head['global_attention_ratio']:.3f}\n")
                
            if highest_entropy_head:
                f.write(f"Most distributed attention head: Layer {highest_entropy_head['layer']}, Head {highest_entropy_head['head']}\n")
                f.write(f"  Average entropy: {np.mean(highest_entropy_head['attention_entropy']):.3f}\n")
            
            f.write(f"\nLAYER-WISE ANALYSIS:\n")
            for layer_idx, layer_heads in enumerate(head_analysis['layer_head_patterns']):
                local_ratios = [h['local_attention_ratio'] for h in layer_heads]
                diagonal_attns = [h['diagonal_attention'] for h in layer_heads]
                entropies = [np.mean(h['attention_entropy']) for h in layer_heads]
                
                f.write(f"Layer {layer_idx}:\n")
                f.write(f"  Avg local attention: {np.mean(local_ratios):.3f}\n")
                f.write(f"  Avg self-attention: {np.mean(diagonal_attns):.3f}\n")
                f.write(f"  Avg entropy: {np.mean(entropies):.3f}\n")
        
        print(f"Saved: attention_analysis_summary.txt")

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
        # Extract 300bp region around TATA box
        start = max(0, tata_pos - 150)
        end = min(len(sequence), tata_pos + 150)
        sequence = sequence[start:end]
    
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
    print(f"  - attention_patterns_by_layer.png")
    print(f"  - attention_head_analysis.png")
    print(f"  - interesting_attention_heads.png")
    print(f"  - attention_analysis_summary.txt")
    
    # Quick summary
    head_analysis = results['head_analysis']
    print(f"\nQUICK SUMMARY:")
    print(f"  Sequence tokenized to: {len(results['tokens'])} tokens")
    print(f"  Attention matrices extracted from {len(results['attention_weights'])} layers")
    print(f"  {len(results['motif_positions'])} motif types found")

if __name__ == "__main__":
    main()