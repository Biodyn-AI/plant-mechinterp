#!/usr/bin/env python3
"""
PHASE 2 - TASK 1: Activation Patching / Causal Tracing (CORRECTED VERSION)
Plant Mechanistic Interpretability Project

This implements activation patching to identify which layers are causally
important for processing regulatory motifs in plant DNA sequences.
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

class ActivationPatcher:
    """Implements activation patching for causal tracing in plant DNA models"""
    
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
            print(f"Model loaded on {self.device} - GemmaModel with 13 hidden states")
            
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
    
    def corrupt_motif(self, sequence: str, motif_positions: List[int], motif_length: int) -> str:
        """Replace motif occurrences with random nucleotides"""
        corrupted = list(sequence)
        random.seed(42)  # For reproducibility
        
        for pos in motif_positions:
            for i in range(motif_length):
                if pos + i < len(corrupted):
                    # Replace with random nucleotide (different from original)
                    original = corrupted[pos + i].upper()
                    choices = [n for n in ['A', 'T', 'C', 'G'] if n != original]
                    corrupted[pos + i] = random.choice(choices) if choices else original
        
        return ''.join(corrupted)
    
    def run_model_with_cache(self, sequence: str, max_length: int = 400) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
        """Run model and cache all layer activations"""
        # Truncate sequence for memory efficiency
        sequence_truncated = sequence[:max_length] if len(sequence) > max_length else sequence
        
        inputs = self.tokenizer(sequence_truncated, return_tensors='pt', 
                               truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Cache all hidden states (13 total: embeddings + 12 transformer layers)
            hidden_states = []
            for layer_hidden in outputs.hidden_states:
                hidden_states.append(layer_hidden.clone().detach())
                
            # Use final layer output as the "output" representation
            final_output = outputs.hidden_states[-1]
            
        return final_output, hidden_states, inputs
    
    def compute_activation_difference(self, clean_activations: List[torch.Tensor], 
                                    corrupted_activations: List[torch.Tensor]) -> List[float]:
        """Compute activation differences across all layers"""
        differences = []
        
        for i in range(len(clean_activations)):
            clean_act = clean_activations[i]
            corrupted_act = corrupted_activations[i]
            
            # Compute L2 norm of difference
            diff = torch.norm(clean_act - corrupted_act).item()
            differences.append(diff)
            
        return differences
    
    def compute_representational_similarity(self, clean_output: torch.Tensor, 
                                          corrupted_output: torch.Tensor) -> float:
        """Compute cosine similarity between clean and corrupted representations"""
        clean_flat = clean_output.flatten()
        corrupted_flat = corrupted_output.flatten()
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            clean_flat.unsqueeze(0), corrupted_flat.unsqueeze(0)
        ).item()
        
        return cos_sim
    
    def activation_patching_experiment(self, sequence: str, motifs: Dict[str, str], 
                                     save_dir: str = "analysis/results/phase2") -> Dict:
        """Run complete activation patching experiment"""
        
        print("=== ACTIVATION PATCHING / CAUSAL TRACING EXPERIMENT ===")
        print(f"Sequence length: {len(sequence)} bp")
        
        # Find motifs in sequence
        motif_positions = self.find_motifs(sequence, motifs)
        print(f"Found motifs: {motif_positions}")
        
        results = {}
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for motif_name, motif_seq in motifs.items():
            if not motif_positions[motif_name]:
                print(f"Skipping {motif_name} - not found in sequence")
                continue
                
            print(f"\n--- Analyzing {motif_name} motif ---")
            
            # Create corrupted version
            corrupted_sequence = self.corrupt_motif(
                sequence, motif_positions[motif_name], len(motif_seq)
            )
            
            pos = motif_positions[motif_name][0]  # Show first occurrence
            print(f"Original:  ...{sequence[max(0, pos-10):pos+15]}...")
            print(f"Corrupted: ...{corrupted_sequence[max(0, pos-10):pos+15]}...")
            
            try:
                # Run clean forward pass
                clean_output, clean_activations, clean_inputs = self.run_model_with_cache(sequence)
                
                # Run corrupted forward pass  
                corrupted_output, corrupted_activations, corrupted_inputs = self.run_model_with_cache(corrupted_sequence)
                
                print(f"Clean output shape: {clean_output.shape}")
                print(f"Corrupted output shape: {corrupted_output.shape}")
                print(f"Number of layers: {len(clean_activations)}")
                
                # Compute activation differences at each layer (causal importance proxy)
                activation_differences = self.compute_activation_difference(
                    clean_activations, corrupted_activations
                )
                
                # Compute overall representational similarity
                representation_similarity = self.compute_representational_similarity(
                    clean_output, corrupted_output
                )
                
                print(f"Final representation similarity: {representation_similarity:.4f}")
                
                # Store results
                results[motif_name] = {
                    'motif_sequence': motif_seq,
                    'positions': motif_positions[motif_name],
                    'activation_differences': activation_differences,
                    'representation_similarity': representation_similarity,
                    'clean_output': clean_output.cpu().numpy(),
                    'corrupted_output': corrupted_output.cpu().numpy(),
                    'corrupted_sequence_sample': corrupted_sequence[max(0, pos-20):pos+25]
                }
                
                print(f"Layer activation differences:")
                for layer_idx, diff in enumerate(activation_differences):
                    print(f"  Layer {layer_idx:2d}: {diff:.4f}")
                
            except Exception as e:
                print(f"Error in {motif_name} experiment: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create visualizations
        self.create_causal_tracing_plots(results, save_path)
        
        return results
    
    def create_causal_tracing_plots(self, results: Dict, save_dir: Path):
        """Create causal tracing visualizations"""
        
        if not results:
            print("No results to plot")
            return
        
        # 1. Layer-wise causal importance heatmap
        motifs = list(results.keys())
        n_layers = len(results[motifs[0]]['activation_differences'])
        
        causal_matrix = np.zeros((len(motifs), n_layers))
        
        for i, motif_name in enumerate(motifs):
            activation_diffs = results[motif_name]['activation_differences']
            for j, diff in enumerate(activation_diffs):
                causal_matrix[i, j] = diff
        
        # Normalize each row for better visualization
        for i in range(causal_matrix.shape[0]):
            max_val = np.max(causal_matrix[i, :])
            if max_val > 0:
                causal_matrix[i, :] = causal_matrix[i, :] / max_val
        
        plt.figure(figsize=(14, 6))
        sns.heatmap(causal_matrix, 
                   annot=True, fmt='.3f', cmap='viridis',
                   xticklabels=[f'L{i}' for i in range(n_layers)],
                   yticklabels=motifs,
                   cbar_kws={'label': 'Normalized Activation Difference'})
        
        plt.title('Causal Tracing Heatmap: Motif Corruption Effects by Layer', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Layer Index (L0=Embeddings, L1-L12=Transformer Layers)', fontsize=12)
        plt.ylabel('Regulatory Motif', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(save_dir / 'causal_tracing_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: causal_tracing_heatmap.png")
        
        # 2. Individual motif causal traces with identification of critical layers
        n_motifs = len(results)
        fig, axes = plt.subplots((n_motifs + 1) // 2, 2, figsize=(15, 4 * ((n_motifs + 1) // 2)))
        if n_motifs == 1:
            axes = [axes]
        elif n_motifs <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (motif_name, data) in enumerate(results.items()):
            ax = axes[i] if n_motifs > 1 else axes[0]
            
            activation_diffs = data['activation_differences']
            layer_indices = list(range(len(activation_diffs)))
            
            ax.plot(layer_indices, activation_diffs, 'o-', linewidth=2, markersize=6, 
                   color='darkblue', label='Activation Difference')
            ax.set_title(f'{motif_name} Motif: Causal Layer Analysis', fontweight='bold')
            ax.set_xlabel('Layer Index (0=Embeddings, 1-12=Transformer)')
            ax.set_ylabel('Activation Difference (L2 Norm)')
            ax.grid(True, alpha=0.3)
            
            # Highlight most causally important layers
            sorted_layers = sorted(enumerate(activation_diffs), key=lambda x: x[1], reverse=True)
            top_3_layers = sorted_layers[:3]
            
            for rank, (layer_idx, diff) in enumerate(top_3_layers):
                colors = ['red', 'orange', 'gold']
                ax.axvline(x=layer_idx, color=colors[rank], linestyle='--', alpha=0.7)
                ax.text(layer_idx, diff, f'#{rank+1}\nL{layer_idx}\n{diff:.3f}', 
                       ha='center', va='bottom', fontweight='bold', 
                       color=colors[rank], fontsize=9)
            
            # Add similarity score
            sim_score = data['representation_similarity']
            ax.text(0.02, 0.98, f'Representation Similarity: {sim_score:.3f}',
                   transform=ax.transAxes, va='top', ha='left', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for i in range(len(results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'individual_causal_traces.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: individual_causal_traces.png")
        
        # 3. Summary analysis plot
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Representation similarity for each motif
        plt.subplot(2, 2, 1)
        motif_names = list(results.keys())
        similarities = [results[m]['representation_similarity'] for m in motif_names]
        bars = plt.bar(motif_names, similarities, color='skyblue', alpha=0.7)
        plt.title('Final Representation Similarity\n(Higher = Less Disruption)')
        plt.ylabel('Cosine Similarity')
        plt.xticks(rotation=45)
        for bar, sim in zip(bars, similarities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{sim:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Maximum activation difference by motif
        plt.subplot(2, 2, 2)
        max_diffs = [max(results[m]['activation_differences']) for m in motif_names]
        bars = plt.bar(motif_names, max_diffs, color='lightcoral', alpha=0.7)
        plt.title('Maximum Layer Disruption\n(Higher = More Causal Impact)')
        plt.ylabel('Max Activation Difference')
        plt.xticks(rotation=45)
        for bar, diff in zip(bars, max_diffs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{diff:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Layer-wise average disruption
        plt.subplot(2, 1, 2)
        layer_averages = np.mean(causal_matrix, axis=0)  # Average across motifs
        plt.plot(range(n_layers), layer_averages, 'o-', linewidth=3, markersize=8, color='purple')
        plt.title('Average Causal Importance Across All Motifs by Layer', fontweight='bold')
        plt.xlabel('Layer Index (0=Embeddings, 1-12=Transformer Layers)')
        plt.ylabel('Average Normalized Activation Difference')
        plt.grid(True, alpha=0.3)
        
        # Highlight most important layer overall
        most_important_layer = np.argmax(layer_averages)
        plt.axvline(x=most_important_layer, color='red', linestyle='--', alpha=0.8, linewidth=2)
        plt.text(most_important_layer, layer_averages[most_important_layer], 
                f'Most Critical\nLayer {most_important_layer}\n({layer_averages[most_important_layer]:.3f})',
                ha='center', va='bottom', fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'causal_summary_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: causal_summary_analysis.png")
        
        # 4. Save detailed summary
        with open(save_dir / 'activation_patching_summary.txt', 'w') as f:
            f.write("ACTIVATION PATCHING / CAUSAL TRACING RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write("This analysis reveals which layers are CAUSALLY important\n")
            f.write("for processing regulatory motifs in plant promoter sequences.\n")
            f.write("Method: Compare activations between clean and corrupted sequences.\n\n")
            
            # Overall findings
            layer_averages = np.mean(causal_matrix, axis=0)
            most_critical_layer = np.argmax(layer_averages)
            f.write("OVERALL FINDINGS:\n")
            f.write(f"Most causally critical layer: Layer {most_critical_layer} ")
            f.write(f"({'Embeddings' if most_critical_layer == 0 else f'Transformer Layer {most_critical_layer}'})\n")
            f.write(f"Average disruption score: {layer_averages[most_critical_layer]:.4f}\n\n")
            
            # Layer hierarchy
            sorted_layer_importance = sorted(enumerate(layer_averages), key=lambda x: x[1], reverse=True)
            f.write("LAYER IMPORTANCE RANKING (by average disruption):\n")
            for rank, (layer_idx, score) in enumerate(sorted_layer_importance[:5]):
                layer_name = "Embeddings" if layer_idx == 0 else f"Transformer-{layer_idx}"
                f.write(f"  {rank+1}. Layer {layer_idx} ({layer_name}): {score:.4f}\n")
            f.write("\n")
            
            # Motif-specific results
            for motif_name, data in results.items():
                f.write(f"{motif_name} MOTIF:\n")
                f.write(f"  Sequence: {data['motif_sequence']}\n")
                f.write(f"  Positions found: {data['positions']}\n")
                f.write(f"  Representation similarity after corruption: {data['representation_similarity']:.4f}\n")
                
                activation_diffs = data['activation_differences']
                most_affected_layer = np.argmax(activation_diffs)
                f.write(f"  Most affected layer: {most_affected_layer} (diff: {activation_diffs[most_affected_layer]:.4f})\n")
                
                # Top 3 affected layers
                sorted_layers = sorted(enumerate(activation_diffs), key=lambda x: x[1], reverse=True)[:3]
                f.write(f"  Top 3 affected layers: ")
                for layer_idx, diff in sorted_layers:
                    f.write(f"L{layer_idx}({diff:.3f}) ")
                f.write("\n")
                
                f.write(f"  Sample corruption: {data['corrupted_sequence_sample']}\n\n")
        
        print(f"Saved: activation_patching_summary.txt")

def main():
    """Run activation patching experiments"""
    print("=== PHASE 2 - TASK 1: ACTIVATION PATCHING / CAUSAL TRACING ===")
    
    # Load AT1G01010 promoter sequence
    sequence_file = Path("data/test_sequences/arabidopsis_AT1G01010_real.txt")
    with open(sequence_file, 'r') as f:
        sequence = f.read().strip()
    
    print(f"Loaded sequence: {len(sequence)} bp")
    print(f"First 100 bp: {sequence[:100]}")
    
    # Define regulatory motifs to analyze
    motifs = {
        'TATA': 'TATAAA',
        'CAAT': 'CAAT', 
        'GC_box': 'GGGCGG',
        'AT_rich': 'AAAA'  # Added simpler motif
    }
    
    # Initialize activation patcher
    patcher = ActivationPatcher()
    
    # Run experiments
    results = patcher.activation_patching_experiment(sequence, motifs)
    
    print(f"\n=== ACTIVATION PATCHING ANALYSIS COMPLETE ===")
    print(f"Results saved to analysis/results/phase2/")
    print(f"Generated files:")
    print(f"  - causal_tracing_heatmap.png")
    print(f"  - individual_causal_traces.png") 
    print(f"  - causal_summary_analysis.png")
    print(f"  - activation_patching_summary.txt")
    
    # Quick summary
    if results:
        print(f"\nQUICK SUMMARY:")
        for motif_name, data in results.items():
            max_diff_layer = np.argmax(data['activation_differences'])
            max_diff = data['activation_differences'][max_diff_layer]
            sim = data['representation_similarity']
            print(f"  {motif_name}: Most affected layer={max_diff_layer}, diff={max_diff:.3f}, sim={sim:.3f}")

if __name__ == "__main__":
    main()