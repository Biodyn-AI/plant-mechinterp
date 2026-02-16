#!/usr/bin/env python3
"""
PHASE 2 - TASK 3: Probing at Scale
Plant Mechanistic Interpretability Project

Train linear probes at each layer to classify sequence types (promoters vs introns vs exons vs intergenic).
This reveals the model's representational hierarchy - where does the model "know" what type of sequence it's looking at?
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
import json
import random
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle

class SequenceProber:
    """Implements large-scale probing analysis for plant genomic sequences"""
    
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
            self.hidden_size = config.hidden_size
            
            print(f"Model loaded on {self.device}")
            print(f"Architecture: {self.num_layers} layers, {self.hidden_size}d hidden size")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def create_synthetic_sequences(self, n_per_type: int = 30) -> Dict[str, List[str]]:
        """Create synthetic sequences of different types for probing"""
        
        print(f"Generating {n_per_type} synthetic sequences per type...")
        
        sequences = {
            'promoter': [],
            'exon': [],
            'intron': [], 
            'intergenic': []
        }
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Define sequence characteristics
        base_length = 200
        
        for i in range(n_per_type):
            
            # PROMOTER sequences - high AT content, regulatory motifs
            promoter_seq = self.generate_promoter_like_sequence(base_length)
            sequences['promoter'].append(promoter_seq)
            
            # EXON sequences - balanced composition, no stop codons in frame
            exon_seq = self.generate_exon_like_sequence(base_length)
            sequences['exon'].append(exon_seq)
            
            # INTRON sequences - AT-rich, splicing signals
            intron_seq = self.generate_intron_like_sequence(base_length)
            sequences['intron'].append(intron_seq)
            
            # INTERGENIC sequences - more random, repetitive elements
            intergenic_seq = self.generate_intergenic_like_sequence(base_length)
            sequences['intergenic'].append(intergenic_seq)
        
        return sequences
    
    def generate_promoter_like_sequence(self, length: int) -> str:
        """Generate a promoter-like sequence with regulatory motifs"""
        
        # Start with AT-rich background (60% AT)
        seq = ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=length, 
                                     p=[0.3, 0.3, 0.2, 0.2]))
        
        seq = list(seq)
        
        # Add TATA box around position 25-35
        if length > 40:
            tata_pos = random.randint(20, 35)
            seq[tata_pos:tata_pos+6] = list('TATAAA')
        
        # Add CAAT box around position 60-80
        if length > 90:
            caat_pos = random.randint(60, 80)
            seq[caat_pos:caat_pos+4] = list('CAAT')
        
        # Add GC box occasionally
        if length > 120 and random.random() < 0.3:
            gc_pos = random.randint(100, 120)
            seq[gc_pos:gc_pos+6] = list('GGGCGG')
            
        # Add some CpG dinucleotides (higher in promoters)
        for _ in range(random.randint(3, 8)):
            pos = random.randint(0, length-2)
            if pos < len(seq) - 1:
                seq[pos:pos+2] = ['C', 'G']
        
        return ''.join(seq)
    
    def generate_exon_like_sequence(self, length: int) -> str:
        """Generate an exon-like sequence with balanced composition"""
        
        # More balanced nucleotide composition (closer to coding sequences)
        seq = ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=length, 
                                     p=[0.25, 0.25, 0.25, 0.25]))
        
        seq = list(seq)
        
        # Avoid in-frame stop codons (TAA, TAG, TGA)
        for i in range(0, len(seq)-2, 3):  # Check every codon
            codon = ''.join(seq[i:i+3])
            if codon in ['TAA', 'TAG', 'TGA']:
                # Replace with a random non-stop codon
                seq[i:i+3] = list(random.choice(['AAA', 'CCC', 'GGG', 'TTT']))
        
        # Add some start codons occasionally
        if length > 20 and random.random() < 0.2:
            start_pos = random.randint(0, 10)
            if start_pos + 3 <= len(seq):
                seq[start_pos:start_pos+3] = list('ATG')
        
        return ''.join(seq)
    
    def generate_intron_like_sequence(self, length: int) -> str:
        """Generate an intron-like sequence with splicing signals"""
        
        # AT-rich like introns (65% AT)
        seq = ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=length, 
                                     p=[0.325, 0.325, 0.175, 0.175]))
        
        seq = list(seq)
        
        # Add 5' splice site (GT) near start
        if length > 10:
            seq[2:4] = list('GT')
        
        # Add 3' splice site (AG) near end  
        if length > 20:
            seq[-4:-2] = list('AG')
        
        # Add branch point (A-rich region) in middle
        if length > 50:
            branch_pos = length // 2
            for i in range(5):
                if branch_pos + i < len(seq):
                    seq[branch_pos + i] = 'A'
        
        return ''.join(seq)
    
    def generate_intergenic_like_sequence(self, length: int) -> str:
        """Generate an intergenic-like sequence (more random)"""
        
        # More random composition
        seq = ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=length, 
                                     p=[0.3, 0.3, 0.2, 0.2]))
        
        seq = list(seq)
        
        # Add some repetitive elements
        if length > 40 and random.random() < 0.4:
            # Simple repeat
            repeat_unit = random.choice(['AT', 'TA', 'CA', 'TG'])
            repeat_pos = random.randint(10, length-20)
            repeat_len = min(10, length - repeat_pos)
            for i in range(repeat_len):
                seq[repeat_pos + i] = repeat_unit[i % len(repeat_unit)]
        
        return ''.join(seq)
    
    def get_arabidopsis_sequences_ensembl(self, n_per_type: int = 25) -> Dict[str, List[str]]:
        """Download real Arabidopsis sequences from Ensembl Plants REST API"""
        
        print(f"Attempting to download {n_per_type} real Arabidopsis sequences per type from Ensembl...")
        
        sequences = {
            'promoter': [],
            'exon': [],
            'intron': [],
            'intergenic': []
        }
        
        try:
            # Ensembl Plants REST API endpoints
            server = "https://rest.ensembl.org"
            
            # Get some Arabidopsis gene IDs first
            gene_endpoint = f"{server}/lookup/genome/arabidopsis_thaliana"
            
            # For now, use a list of known Arabidopsis gene IDs
            known_genes = [
                'AT1G01010', 'AT1G01020', 'AT1G01030', 'AT1G01040', 'AT1G01050',
                'AT1G01060', 'AT1G01070', 'AT1G01080', 'AT1G01090', 'AT1G01100',
                'AT2G01010', 'AT2G01020', 'AT2G01030', 'AT3G01010', 'AT3G01020',
                'AT4G01010', 'AT4G01020', 'AT5G01010', 'AT5G01020', 'AT5G01030'
            ]
            
            # Try to get sequences for these genes
            for gene_id in known_genes[:n_per_type]:
                try:
                    # Get gene info
                    gene_url = f"{server}/lookup/id/{gene_id}?species=arabidopsis_thaliana;expand=1"
                    gene_response = requests.get(gene_url, headers={"Content-Type": "application/json"})
                    
                    if gene_response.status_code == 200:
                        gene_data = gene_response.json()
                        
                        # Get promoter region (1kb upstream)
                        if gene_data.get('start') and gene_data.get('seq_region_name'):
                            promoter_start = max(1, gene_data['start'] - 1000)
                            promoter_end = gene_data['start'] - 1
                            
                            promoter_url = f"{server}/sequence/region/arabidopsis_thaliana/{gene_data['seq_region_name']}:{promoter_start}..{promoter_end}:1"
                            prom_response = requests.get(promoter_url, headers={"Content-Type": "text/plain"})
                            
                            if prom_response.status_code == 200 and len(prom_response.text) > 100:
                                sequences['promoter'].append(prom_response.text.strip().upper())
                        
                        # Get exon sequences from transcripts
                        if 'Transcript' in gene_data:
                            for transcript in gene_data['Transcript'][:1]:  # First transcript
                                if 'Exon' in transcript:
                                    for exon in transcript['Exon'][:2]:  # First 2 exons
                                        exon_url = f"{server}/sequence/id/{exon['id']}"
                                        exon_response = requests.get(exon_url, headers={"Content-Type": "text/plain"})
                                        
                                        if exon_response.status_code == 200 and len(exon_response.text) > 50:
                                            sequences['exon'].append(exon_response.text.strip().upper())
                
                except Exception as e:
                    print(f"Failed to get data for {gene_id}: {e}")
                    continue
                    
                # Rate limiting
                import time
                time.sleep(0.2)
        
        except Exception as e:
            print(f"Ensembl API failed: {e}")
        
        print(f"Downloaded: {len(sequences['promoter'])} promoters, {len(sequences['exon'])} exons")
        
        # Fill with synthetic sequences if we don't have enough real ones
        if len(sequences['promoter']) < n_per_type:
            print("Filling remaining sequences with synthetic ones...")
            synthetic = self.create_synthetic_sequences(n_per_type)
            
            for seq_type in sequences:
                needed = n_per_type - len(sequences[seq_type])
                if needed > 0:
                    sequences[seq_type].extend(synthetic[seq_type][:needed])
        
        return sequences
    
    def extract_layer_representations(self, sequences: Dict[str, List[str]], 
                                    max_length: int = 128) -> Dict:
        """Extract hidden representations from all layers for all sequences"""
        
        print(f"Extracting hidden representations from {self.num_layers} layers...")
        
        # Prepare data
        all_sequences = []
        all_labels = []
        
        for seq_type, seqs in sequences.items():
            all_sequences.extend(seqs)
            all_labels.extend([seq_type] * len(seqs))
        
        print(f"Total sequences: {len(all_sequences)}")
        print(f"Label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
        
        # Extract representations layer by layer
        layer_representations = {i: [] for i in range(self.num_layers + 1)}  # +1 for embeddings
        
        for i, sequence in enumerate(all_sequences):
            if i % 10 == 0:
                print(f"Processing sequence {i+1}/{len(all_sequences)}")
            
            try:
                # Truncate sequence for consistent processing
                seq_truncated = sequence[:400] if len(sequence) > 400 else sequence
                
                inputs = self.tokenizer(seq_truncated, return_tensors='pt', 
                                       truncation=True, max_length=max_length).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                
                # Extract representation from each layer (use mean pooling across sequence length)
                for layer_idx, layer_hidden in enumerate(hidden_states):
                    # layer_hidden shape: (batch=1, seq_len, hidden_size)
                    pooled_repr = torch.mean(layer_hidden[0], dim=0).cpu().numpy()  # Mean pool across tokens
                    layer_representations[layer_idx].append(pooled_repr)
                    
            except Exception as e:
                print(f"Error processing sequence {i}: {e}")
                # Add zero representations for failed sequences
                for layer_idx in range(self.num_layers + 1):
                    layer_representations[layer_idx].append(np.zeros(self.hidden_size))
                continue
        
        # Convert to numpy arrays
        for layer_idx in layer_representations:
            layer_representations[layer_idx] = np.array(layer_representations[layer_idx])
            
        print(f"Extracted representations: {[repr.shape for repr in layer_representations.values()][:3]}...")
        
        return {
            'layer_representations': layer_representations,
            'labels': all_labels,
            'sequences': all_sequences
        }
    
    def train_layer_probes(self, representation_data: Dict, save_dir: str = "analysis/results/phase2") -> Dict:
        """Train linear probes for each layer to classify sequence types"""
        
        print(f"Training linear probes for {len(representation_data['layer_representations'])} layers...")
        
        layer_representations = representation_data['layer_representations']
        labels = representation_data['labels']
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        results = {
            'layer_accuracies': [],
            'layer_models': {},
            'layer_reports': {},
            'label_encoder': label_encoder,
            'class_names': label_encoder.classes_
        }
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for layer_idx in sorted(layer_representations.keys()):
            print(f"Training probe for layer {layer_idx}...")
            
            X = layer_representations[layer_idx]
            y = encoded_labels
            
            # Check if we have valid data
            if X.shape[0] == 0 or np.all(X == 0):
                print(f"Layer {layer_idx}: No valid data, skipping")
                results['layer_accuracies'].append(0.0)
                continue
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train logistic regression probe
            probe = LogisticRegression(random_state=42, max_iter=1000)
            probe.fit(X_train, y_train)
            
            # Evaluate
            y_pred = probe.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Layer {layer_idx}: Accuracy = {accuracy:.3f}")
            
            # Store results
            results['layer_accuracies'].append(accuracy)
            results['layer_models'][layer_idx] = probe
            results['layer_reports'][layer_idx] = classification_report(
                y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
            )
        
        print(f"Best layer: {np.argmax(results['layer_accuracies'])} (accuracy: {np.max(results['layer_accuracies']):.3f})")
        
        # Save results
        with open(save_path / 'probing_results.pkl', 'wb') as f:
            pickle.dump(results, f)
            
        return results
    
    def create_probing_visualizations(self, results: Dict, representation_data: Dict,
                                    save_dir: str = "analysis/results/phase2"):
        """Create comprehensive visualizations of probing results"""
        
        save_path = Path(save_dir)
        
        # 1. Layer-wise accuracy curve
        plt.figure(figsize=(12, 8))
        
        # Main accuracy plot
        plt.subplot(2, 2, 1)
        layer_indices = list(range(len(results['layer_accuracies'])))
        accuracies = results['layer_accuracies']
        
        plt.plot(layer_indices, accuracies, 'o-', linewidth=3, markersize=8, color='darkblue')
        plt.axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='Random chance (25%)')
        
        # Highlight best layer
        best_layer = np.argmax(accuracies)
        best_accuracy = accuracies[best_layer]
        plt.axvline(x=best_layer, color='orange', linestyle='--', alpha=0.7)
        plt.scatter([best_layer], [best_accuracy], color='red', s=100, zorder=5)
        plt.text(best_layer, best_accuracy + 0.02, f'Best: L{best_layer}\n{best_accuracy:.3f}', 
                ha='center', va='bottom', fontweight='bold', color='red')
        
        plt.xlabel('Layer Index (0=Embeddings, 1-12=Transformer)')
        plt.ylabel('Classification Accuracy')
        plt.title('Linear Probe Accuracy Across Layers', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)
        
        # 2. Improvement over chance
        plt.subplot(2, 2, 2)
        improvements = [(acc - 0.25) / 0.25 * 100 for acc in accuracies]  # Percent improvement over 25%
        plt.bar(layer_indices, improvements, color='lightcoral', alpha=0.7)
        plt.xlabel('Layer Index')
        plt.ylabel('Improvement over Random Chance (%)')
        plt.title('Relative Performance by Layer')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Annotate best layers
        sorted_improvements = sorted(enumerate(improvements), key=lambda x: x[1], reverse=True)
        for i, (layer_idx, improvement) in enumerate(sorted_improvements[:3]):
            plt.text(layer_idx, improvement + 5, f'{improvement:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 3. Class-wise performance for best layer
        plt.subplot(2, 2, 3)
        if best_layer in results['layer_reports']:
            report = results['layer_reports'][best_layer]
            class_names = results['class_names']
            f1_scores = [report[cls]['f1-score'] for cls in class_names]
            
            bars = plt.bar(class_names, f1_scores, color='skyblue', alpha=0.7)
            plt.xlabel('Sequence Type')
            plt.ylabel('F1-Score')
            plt.title(f'Per-Class Performance (Layer {best_layer})')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Annotate bars
            for bar, f1 in zip(bars, f1_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Representational hierarchy analysis
        plt.subplot(2, 2, 4)
        
        # Calculate "knowledge emergence" - where accuracy starts increasing significantly
        accuracy_diffs = np.diff(accuracies)  # First derivative
        knowledge_emergence = np.argmax(accuracy_diffs) if len(accuracy_diffs) > 0 else 0
        
        # Calculate "knowledge peak" - layer with maximum accuracy
        knowledge_peak = np.argmax(accuracies)
        
        plt.plot(layer_indices, accuracies, 'o-', linewidth=3, markersize=6, alpha=0.7, label='Accuracy')
        
        # Mark important points
        plt.axvline(x=knowledge_emergence, color='green', linestyle=':', alpha=0.8, 
                   label=f'Knowledge Emergence (L{knowledge_emergence})')
        plt.axvline(x=knowledge_peak, color='purple', linestyle=':', alpha=0.8,
                   label=f'Knowledge Peak (L{knowledge_peak})')
        
        plt.xlabel('Layer Index')
        plt.ylabel('Accuracy')
        plt.title('Representational Hierarchy')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'probing_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: probing_analysis_comprehensive.png")
        
        # 5. Confusion matrix for best layer
        if best_layer in results['layer_models']:
            plt.figure(figsize=(8, 6))
            
            # Get test predictions for confusion matrix
            layer_representations = representation_data['layer_representations']
            labels = representation_data['labels']
            
            X = layer_representations[best_layer]
            y = results['label_encoder'].transform(labels)
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            y_pred = results['layer_models'][best_layer].predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=results['class_names'],
                       yticklabels=results['class_names'])
            plt.title(f'Confusion Matrix - Layer {best_layer} (Best Performance)', fontweight='bold')
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')
            
            plt.tight_layout()
            plt.savefig(save_path / 'confusion_matrix_best_layer.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: confusion_matrix_best_layer.png")
        
        # 6. Save detailed summary
        with open(save_path / 'probing_at_scale_summary.txt', 'w') as f:
            f.write("PROBING AT SCALE ANALYSIS RESULTS\n")
            f.write("=" * 40 + "\n\n")
            f.write("This analysis reveals the model's representational hierarchy:\n")
            f.write("Where does the model 'know' what type of sequence it's looking at?\n\n")
            
            f.write("OVERALL FINDINGS:\n")
            f.write(f"Best performing layer: {best_layer} ({'Embeddings' if best_layer == 0 else f'Transformer Layer {best_layer}'})\n")
            f.write(f"Best accuracy: {best_accuracy:.3f} ({(best_accuracy-0.25)/0.25*100:.1f}% above chance)\n")
            f.write(f"Random baseline: 0.250 (4 classes)\n\n")
            
            # Layer-by-layer breakdown
            f.write("LAYER-WISE PERFORMANCE:\n")
            for i, acc in enumerate(accuracies):
                layer_name = "Embeddings" if i == 0 else f"Transformer-{i}"
                improvement = (acc - 0.25) / 0.25 * 100
                f.write(f"Layer {i:2d} ({layer_name:>12}): {acc:.3f} (+{improvement:4.1f}%)\n")
            f.write("\n")
            
            # Best layer detailed analysis
            if best_layer in results['layer_reports']:
                f.write(f"DETAILED ANALYSIS - LAYER {best_layer}:\n")
                report = results['layer_reports'][best_layer]
                
                f.write("Per-class performance:\n")
                for class_name in results['class_names']:
                    if class_name in report:
                        precision = report[class_name]['precision']
                        recall = report[class_name]['recall']
                        f1 = report[class_name]['f1-score']
                        support = report[class_name]['support']
                        f.write(f"  {class_name:>12}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (n={support})\n")
                
                f.write(f"\nOverall metrics:\n")
                f.write(f"  Macro avg F1: {report['macro avg']['f1-score']:.3f}\n")
                f.write(f"  Weighted avg F1: {report['weighted avg']['f1-score']:.3f}\n")
            
            # Representational insights
            f.write("\nREPRESENTATIONAL HIERARCHY INSIGHTS:\n")
            
            # Find layers where accuracy increases significantly
            significant_jumps = []
            for i in range(1, len(accuracies)):
                if accuracies[i] - accuracies[i-1] > 0.05:  # 5% jump
                    significant_jumps.append((i, accuracies[i] - accuracies[i-1]))
            
            if significant_jumps:
                f.write("Significant accuracy improvements:\n")
                for layer, jump in significant_jumps:
                    f.write(f"  Layer {layer-1} -> {layer}: +{jump:.3f} (+{jump/0.25*100:.1f}%)\n")
            else:
                f.write("No significant jumps in accuracy - gradual learning\n")
                
            f.write(f"\nSequence type learnability ranking:\n")
            if best_layer in results['layer_reports']:
                report = results['layer_reports'][best_layer]
                class_f1s = [(cls, report[cls]['f1-score']) for cls in results['class_names'] if cls in report]
                class_f1s.sort(key=lambda x: x[1], reverse=True)
                
                for i, (class_name, f1) in enumerate(class_f1s):
                    f.write(f"  {i+1}. {class_name}: F1={f1:.3f}\n")
        
        print(f"Saved: probing_at_scale_summary.txt")
    
    def run_comprehensive_analysis(self, n_sequences: int = 40, save_dir: str = "analysis/results/phase2") -> Dict:
        """Run the complete probing at scale analysis"""
        
        print("=== PROBING AT SCALE ANALYSIS ===")
        print(f"Target: {n_sequences} sequences per class")
        
        # 1. Get sequence data
        print("\n1. Collecting sequence data...")
        sequences = self.create_synthetic_sequences(n_sequences)  # Start with synthetic
        
        # Optionally try to add real sequences
        try:
            real_sequences = self.get_arabidopsis_sequences_ensembl(n_sequences // 4)  # Fewer real ones
            for seq_type in sequences:
                sequences[seq_type].extend(real_sequences[seq_type][:5])  # Add up to 5 real ones per type
        except Exception as e:
            print(f"Could not fetch real sequences: {e}")
        
        print(f"Final dataset: {[f'{k}={len(v)}' for k, v in sequences.items()]}")
        
        # 2. Extract representations
        print("\n2. Extracting layer representations...")
        representation_data = self.extract_layer_representations(sequences)
        
        # 3. Train probes
        print("\n3. Training linear probes...")
        probe_results = self.train_layer_probes(representation_data, save_dir)
        
        # 4. Create visualizations
        print("\n4. Creating visualizations...")
        self.create_probing_visualizations(probe_results, representation_data, save_dir)
        
        return {
            'sequences': sequences,
            'representation_data': representation_data,
            'probe_results': probe_results
        }

def main():
    """Run probing at scale analysis"""
    print("=== PHASE 2 - TASK 3: PROBING AT SCALE ===")
    
    # Initialize prober
    prober = SequenceProber()
    
    # Run comprehensive analysis
    results = prober.run_comprehensive_analysis(n_sequences=30)  # 30 per class for efficiency
    
    print(f"\n=== PROBING AT SCALE ANALYSIS COMPLETE ===")
    print(f"Results saved to analysis/results/phase2/")
    print(f"Generated files:")
    print(f"  - probing_analysis_comprehensive.png")
    print(f"  - confusion_matrix_best_layer.png")
    print(f"  - probing_at_scale_summary.txt")
    print(f"  - probing_results.pkl")
    
    # Quick summary
    best_accuracy = max(results['probe_results']['layer_accuracies'])
    best_layer = np.argmax(results['probe_results']['layer_accuracies'])
    improvement = (best_accuracy - 0.25) / 0.25 * 100
    
    print(f"\nQUICK SUMMARY:")
    print(f"  Best layer: {best_layer} ({'Embeddings' if best_layer == 0 else f'Transformer-{best_layer}'})")
    print(f"  Best accuracy: {best_accuracy:.3f} ({improvement:.1f}% above chance)")
    print(f"  Total sequences analyzed: {sum(len(seqs) for seqs in results['sequences'].values())}")

if __name__ == "__main__":
    main()