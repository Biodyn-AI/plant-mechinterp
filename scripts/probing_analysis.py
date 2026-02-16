#!/usr/bin/env python3
"""Probing classifiers for genomic features in plant foundation models"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import pickle

def create_synthetic_genomic_dataset():
    """Create synthetic genomic sequences with labels for probing"""
    np.random.seed(42)
    
    sequences = []
    labels = []
    feature_names = ['promoter', 'exon', 'intron', 'utr']
    
    # Generate synthetic sequences with known patterns
    patterns = {
        'promoter': ['TATA', 'CAAT', 'GC', 'AT'],  # Promoter-like patterns
        'exon': ['ATG', 'TGA', 'TAG', 'TAA'],       # Start/stop codons
        'intron': ['GT', 'AG'],                     # Splice sites
        'utr': ['AAA', 'TTT']                       # UTR patterns
    }
    
    for feature_idx, feature in enumerate(feature_names):
        for _ in range(100):  # 100 sequences per feature
            # Create base random sequence
            seq = ''.join(np.random.choice(['A', 'T', 'C', 'G'], 200))
            
            # Insert characteristic patterns
            feature_patterns = patterns[feature]
            for pattern in feature_patterns:
                if np.random.random() < 0.7:  # 70% chance of pattern
                    pos = np.random.randint(0, len(seq) - len(pattern))
                    seq = seq[:pos] + pattern + seq[pos + len(pattern):]
            
            sequences.append(seq)
            labels.append(feature_idx)
    
    return sequences, labels, feature_names

def extract_hidden_states(model, tokenizer, sequences, device='cpu'):
    """Extract hidden states from all layers"""
    model.to(device)
    model.eval()
    
    all_hidden_states = []
    
    print(f"Extracting hidden states from {len(sequences)} sequences...")
    
    for i, sequence in enumerate(sequences):
        if i % 50 == 0:
            print(f"Processing sequence {i+1}/{len(sequences)}")
        
        inputs = tokenizer(sequence, return_tensors='pt', 
                          truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Extract hidden states from all layers
        hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden_size)
        
        # Pool sequence representations (mean pooling)
        layer_representations = []
        for layer_hidden in hidden_states:
            # Mean pool over sequence length
            pooled = torch.mean(layer_hidden[0], dim=0).cpu().numpy()
            layer_representations.append(pooled)
        
        all_hidden_states.append(layer_representations)
    
    return all_hidden_states

def train_probing_classifiers(hidden_states, labels, feature_names):
    """Train linear probing classifiers for each layer"""
    n_layers = len(hidden_states[0])
    n_sequences = len(hidden_states)
    
    layer_accuracies = []
    layer_classifiers = []
    
    print(f"Training probing classifiers for {n_layers} layers...")
    
    for layer_idx in range(n_layers):
        print(f"Training classifier for layer {layer_idx + 1}/{n_layers}")
        
        # Extract features for this layer
        layer_features = np.array([hidden_states[i][layer_idx] for i in range(n_sequences)])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            layer_features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Train classifier
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        layer_accuracies.append(accuracy)
        layer_classifiers.append(classifier)
        
        print(f"  Layer {layer_idx + 1} accuracy: {accuracy:.3f}")
    
    return layer_accuracies, layer_classifiers

def analyze_feature_importance(classifiers, feature_names):
    """Analyze feature importance in each layer's classifier"""
    n_layers = len(classifiers)
    n_features = len(feature_names)
    
    # Get coefficient magnitudes for each layer
    layer_importance = []
    
    for layer_idx, classifier in enumerate(classifiers):
        if hasattr(classifier, 'coef_'):
            # For multi-class, coef_ is (n_classes, n_features)
            coefs = classifier.coef_
            # Take L2 norm across classes for each feature
            importance = np.linalg.norm(coefs, axis=0)
            layer_importance.append(importance)
        else:
            layer_importance.append(np.zeros(classifier.n_features_in_))
    
    return np.array(layer_importance)

def plot_probing_results(accuracies, feature_names, save_dir="analysis/results"):
    """Plot probing classifier results"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot accuracy by layer
    plt.figure(figsize=(10, 6))
    layers = list(range(1, len(accuracies) + 1))
    plt.plot(layers, accuracies, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.title(f'Linear Probing Accuracy by Layer\n({", ".join(feature_names)} classification)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add horizontal line for chance level
    chance_level = 1.0 / len(feature_names)
    plt.axhline(y=chance_level, color='red', linestyle='--', alpha=0.7, 
                label=f'Chance level ({chance_level:.2f})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'probing_accuracy_by_layer.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved probing accuracy plot: probing_accuracy_by_layer.png")

def plot_feature_importance_heatmap(importance_matrix, feature_names, save_dir="analysis/results"):
    """Plot feature importance heatmap across layers"""
    save_dir = Path(save_dir)
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    im = plt.imshow(importance_matrix.T, aspect='auto', cmap='viridis')
    
    # Set labels
    plt.xlabel('Layer')
    plt.ylabel('Hidden Unit')
    plt.title('Feature Importance by Layer and Hidden Unit')
    
    # Set ticks
    plt.xticks(range(len(importance_matrix)), 
               [f'L{i+1}' for i in range(len(importance_matrix))])
    
    # Add colorbar
    plt.colorbar(im, label='Importance Score')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_importance_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved feature importance heatmap: feature_importance_heatmap.png")

def analyze_layer_specialization(accuracies, feature_names):
    """Analyze which layers are most specialized for different features"""
    best_layer = np.argmax(accuracies) + 1
    best_accuracy = np.max(accuracies)
    
    # Calculate improvement over baseline
    baseline_accuracy = 1.0 / len(feature_names)
    improvement = best_accuracy - baseline_accuracy
    
    results = {
        'best_layer': best_layer,
        'best_accuracy': best_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'improvement': improvement,
        'layer_accuracies': accuracies
    }
    
    print("\nLayer Specialization Analysis:")
    print(f"Best performing layer: {best_layer}")
    print(f"Best accuracy: {best_accuracy:.3f}")
    print(f"Improvement over chance: {improvement:.3f}")
    print(f"Relative improvement: {improvement/baseline_accuracy*100:.1f}%")
    
    return results

def save_analysis_results(results_dict, save_dir="analysis/results"):
    """Save all analysis results"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle
    with open(save_dir / "probing_analysis_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)
    
    # Save summary as text
    with open(save_dir / "probing_analysis_summary.txt", "w") as f:
        f.write("Probing Analysis Summary\n")
        f.write("========================\n\n")
        
        f.write(f"Model: {results_dict['model_name']}\n")
        f.write(f"Number of layers: {len(results_dict['accuracies'])}\n")
        f.write(f"Number of features: {len(results_dict['feature_names'])}\n")
        f.write(f"Dataset size: {results_dict['dataset_size']}\n\n")
        
        f.write("Layer-wise accuracies:\n")
        for i, acc in enumerate(results_dict['accuracies']):
            f.write(f"  Layer {i+1}: {acc:.3f}\n")
        
        f.write(f"\nBest layer: {results_dict['specialization']['best_layer']}\n")
        f.write(f"Best accuracy: {results_dict['specialization']['best_accuracy']:.3f}\n")
        f.write(f"Improvement over chance: {results_dict['specialization']['improvement']:.3f}\n")
    
    print("Analysis results saved!")

def main():
    """Main probing analysis pipeline"""
    print("Starting probing classifier analysis...")
    
    # Load successful model
    with open("models/successful_model.txt", "r") as f:
        model_name = f.read().strip()
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create synthetic dataset
    print("Creating synthetic genomic dataset...")
    sequences, labels, feature_names = create_synthetic_genomic_dataset()
    print(f"Created {len(sequences)} sequences with {len(feature_names)} features")
    
    # Extract hidden states
    hidden_states = extract_hidden_states(model, tokenizer, sequences, device)
    
    # Train probing classifiers
    accuracies, classifiers = train_probing_classifiers(hidden_states, labels, feature_names)
    
    # Analyze feature importance
    importance_matrix = analyze_feature_importance(classifiers, feature_names)
    
    # Analyze layer specialization
    specialization_results = analyze_layer_specialization(accuracies, feature_names)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_probing_results(accuracies, feature_names)
    plot_feature_importance_heatmap(importance_matrix, feature_names)
    
    # Save results
    results_dict = {
        'model_name': model_name,
        'feature_names': feature_names,
        'accuracies': accuracies,
        'classifiers': classifiers,
        'importance_matrix': importance_matrix,
        'specialization': specialization_results,
        'dataset_size': len(sequences)
    }
    
    save_analysis_results(results_dict)
    
    print("Probing analysis complete!")

if __name__ == "__main__":
    main()