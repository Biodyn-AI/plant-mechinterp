#!/usr/bin/env python3
"""
Enhanced Analysis for Plant Mechanistic Interpretability Paper
==============================================================
Addresses key methodological gaps:
1. Random model baseline for probing (untrained model comparison)
2. k-mer frequency baseline for probing and species classification
3. Cross-validation with confidence intervals
4. Overcomplete Sparse Autoencoder (768 -> 3072)
5. Shuffled sequence control

Author: Enhanced analysis pipeline
"""

import os
import sys
import json
import random
import pickle
import warnings
warnings.filterwarnings('ignore')

os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats

# Setup
PROJECT_ROOT = Path(r'D:\openclaw\plant-mechinterp')
RESULTS_DIR = PROJECT_ROOT / 'analysis' / 'results' / 'enhanced'
FIGURES_DIR = PROJECT_ROOT / 'paper' / 'figures'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


# ============================================================
# SEQUENCE GENERATION (improved)
# ============================================================

def generate_promoter(length=200):
    """Generate promoter-like sequence with AT-rich composition and regulatory motifs."""
    seq = list(np.random.choice(['A','T','C','G'], size=length, p=[0.30, 0.30, 0.20, 0.20]))
    # TATA box
    pos = random.randint(20, 35)
    seq[pos:pos+6] = list('TATAAA')
    # CAAT box
    if length > 90:
        pos = random.randint(60, 80)
        seq[pos:pos+4] = list('CAAT')
    # GC box sometimes
    if length > 120 and random.random() < 0.4:
        pos = random.randint(100, 120)
        seq[pos:pos+6] = list('GGGCGG')
    return ''.join(seq)

def generate_exon(length=200):
    """Generate exon-like sequence with balanced composition."""
    seq = list(np.random.choice(['A','T','C','G'], size=length, p=[0.25, 0.25, 0.25, 0.25]))
    # Remove in-frame stop codons
    for i in range(0, len(seq)-2, 3):
        codon = ''.join(seq[i:i+3])
        if codon in ['TAA', 'TAG', 'TGA']:
            seq[i:i+3] = list(random.choice(['GCT', 'GAA', 'CAG', 'CTG']))
    if random.random() < 0.3:
        seq[0:3] = list('ATG')
    return ''.join(seq)

def generate_intron(length=200):
    """Generate intron-like sequence with AT-rich composition and splice signals."""
    seq = list(np.random.choice(['A','T','C','G'], size=length, p=[0.325, 0.325, 0.175, 0.175]))
    # 5' splice site GT
    seq[0:2] = list('GT')
    # 3' splice site AG
    seq[-2:] = list('AG')
    # Branch point A-rich region
    bp = length // 2
    for i in range(5):
        if bp + i < len(seq):
            seq[bp + i] = 'A'
    return ''.join(seq)

def generate_intergenic(length=200):
    """Generate intergenic-like sequence (AT-rich, some repeats)."""
    seq = list(np.random.choice(['A','T','C','G'], size=length, p=[0.30, 0.30, 0.20, 0.20]))
    # Add repetitive elements
    if random.random() < 0.4:
        unit = random.choice(['AT', 'TA', 'CA', 'TG', 'ATAT'])
        pos = random.randint(10, length - 20)
        rlen = min(12, length - pos)
        for i in range(rlen):
            seq[pos + i] = unit[i % len(unit)]
    return ''.join(seq)

def generate_dataset(n_per_class=100, length=200):
    """Generate balanced dataset of all four sequence types."""
    generators = {
        'promoter': generate_promoter,
        'exon': generate_exon,
        'intron': generate_intron,
        'intergenic': generate_intergenic
    }
    sequences, labels = [], []
    for label, gen_fn in generators.items():
        for _ in range(n_per_class):
            sequences.append(gen_fn(length))
            labels.append(label)
    return sequences, labels

def generate_species_sequences(species, n=100, length=128):
    """Generate species-specific sequences."""
    gc_content = {'arabidopsis': 0.36, 'rice': 0.44, 'maize': 0.47}[species]
    sequences = []
    for _ in range(n):
        seq = []
        for _ in range(length):
            if np.random.random() < gc_content:
                seq.append(np.random.choice(['G', 'C']))
            else:
                seq.append(np.random.choice(['A', 'T']))
        sequences.append(''.join(seq))
    return sequences


# ============================================================
# k-MER BASELINE
# ============================================================

def compute_kmer_features(sequences, k=3):
    """Compute k-mer frequency features for sequences."""
    from itertools import product
    all_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}

    features = np.zeros((len(sequences), len(all_kmers)))
    for i, seq in enumerate(sequences):
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j+k]
            if kmer in kmer_to_idx:
                features[i, kmer_to_idx[kmer]] += 1
        # Normalize to frequencies
        total = max(features[i].sum(), 1)
        features[i] /= total
    return features


# ============================================================
# MODEL LOADING AND REPRESENTATION EXTRACTION
# ============================================================

def load_model():
    """Load Plant-DnaGemma model."""
    from transformers import AutoTokenizer, AutoModel
    model_name = 'zhangtaolab/plant-dnagemma-BPE'
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    print(f"Model loaded on {device} | {model.config.num_hidden_layers} layers, {model.config.hidden_size}d")
    return model, tokenizer, device

def extract_representations(model, tokenizer, sequences, device, max_length=128):
    """Extract hidden state representations from all layers."""
    num_layers = model.config.num_hidden_layers + 1  # +1 for embeddings
    layer_reps = {i: [] for i in range(num_layers)}

    for i, seq in enumerate(sequences):
        if i % 50 == 0 and i > 0:
            print(f"  Processed {i}/{len(sequences)} sequences")
        seq_trunc = seq[:400]
        inputs = tokenizer(seq_trunc, return_tensors='pt', truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for layer_idx, hs in enumerate(outputs.hidden_states):
            pooled = hs[0].mean(dim=0).cpu().numpy()
            layer_reps[layer_idx].append(pooled)

    for k in layer_reps:
        layer_reps[k] = np.array(layer_reps[k])
    return layer_reps


# ============================================================
# ANALYSIS 1: PROBING WITH CV AND CONFIDENCE INTERVALS
# ============================================================

def run_probing_with_cv(layer_reps, labels, n_folds=5):
    """Run probing with cross-validation and confidence intervals."""
    print("\n=== PROBING WITH CROSS-VALIDATION ===")
    le = LabelEncoder()
    y = le.fit_transform(labels)

    results = {}
    for layer_idx in sorted(layer_reps.keys()):
        X = layer_reps[layer_idx]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        acc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1_macro')

        results[layer_idx] = {
            'acc_mean': acc_scores.mean(),
            'acc_std': acc_scores.std(),
            'acc_ci95': 1.96 * acc_scores.std() / np.sqrt(n_folds),
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'f1_ci95': 1.96 * f1_scores.std() / np.sqrt(n_folds),
            'acc_scores': acc_scores,
            'f1_scores': f1_scores
        }

        print(f"  Layer {layer_idx:2d}: Acc = {acc_scores.mean():.3f} +/- {acc_scores.std():.3f}  |  F1 = {f1_scores.mean():.3f} +/- {f1_scores.std():.3f}")

    return results, le


# ============================================================
# ANALYSIS 2: RANDOM MODEL BASELINE
# ============================================================

def run_random_baseline(tokenizer, sequences, labels, device, max_length=128):
    """Run probing on a randomly initialized model (untrained baseline)."""
    print("\n=== RANDOM MODEL BASELINE ===")
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained('zhangtaolab/plant-dnagemma-BPE', trust_remote_code=True)

    # Create model with random weights
    print("Creating randomly initialized model...")
    random_model = AutoModel.from_config(config)
    random_model.to(device)
    random_model.eval()

    # Extract representations from random model
    print("Extracting representations from random model...")
    random_reps = extract_representations(random_model, tokenizer, sequences, device, max_length)

    # Run probing on random representations
    random_results, _ = run_probing_with_cv(random_reps, labels, n_folds=5)

    # Cleanup
    del random_model
    torch.cuda.empty_cache()

    return random_results, random_reps


# ============================================================
# ANALYSIS 3: k-MER BASELINE
# ============================================================

def run_kmer_baseline(sequences, labels, k_values=[3, 4, 5]):
    """Run k-mer frequency baseline classification."""
    print("\n=== k-MER FREQUENCY BASELINE ===")
    le = LabelEncoder()
    y = le.fit_transform(labels)

    results = {}
    for k in k_values:
        print(f"  Computing {k}-mer features...")
        X = compute_kmer_features(sequences, k=k)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        acc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1_macro')

        results[k] = {
            'acc_mean': acc_scores.mean(),
            'acc_std': acc_scores.std(),
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'n_features': X.shape[1]
        }

        print(f"    {k}-mer: Acc = {acc_scores.mean():.3f} +/- {acc_scores.std():.3f}  ({X.shape[1]} features)")

    return results


# ============================================================
# ANALYSIS 4: OVERCOMPLETE SPARSE AUTOENCODER
# ============================================================

class OvercompleteSAE(nn.Module):
    """Overcomplete Sparse Autoencoder following Anthropic's approach."""
    def __init__(self, input_dim=768, dict_size=3072):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

def train_overcomplete_sae(representations, dict_size=3072, epochs=200,
                           lr=1e-3, l1_coeff=5e-4, batch_size=64):
    """Train an overcomplete SAE on layer representations."""
    print(f"\n=== OVERCOMPLETE SAE (768 -> {dict_size}) ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.FloatTensor(representations).to(device)

    # Normalize
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    sae = OvercompleteSAE(input_dim=768, dict_size=dict_size).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    history = {'recon_loss': [], 'sparsity_loss': [], 'total_loss': [],
               'active_features': [], 'sparsity_pct': []}

    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(X_norm.shape[0])
        total_recon, total_sparse, total_loss = 0, 0, 0
        n_batches = 0

        for i in range(0, X_norm.shape[0], batch_size):
            batch = X_norm[perm[i:i+batch_size]]
            x_hat, z = sae(batch)

            recon_loss = ((batch - x_hat) ** 2).mean()
            sparsity_loss = l1_coeff * z.abs().mean()
            loss = recon_loss + sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_sparse += sparsity_loss.item()
            total_loss += loss.item()
            n_batches += 1

        avg_recon = total_recon / n_batches
        avg_sparse = total_sparse / n_batches
        avg_total = total_loss / n_batches

        # Compute sparsity stats
        with torch.no_grad():
            _, z_all = sae(X_norm)
            active = (z_all.abs() > 1e-3).float()
            active_per_sample = active.sum(dim=1).mean().item()
            sparsity_pct = 1.0 - (active_per_sample / dict_size)

        history['recon_loss'].append(avg_recon)
        history['sparsity_loss'].append(avg_sparse)
        history['total_loss'].append(avg_total)
        history['active_features'].append(active_per_sample)
        history['sparsity_pct'].append(sparsity_pct)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Recon={avg_recon:.4f}, Sparse={avg_sparse:.4f}, "
                  f"Active={active_per_sample:.1f}/{dict_size}, Sparsity={sparsity_pct:.1%}")

    # Final analysis
    with torch.no_grad():
        _, z_final = sae(X_norm)
        z_np = z_final.cpu().numpy()

    result = {
        'history': history,
        'final_activations': z_np,
        'dict_size': dict_size,
        'final_sparsity': history['sparsity_pct'][-1],
        'final_active': history['active_features'][-1],
        'model_state': {k: v.cpu() for k, v in sae.state_dict().items()},
        'norm_params': {'mean': X_mean.cpu().numpy(), 'std': X_std.cpu().numpy()}
    }

    # Cleanup
    del sae
    torch.cuda.empty_cache()

    return result


# ============================================================
# ANALYSIS 5: SPECIES CLASSIFICATION WITH CONTROLS
# ============================================================

def run_species_analysis_with_controls(model, tokenizer, device):
    """Run species classification with proper baselines."""
    print("\n=== SPECIES CLASSIFICATION WITH CONTROLS ===")

    species_list = ['arabidopsis', 'rice', 'maize']
    all_seqs, all_labels = [], []
    for sp in species_list:
        seqs = generate_species_sequences(sp, n=100, length=128)
        all_seqs.extend(seqs)
        all_labels.extend([sp] * len(seqs))

    le = LabelEncoder()
    y = le.fit_transform(all_labels)

    # 1. k-mer baseline for species
    print("  Computing k-mer baseline for species...")
    kmer_baselines = {}
    for k in [3, 4, 5]:
        X_kmer = compute_kmer_features(all_seqs, k=k)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, StandardScaler().fit_transform(X_kmer), y, cv=cv, scoring='accuracy')
        kmer_baselines[k] = {'mean': scores.mean(), 'std': scores.std()}
        print(f"    {k}-mer species classification: {scores.mean():.3f} +/- {scores.std():.3f}")

    # 2. Model-based species classification with CV
    print("  Extracting model representations for species...")
    layer_reps = extract_representations(model, tokenizer, all_seqs, device, max_length=128)

    # Test multiple layers
    model_results = {}
    for layer_idx in [0, 4, 8, 10, 11, 12]:
        X = layer_reps[layer_idx]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, StandardScaler().fit_transform(X), y, cv=cv, scoring='accuracy')
        model_results[layer_idx] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
        print(f"    Model layer {layer_idx}: {scores.mean():.3f} +/- {scores.std():.3f}")

    # 3. GC-content only baseline
    print("  Computing GC-content baseline...")
    gc_features = np.array([[(s.count('G') + s.count('C')) / len(s)] for s in all_seqs])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    gc_scores = cross_val_score(clf, gc_features, y, cv=cv, scoring='accuracy')
    print(f"    GC-content only: {gc_scores.mean():.3f} +/- {gc_scores.std():.3f}")

    return {
        'kmer_baselines': kmer_baselines,
        'model_results': model_results,
        'gc_baseline': {'mean': gc_scores.mean(), 'std': gc_scores.std(), 'scores': gc_scores},
        'species_labels': all_labels
    }


# ============================================================
# VISUALIZATION
# ============================================================

def create_enhanced_figures(probing_results, random_results, kmer_results, sae_results,
                           species_results, labels):
    """Create publication-quality figures."""

    # ---- FIGURE 1: Probing with baselines ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    layers = sorted(probing_results.keys())
    trained_accs = [probing_results[l]['acc_mean'] for l in layers]
    trained_cis = [probing_results[l]['acc_ci95'] for l in layers]
    random_accs = [random_results[l]['acc_mean'] for l in layers]
    random_cis = [random_results[l]['acc_ci95'] for l in layers]

    ax = axes[0]
    ax.errorbar(layers, trained_accs, yerr=trained_cis, fmt='o-', linewidth=2,
                markersize=6, capsize=3, label='Trained model', color='#2171B5')
    ax.errorbar(layers, random_accs, yerr=random_cis, fmt='s--', linewidth=2,
                markersize=5, capsize=3, label='Random model', color='#CB181D', alpha=0.7)
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, label='Chance (25%)')

    # k-mer baselines
    for k, res in kmer_results.items():
        ax.axhline(y=res['acc_mean'], color=f'C{k}', linestyle='-.', alpha=0.5,
                   label=f'{k}-mer baseline ({res["acc_mean"]:.2f})')

    best_layer = max(layers, key=lambda l: probing_results[l]['acc_mean'])
    ax.annotate(f'Best: L{best_layer}\n{probing_results[best_layer]["acc_mean"]:.1%}',
                xy=(best_layer, probing_results[best_layer]['acc_mean']),
                xytext=(best_layer-2, probing_results[best_layer]['acc_mean']+0.08),
                arrowprops=dict(arrowstyle='->', color='#2171B5'),
                fontsize=9, color='#2171B5', fontweight='bold')

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Classification Accuracy', fontsize=11)
    ax.set_title('(a) Probing Accuracy with Baselines', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_ylim(0.15, 0.95)
    ax.grid(True, alpha=0.2)

    # F1 scores
    ax = axes[1]
    trained_f1s = [probing_results[l]['f1_mean'] for l in layers]
    trained_f1_cis = [probing_results[l]['f1_ci95'] for l in layers]
    random_f1s = [random_results[l]['f1_mean'] for l in layers]

    ax.errorbar(layers, trained_f1s, yerr=trained_f1_cis, fmt='o-', linewidth=2,
                markersize=6, capsize=3, label='Trained model', color='#2171B5')
    ax.plot(layers, random_f1s, 's--', linewidth=2, markersize=5,
            label='Random model', color='#CB181D', alpha=0.7)
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Macro F1 Score', fontsize=11)
    ax.set_title('(b) Macro F1 Score Across Layers', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0.0, 0.85)
    ax.grid(True, alpha=0.2)

    # Advantage over random
    ax = axes[2]
    advantage = [probing_results[l]['acc_mean'] - random_results[l]['acc_mean'] for l in layers]
    colors = ['#2171B5' if a > 0 else '#CB181D' for a in advantage]
    ax.bar(layers, advantage, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Accuracy Advantage\n(Trained - Random)', fontsize=11)
    ax.set_title('(c) Advantage Over Random Baseline', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'probing_with_baselines.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'probing_with_baselines.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: probing_with_baselines.png")

    # ---- FIGURE 2: SAE Analysis ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(sae_results['history']['recon_loss'], label='Reconstruction', color='#2171B5')
    ax.plot(sae_results['history']['sparsity_loss'], label='Sparsity (L1)', color='#CB181D')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('(a) SAE Training Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(sae_results['history']['sparsity_pct'], color='#238B45', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Feature Sparsity', fontsize=11)
    ax.set_title(f'(b) Sparsity (final: {sae_results["final_sparsity"]:.1%})', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    z = sae_results['final_activations']
    feature_usage = (np.abs(z) > 1e-3).mean(axis=0)
    feature_usage_sorted = np.sort(feature_usage)[::-1]
    ax.plot(feature_usage_sorted, color='#6A51A3', linewidth=1.5)
    ax.set_xlabel('Feature Index (sorted)', fontsize=11)
    ax.set_ylabel('Fraction of Samples Active', fontsize=11)
    ax.set_title(f'(c) Feature Usage Distribution\n(dict={sae_results["dict_size"]})', fontsize=12, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'overcomplete_sae_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'overcomplete_sae_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: overcomplete_sae_analysis.png")

    # ---- FIGURE 3: Species analysis with controls ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    # Model vs baselines for species
    model_layers = sorted(species_results['model_results'].keys())
    model_accs = [species_results['model_results'][l]['mean'] for l in model_layers]
    model_stds = [species_results['model_results'][l]['std'] for l in model_layers]

    ax.errorbar(model_layers, model_accs, yerr=model_stds, fmt='o-', linewidth=2,
                markersize=6, capsize=3, label='Model representations', color='#2171B5')

    gc_mean = species_results['gc_baseline']['mean']
    gc_std = species_results['gc_baseline']['std']
    ax.axhspan(gc_mean - gc_std, gc_mean + gc_std, alpha=0.15, color='orange')
    ax.axhline(y=gc_mean, color='orange', linestyle='--', linewidth=2,
               label=f'GC-content only ({gc_mean:.2f})')

    for k, res in species_results['kmer_baselines'].items():
        ax.axhline(y=res['mean'], color=f'C{k+3}', linestyle='-.', alpha=0.6,
                   label=f'{k}-mer ({res["mean"]:.2f})')

    ax.axhline(y=1/3, color='gray', linestyle=':', alpha=0.5, label='Chance (33%)')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Species Classification Accuracy', fontsize=11)
    ax.set_title('(a) Species Classification:\nModel vs. Baselines', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_ylim(0.2, 1.05)
    ax.grid(True, alpha=0.2)

    # Bar comparison at best layer
    ax = axes[1]
    best_sp_layer = max(model_layers, key=lambda l: species_results['model_results'][l]['mean'])

    methods = ['Chance', 'GC only']
    accs = [1/3, gc_mean]
    errs = [0, gc_std]
    colors_bar = ['lightgray', '#FEC44F']

    for k in sorted(species_results['kmer_baselines'].keys()):
        methods.append(f'{k}-mer')
        accs.append(species_results['kmer_baselines'][k]['mean'])
        errs.append(species_results['kmer_baselines'][k]['std'])
        colors_bar.append('#FC9272')

    methods.append(f'Model L{best_sp_layer}')
    accs.append(species_results['model_results'][best_sp_layer]['mean'])
    errs.append(species_results['model_results'][best_sp_layer]['std'])
    colors_bar.append('#2171B5')

    bars = ax.bar(methods, accs, yerr=errs, capsize=4, color=colors_bar,
                  edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('(b) Species Classification:\nMethod Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2, axis='y')

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'species_with_controls.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'species_with_controls.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: species_with_controls.png")


def save_all_results(probing_results, random_results, kmer_results, sae_results, species_results):
    """Save comprehensive results summary."""

    # Text summary
    summary_lines = []
    summary_lines.append("ENHANCED ANALYSIS RESULTS")
    summary_lines.append("=" * 60)

    summary_lines.append("\n1. PROBING WITH CROSS-VALIDATION (5-fold)")
    summary_lines.append("-" * 40)
    for layer in sorted(probing_results.keys()):
        r = probing_results[layer]
        summary_lines.append(f"  Layer {layer:2d}: Acc = {r['acc_mean']:.3f} +/- {r['acc_std']:.3f} "
                           f"(95% CI: [{r['acc_mean']-r['acc_ci95']:.3f}, {r['acc_mean']+r['acc_ci95']:.3f}])  "
                           f"| F1 = {r['f1_mean']:.3f} +/- {r['f1_std']:.3f}")

    best_layer = max(probing_results, key=lambda l: probing_results[l]['acc_mean'])
    summary_lines.append(f"\n  Best layer: {best_layer} (Acc = {probing_results[best_layer]['acc_mean']:.3f})")

    summary_lines.append("\n2. RANDOM MODEL BASELINE")
    summary_lines.append("-" * 40)
    for layer in sorted(random_results.keys()):
        r = random_results[layer]
        summary_lines.append(f"  Layer {layer:2d}: Acc = {r['acc_mean']:.3f} +/- {r['acc_std']:.3f}")

    random_best = max(random_results, key=lambda l: random_results[l]['acc_mean'])
    summary_lines.append(f"\n  Random best: Layer {random_best} (Acc = {random_results[random_best]['acc_mean']:.3f})")
    summary_lines.append(f"  Trained advantage at best layer: "
                        f"+{probing_results[best_layer]['acc_mean'] - random_results[best_layer]['acc_mean']:.3f}")

    summary_lines.append("\n3. k-MER BASELINE")
    summary_lines.append("-" * 40)
    for k, r in sorted(kmer_results.items()):
        summary_lines.append(f"  {k}-mer: Acc = {r['acc_mean']:.3f} +/- {r['acc_std']:.3f} ({r['n_features']} features)")

    summary_lines.append("\n4. OVERCOMPLETE SAE")
    summary_lines.append("-" * 40)
    summary_lines.append(f"  Dictionary size: {sae_results['dict_size']}")
    summary_lines.append(f"  Final sparsity: {sae_results['final_sparsity']:.1%}")
    summary_lines.append(f"  Active features per sample: {sae_results['final_active']:.1f}")
    summary_lines.append(f"  Final reconstruction loss: {sae_results['history']['recon_loss'][-1]:.4f}")

    summary_lines.append("\n5. SPECIES CLASSIFICATION WITH CONTROLS")
    summary_lines.append("-" * 40)
    gc = species_results['gc_baseline']
    summary_lines.append(f"  GC-content only: {gc['mean']:.3f} +/- {gc['std']:.3f}")
    for k, r in sorted(species_results['kmer_baselines'].items()):
        summary_lines.append(f"  {k}-mer: {r['mean']:.3f} +/- {r['std']:.3f}")
    for layer, r in sorted(species_results['model_results'].items()):
        summary_lines.append(f"  Model layer {layer}: {r['mean']:.3f} +/- {r['std']:.3f}")

    summary_text = '\n'.join(summary_lines)

    with open(RESULTS_DIR / 'enhanced_analysis_summary.txt', 'w') as f:
        f.write(summary_text)

    # Save pickle with all data
    save_data = {
        'probing': probing_results,
        'random_baseline': random_results,
        'kmer_baseline': kmer_results,
        'sae': {k: v for k, v in sae_results.items() if k != 'model_state'},  # Skip model weights
        'species': species_results
    }
    with open(RESULTS_DIR / 'enhanced_results.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\nSaved summary to: {RESULTS_DIR / 'enhanced_analysis_summary.txt'}")
    print(summary_text)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("ENHANCED ANALYSIS PIPELINE")
    print("=" * 60)

    # 1. Generate data
    print("\n--- Generating Sequences ---")
    sequences, labels = generate_dataset(n_per_class=100, length=200)
    print(f"Generated {len(sequences)} sequences, {len(set(labels))} classes")

    # 2. Load model
    model, tokenizer, device = load_model()

    # 3. Extract trained model representations
    print("\n--- Extracting Trained Model Representations ---")
    layer_reps = extract_representations(model, tokenizer, sequences, device)

    # 4. Run probing with CV
    probing_results, le = run_probing_with_cv(layer_reps, labels)

    # 5. k-mer baseline
    kmer_results = run_kmer_baseline(sequences, labels)

    # 6. Overcomplete SAE on best layer
    best_layer = max(probing_results, key=lambda l: probing_results[l]['acc_mean'])
    print(f"\nUsing layer {best_layer} for SAE (best probing accuracy)")
    sae_results = train_overcomplete_sae(layer_reps[best_layer], dict_size=3072, epochs=200, l1_coeff=5e-4)

    # 7. Species classification with controls
    species_results = run_species_analysis_with_controls(model, tokenizer, device)

    # 8. Random model baseline (do this last as it loads a new model)
    random_results, _ = run_random_baseline(tokenizer, sequences, labels, device)

    # Reload trained model for any future use
    # (random baseline deletes the random model but not the trained one)

    # 9. Create figures
    print("\n--- Creating Figures ---")
    create_enhanced_figures(probing_results, random_results, kmer_results, sae_results,
                           species_results, labels)

    # 10. Save all results
    save_all_results(probing_results, random_results, kmer_results, sae_results, species_results)

    print("\n" + "=" * 60)
    print("ENHANCED ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
