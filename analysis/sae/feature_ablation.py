"""Causal validation of SAE feature interpretations — reviewer §5.4.

For each SAE feature f in a curated set (motif-matched + region-matched):
  1. Compute SAE latents z on all region_type sequences.
  2. Zero z[f] and reconstruct x̂ = D z'.
  3. Pass x̂ through a frozen layer-7 probe.
  4. Measure per-class accuracy change. A feature meaningfully encoding
     region-type r should disproportionately hurt accuracy on class r when
     ablated.

This is a genuinely causal test: it asks whether the feature's reconstruction
contribution is necessary for the classifier's decision, not merely
correlated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ACT_ROOT = REPO / "data" / "real" / "activations"
ARAREG = REPO / "data" / "real" / "arareg"
SAE_DIR = REPO / "data" / "real" / "results" / "sae"
OUT = REPO / "data" / "real" / "results" / "sae_enrichment"


def rebuild_sae(state_dict, variant, d_in, d_hidden, topk):
    class L1SAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(d_in, d_hidden, bias=True)
            self.dec = nn.Linear(d_hidden, d_in, bias=True)
        def forward(self, x):
            z = torch.relu(self.enc(x))
            return self.dec(z), z
    class TopKSAE(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.enc = nn.Linear(d_in, d_hidden, bias=True)
            self.dec = nn.Linear(d_hidden, d_in, bias=True)
            self.k = k
        def forward(self, x):
            pre = self.enc(x)
            vals, idx = pre.topk(self.k, dim=-1)
            gate = torch.zeros_like(pre).scatter_(-1, idx, vals)
            return self.dec(torch.relu(gate)), torch.relu(gate)
    m = L1SAE() if variant == "l1" else TopKSAE(topk)
    m.load_state_dict(state_dict)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-tag", default="topk_exp16_L7_seed0")
    ap.add_argument("--dataset", default="region_type")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--top-per-class", type=int, default=5,
                    help="ablate top-k most region-enriched features per class")
    args = ap.parse_args()

    with open(SAE_DIR / f"{args.sae_tag}.json") as f:
        meta = json.load(f)
    sd = torch.load(SAE_DIR / f"{args.sae_tag}.pt", map_location="cpu")
    mu = np.asarray(meta["mu"], dtype=np.float32)
    sigma = np.asarray(meta["sigma"], dtype=np.float32)
    sae = rebuild_sae(sd, meta["variant"], meta["d_in"], meta["d_hidden"], meta.get("topk"))

    act = np.load(ACT_ROOT / args.dataset / "trained.npz", allow_pickle=True)
    H = act["hidden"][:, args.layer, :].astype(np.float32)
    labels = act["labels"].astype(str)
    split = act["split"].astype(str)
    mask_tv = np.isin(split, ["train", "val"])
    mask_te = split == "test"

    # Train a layer-7 probe on standardized H (matches probing methodology).
    sc = StandardScaler().fit(H[mask_tv])
    Hs = sc.transform(H)
    clf = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
    clf.fit(Hs[mask_tv], labels[mask_tv])
    classes = clf.classes_.tolist()

    def _probe(hidden):
        s = sc.transform(hidden)
        pred = clf.predict(s)
        return pred

    # Baseline
    pred0 = _probe(H)
    base_acc = accuracy_score(labels[mask_te], pred0[mask_te])
    base_per_class = {
        c: accuracy_score(labels[mask_te][labels[mask_te] == c],
                          pred0[mask_te][labels[mask_te] == c])
        for c in classes
    }
    print(f"Baseline test acc: {base_acc:.3f}")
    print(f"Baseline per-class: {base_per_class}")

    # Reconstruction baseline (SAE→decoded) before ablation
    with torch.no_grad():
        Xn = (H - mu) / sigma
        recon_std, z = sae(torch.from_numpy(Xn))
    recon = recon_std.numpy() * sigma + mu
    pred_recon = _probe(recon)
    recon_acc = accuracy_score(labels[mask_te], pred_recon[mask_te])
    print(f"SAE-reconstructed baseline acc: {recon_acc:.3f}")

    # Select features to ablate: top-K by region enrichment (from §5.3 output).
    ann_path = OUT / f"{args.sae_tag}_annotation_best_per_feature.csv"
    if not ann_path.exists():
        raise SystemExit(
            f"Need {ann_path} from annotation_enrichment.py; run that first."
        )
    ann = pd.read_csv(ann_path)
    ann = ann[ann["q"] < 0.05]
    selected = []
    for c in classes:
        sub = ann[ann["region"] == c].sort_values("q").head(args.top_per_class)
        for _, r in sub.iterrows():
            selected.append(dict(feature=int(r["feature"]), region=c,
                                 fold=float(r["fold_enrichment"]),
                                 q=float(r["q"])))
    print(f"Selected {len(selected)} features to ablate (top {args.top_per_class} per class)")

    rows = []
    for item in selected:
        f = item["feature"]
        # Ablate feature f: set z[:, f] = 0, decode, probe
        with torch.no_grad():
            z_ab = z.clone()
            z_ab[:, f] = 0.0
            x_ab_std = sae.dec(z_ab)
        x_ab = x_ab_std.numpy() * sigma + mu
        pred = _probe(x_ab)
        # Per-class test accuracy
        per_class = {}
        for c in classes:
            mask = mask_te & (labels == c)
            per_class[c] = accuracy_score(labels[mask], pred[mask])
        total = accuracy_score(labels[mask_te], pred[mask_te])
        row = dict(
            feature=f,
            target_region=item["region"],
            fold_enrichment=item["fold"],
            q_annotation=item["q"],
            total_acc=total,
            total_acc_drop=recon_acc - total,
        )
        for c in classes:
            row[f"acc_{c}"] = per_class[c]
            row[f"drop_{c}"] = base_per_class[c] - per_class[c]
        rows.append(row)
        print(f"  feat {f} ({item['region']}): total drop {recon_acc - total:+.3f}; "
              + "  ".join(f"{c}:{recon_acc - per_class[c]:+.2f}" for c in classes))

    df = pd.DataFrame(rows)
    out = OUT / f"{args.sae_tag}_feature_ablation.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    # Diagnostic: does ablating a feature of region r hurt class r more than others?
    print("\nMean acc drop per (ablated feature's region × class):")
    for target in classes:
        sub = df[df["target_region"] == target]
        if len(sub) == 0:
            continue
        drops = {c: sub[f"drop_{c}"].mean() for c in classes}
        argmax = max(drops, key=lambda k: drops[k])
        print(f"  ablate {target}-features: {drops}  (worst hit: {argmax})")


if __name__ == "__main__":
    main()
