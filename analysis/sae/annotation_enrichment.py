"""SAE feature → genomic annotation enrichment (reviewer §5.3).

For each SAE feature, test whether its top-activating sequences are enriched
for any of the 5 region classes (exon/intron/utr5/utr3/intergenic) relative
to the base rate, using Fisher's exact test with BH FDR correction.

This is orthogonal to the motif-enrichment analysis: motifs are sequence
patterns; region-class labels are genomic annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ACT_ROOT = REPO / "data" / "real" / "activations"
ARAREG = REPO / "data" / "real" / "arareg"
SAE_DIR = REPO / "data" / "real" / "results" / "sae"
OUT = REPO / "data" / "real" / "results" / "sae_enrichment"
OUT.mkdir(parents=True, exist_ok=True)


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


def bh_fdr(p):
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-tag", required=True)
    ap.add_argument("--dataset", default="region_type")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--top-k-seqs", type=int, default=100)
    ap.add_argument("--n-features", type=int, default=300)
    args = ap.parse_args()

    # Load SAE
    with open(SAE_DIR / f"{args.sae_tag}.json") as f:
        meta = json.load(f)
    sd = torch.load(SAE_DIR / f"{args.sae_tag}.pt", map_location="cpu")
    mu = np.asarray(meta["mu"], dtype=np.float32)
    sigma = np.asarray(meta["sigma"], dtype=np.float32)
    model = rebuild_sae(sd, meta["variant"], meta["d_in"], meta["d_hidden"], meta.get("topk"))

    # Load activations & labels
    act = np.load(ACT_ROOT / args.dataset / "trained.npz", allow_pickle=True)
    X = act["hidden"][:, args.layer, :].astype(np.float32)
    Xn = (X - mu) / sigma
    labels = act["labels"].astype(str)
    with torch.no_grad():
        _, Z = model(torch.from_numpy(Xn))
    Z = Z.numpy()
    N = len(labels)
    classes = sorted(np.unique(labels).tolist())
    base_rates = {c: float((labels == c).mean()) for c in classes}
    print(f"Z shape: {Z.shape}  base rates: {base_rates}")

    feat_mass = Z.sum(axis=0)
    feat_order = np.argsort(-feat_mass)[: args.n_features]

    rows = []
    for fi, f in enumerate(feat_order):
        top_idx = np.argsort(-Z[:, f])[: args.top_k_seqs]
        top_labels = labels[top_idx]
        for c in classes:
            a = int((top_labels == c).sum())  # top & in class
            b = int((top_labels != c).sum())  # top & not in class
            cc = int((labels == c).sum()) - a  # not-top & in class
            dd = int((labels != c).sum()) - b  # not-top & not in class
            table = np.array([[a, b], [cc, dd]])
            try:
                odds_ratio, p = fisher_exact(table, alternative="greater")
            except Exception:
                odds_ratio, p = 1.0, 1.0
            observed_rate = a / max(1, args.top_k_seqs)
            rows.append(dict(
                feature=int(f),
                region=c,
                top_count=a,
                top_rate=observed_rate,
                base_rate=base_rates[c],
                fold_enrichment=observed_rate / base_rates[c] if base_rates[c] > 0 else np.nan,
                odds_ratio=float(odds_ratio),
                p=float(p),
            ))
    res = pd.DataFrame(rows)
    res["q"] = bh_fdr(res["p"].to_numpy())
    best = res.sort_values(["feature", "q"]).groupby("feature", as_index=False).first()
    sig = best[best["q"] < 0.05]
    out = OUT / f"{args.sae_tag}_annotation.csv"
    best_out = OUT / f"{args.sae_tag}_annotation_best_per_feature.csv"
    res.to_csv(out, index=False)
    best.to_csv(best_out, index=False)
    print(f"Wrote {out}  ({len(res)} rows) and {best_out}")
    print(f"{len(sig)}/{len(best)} features have q<0.05 for one region class.")
    print()
    print("Top by class:")
    for c in classes:
        sub = sig[sig["region"] == c].sort_values("q").head(5)
        print(f"\n  {c}: {len(sig[sig['region']==c])} features; top 5:")
        for _, r in sub.iterrows():
            print(f"    feat {int(r['feature'])}: top_rate={r['top_rate']:.2f}  "
                  f"base={r['base_rate']:.2f}  fold={r['fold_enrichment']:.2f}  q={r['q']:.2e}")


if __name__ == "__main__":
    main()
