"""Motif enrichment per SAE feature — reviewer §5.

For each SAE feature f, rank its top-activating real sequences, then test
whether any JASPAR plantae 2024 motif scores higher in those sequences than in
a random background set. Use Mann–Whitney U for significance (fast) and BH
FDR correction across all feature × motif tests.

Implementation optimizations:
- Pre-encode every sequence to a one-hot (N, L, 4) matrix once.
- Per motif: compute (N,) max-log-odds score via batched numpy convolution.
- Pre-cache scores into (N, M) matrix so statistical tests are O(1) per
  feature × motif pair.

Outputs:
    data/real/results/sae_enrichment/<sae_tag>_scores.npy      # (N, M)
    data/real/results/sae_enrichment/<sae_tag>_motif_names.csv
    data/real/results/sae_enrichment/<sae_tag>_feature_motif.csv (long-form)
    data/real/results/sae_enrichment/<sae_tag>_best_per_feature.csv
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ACT_ROOT = REPO / "data" / "real" / "activations"
ARAREG = REPO / "data" / "real" / "arareg"
SAE_DIR = REPO / "data" / "real" / "results" / "sae"
MOTIFS = REPO / "data" / "real" / "raw" / "motifs"
OUT = REPO / "data" / "real" / "results" / "sae_enrichment"
OUT.mkdir(parents=True, exist_ok=True)

BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def parse_meme(path: Path) -> list[dict]:
    text = path.read_text()
    blocks = re.split(r"\nMOTIF\s+", text)
    motifs = []
    bg = np.array([0.25, 0.25, 0.25, 0.25])
    for blk in blocks[1:]:
        lines = blk.splitlines()
        header = lines[0].strip()
        name, _, rest = header.partition(" ")
        alt = rest.strip() or name
        lp_idx = None
        width = None
        for i, ln in enumerate(lines):
            if ln.startswith("letter-probability matrix:"):
                lp_idx = i
                m = re.search(r"w=\s*(\d+)", ln)
                if m:
                    width = int(m.group(1))
                break
        if lp_idx is None or width is None:
            continue
        rows = []
        for ln in lines[lp_idx + 1 : lp_idx + 1 + width]:
            parts = ln.strip().split()
            if len(parts) < 4:
                break
            rows.append([float(x) for x in parts[:4]])
        if len(rows) != width:
            continue
        pwm = np.asarray(rows, dtype=np.float32)
        pwm_p = (pwm + 1e-3) / (1.0 + 4e-3)
        lo = np.log2(pwm_p / bg).astype(np.float32)
        motifs.append(dict(name=name, alt=alt, width=width, log_odds=lo))
    return motifs


def onehot_batch(seqs: list[str], L: int) -> np.ndarray:
    """(N, L, 4) float32 one-hot; Ns are all-zero rows."""
    N = len(seqs)
    arr = np.zeros((N, L, 4), dtype=np.float32)
    for n, s in enumerate(seqs):
        for i, c in enumerate(s[:L]):
            j = BASE_IDX.get(c.upper())
            if j is not None:
                arr[n, i, j] = 1.0
    return arr


def motif_max_scores_batch(one_hot: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """For a single motif log-odds (w, 4), compute each sequence's max score
    (over forward + reverse-complement strands). Returns (N,)."""
    N, L, _ = one_hot.shape
    w = lo.shape[0]
    if L < w:
        return np.full(N, -1e9, dtype=np.float32)
    # Forward: out[n, i] = sum_{j, c} oh[n, i+j, c] * lo[j, c]
    n_pos = L - w + 1
    out_fwd = np.zeros((N, n_pos), dtype=np.float32)
    for j in range(w):
        out_fwd += one_hot[:, j : j + n_pos] @ lo[j]
    # Reverse complement: complement A<->T, C<->G, reverse j order
    rc_lo = lo[::-1, [3, 2, 1, 0]]
    out_rev = np.zeros((N, n_pos), dtype=np.float32)
    for j in range(w):
        out_rev += one_hot[:, j : j + n_pos] @ rc_lo[j]
    best = np.maximum(out_fwd.max(axis=1), out_rev.max(axis=1))
    return best


def rebuild_sae(state_dict: dict, variant: str, d_in: int, d_hidden: int, topk: int | None):
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
            z = torch.relu(gate)
            return self.dec(z), z
    model = L1SAE() if variant == "l1" else TopKSAE(topk)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def bh_fdr(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-tag", required=True)
    ap.add_argument("--dataset", default="region_type")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--top-k-seqs", type=int, default=50)
    ap.add_argument("--n-bg", type=int, default=500)
    ap.add_argument("--n-features", type=int, default=200)
    ap.add_argument(
        "--meme-file",
        default=str(MOTIFS / "JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt"),
    )
    args = ap.parse_args()

    # Load SAE
    json_path = SAE_DIR / f"{args.sae_tag}.json"
    pt_path = SAE_DIR / f"{args.sae_tag}.pt"
    with json_path.open() as f:
        meta = json.load(f)
    sd = torch.load(pt_path, map_location="cpu")
    mu = np.asarray(meta["mu"], dtype=np.float32)
    sigma = np.asarray(meta["sigma"], dtype=np.float32)
    model = rebuild_sae(sd, meta["variant"], meta["d_in"], meta["d_hidden"], meta.get("topk"))

    # Feature activations
    act = np.load(ACT_ROOT / args.dataset / "trained.npz", allow_pickle=True)
    X = act["hidden"][:, args.layer, :].astype(np.float32)
    Xn = (X - mu) / sigma
    with torch.no_grad():
        _, Z = model(torch.from_numpy(Xn))
    Z = Z.numpy()
    print(f"Z shape: {Z.shape}  mean L0 = {(Z > 0).sum(axis=-1).mean():.1f}")

    df = pd.read_parquet(ARAREG / f"{args.dataset}.parquet")
    df = df.set_index("sequence_id").loc[act["sequence_ids"].astype(str)].reset_index()
    seqs = df["sequence"].tolist()
    L = len(seqs[0])

    # Pre-encode sequences once
    print(f"[oh  ] encoding {len(seqs)} sequences to one-hot ({len(seqs)}×{L}×4) ...")
    t0 = time.time()
    one_hot = onehot_batch(seqs, L)
    print(f"[oh  ] done in {time.time()-t0:.1f}s; size {one_hot.nbytes/1e9:.2f} GB")

    motifs = parse_meme(Path(args.meme_file))
    M = len(motifs)
    print(f"[mot ] {M} motifs loaded")

    # Score matrix (N, M) — max log-odds per sequence per motif.
    N = len(seqs)
    dataset_cache = OUT / f"{args.dataset}_motif_scores.npy"
    scores_path = OUT / f"{args.sae_tag}_scores.npy"
    if dataset_cache.exists():
        print(f"[cache] loading dataset scores from {dataset_cache}")
        scores = np.load(dataset_cache)
    elif scores_path.exists():
        print(f"[cache] loading SAE-tagged scores from {scores_path}")
        scores = np.load(scores_path)
    else:
        scores = np.zeros((N, M), dtype=np.float32)
        t0 = time.time()
        for mi, mot in enumerate(motifs):
            scores[:, mi] = motif_max_scores_batch(one_hot, mot["log_odds"])
            if mi % 50 == 0:
                el = time.time() - t0
                rate = (mi + 1) / max(1e-9, el)
                remain = (M - mi - 1) / max(1e-9, rate)
                print(f"  motif {mi+1}/{M}  ({rate:.1f}/s, eta {remain/60:.1f} min)")
        np.save(dataset_cache, scores)
    print(f"[score] (N,M)={scores.shape}")

    # Enrichment tests
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(N, size=args.n_bg, replace=False)
    bg_scores = scores[bg_idx]  # (n_bg, M)
    feat_mass = Z.sum(axis=0)
    feat_order = np.argsort(-feat_mass)[: args.n_features]
    rows = []
    t0 = time.time()
    for fi, f in enumerate(feat_order):
        top_idx = np.argsort(-Z[:, f])[: args.top_k_seqs]
        top_scores = scores[top_idx]  # (k, M)
        for mi, mot in enumerate(motifs):
            try:
                u, p = mannwhitneyu(
                    top_scores[:, mi], bg_scores[:, mi], alternative="greater"
                )
            except Exception:
                p = 1.0
            rows.append(
                dict(
                    feature=int(f),
                    motif=mot["name"],
                    motif_alt=mot["alt"],
                    width=mot["width"],
                    top_mean=float(top_scores[:, mi].mean()),
                    top_max=float(top_scores[:, mi].max()),
                    bg_mean=float(bg_scores[:, mi].mean()),
                    diff=float(top_scores[:, mi].mean() - bg_scores[:, mi].mean()),
                    p=float(p),
                )
            )
        if fi % 40 == 0:
            el = time.time() - t0
            rate = (fi + 1) / max(1e-9, el)
            remain = (len(feat_order) - fi - 1) / max(1e-9, rate)
            print(f"  feat {fi+1}/{len(feat_order)} ({rate:.1f}/s, eta {remain/60:.1f} min)")

    res = pd.DataFrame(rows)
    res["q"] = bh_fdr(res["p"].to_numpy())
    res = res.sort_values(["feature", "q"])
    best = res.sort_values(["feature", "q"]).groupby("feature", as_index=False).first()

    out = OUT / f"{args.sae_tag}_feature_motif.csv"
    best_out = OUT / f"{args.sae_tag}_best_per_feature.csv"
    res.to_csv(out, index=False)
    best.to_csv(best_out, index=False)
    n_sig = int((best["q"] < 0.05).sum())
    print(f"Wrote {out} ({len(res)} rows); {n_sig}/{len(best)} features have q<0.05 best motif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
