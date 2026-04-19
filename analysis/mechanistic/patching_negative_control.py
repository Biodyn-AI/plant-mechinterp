"""Negative-control for activation patching v3 (reviewer §2.1).

For the same splice-site sequences used in patching_v3, we corrupt at a
*random non-motif* position (offset 150 nt from the center, away from the
splice site). If the original patching effect was caused by the splice-site
motif, the negative-control corruption should produce a much smaller logit
change — a specificity test.

Compares peak effects side by side with the splice-centered case.
"""
from __future__ import annotations

import argparse
import json
import random as pyrandom
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODEL_DIR = REPO / "models" / "plant-dnagemma-BPE"
ARAREG = REPO / "data" / "real" / "arareg"
ACT_ROOT = REPO / "data" / "real" / "activations"
OUT = REPO / "data" / "real" / "results" / "patching"

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMP)[::-1]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dinuc_shuffle(seq, rng):
    s = seq.upper()
    if len(s) < 2:
        return s
    alpha = sorted(set(s))
    succ = {a: [] for a in alpha}
    for a, b in zip(s, s[1:]):
        succ[a].append(b)
    for a in alpha:
        rng.shuffle(succ[a])
    last = s[-1]
    end_succ = {}
    for a in alpha:
        if a == last:
            continue
        for i, b in enumerate(succ[a]):
            if b == last:
                end_succ[a] = succ[a].pop(i)
                break
    out = [s[0]]
    cur = s[0]
    while True:
        if succ[cur]:
            nxt = succ[cur].pop()
            out.append(nxt); cur = nxt
        elif cur in end_succ:
            nxt = end_succ.pop(cur); out.append(nxt); cur = nxt
        else:
            break
    if len(out) != len(s):
        out = list(s); rng.shuffle(out)
    return "".join(out)


def corrupt_random_position(seq, rng, half=20, offset=150):
    """Corrupt a 40-nt window offset 150 nt from the center (away from splice
    site, which is at len/2). Dinuc shuffle within that window."""
    L = len(seq)
    mid = L // 2
    # Flip sign per-sequence to mix offsets
    sign = rng.choice([-1, 1])
    center = mid + sign * offset
    lo = max(0, center - half)
    hi = min(L, center + half)
    shuffled = dinuc_shuffle(seq[lo:hi], rng)
    return seq[:lo] + shuffled + seq[hi:]


class LogisticProbe:
    def __init__(self, layer):
        self.layer = layer
    def fit_from_cache(self, dataset):
        cache = np.load(ACT_ROOT / dataset / "trained.npz", allow_pickle=True)
        H = cache["hidden"][:, self.layer, :].astype(np.float32)
        labels = cache["labels"].astype(str)
        split = cache["split"].astype(str)
        mask = np.isin(split, ["train", "val"])
        sc = StandardScaler().fit(H[mask])
        Xs = sc.transform(H[mask])
        clf = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
        clf.fit(Xs, labels[mask])
        self.sc_mean = sc.mean_.astype(np.float32)
        self.sc_std = sc.scale_.astype(np.float32)
        self.W = clf.coef_.astype(np.float32)
        self.b = clf.intercept_.astype(np.float32)
        self.classes_ = clf.classes_.tolist()
        return self
    def logits(self, H_L):
        Xs = (H_L - self.sc_mean) / self.sc_std
        return Xs @ self.W.T + self.b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-layer", type=int, default=7)
    args = ap.parse_args()

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR).to(device).eval()
    probe = LogisticProbe(layer=args.target_layer).fit_from_cache("splice")

    df = pd.read_parquet(ARAREG / "splice.parquet")
    df = df[df["split"] == "test"].copy()
    df = df.groupby("label").head(args.n_per_class).reset_index(drop=True)
    rng = pyrandom.Random(args.seed)

    rows = []
    for _, r in df.iterrows():
        if r["label"] == "nonsite":
            continue
        seq = r["sequence"]
        c_rand = corrupt_random_position(seq, rng)
        rows.append(dict(sid=r["sequence_id"], label=r["label"],
                         clean=seq, corrupt_rand=c_rand))
    print(f"n_samples: {len(rows)}")

    def get_hidden(seqs):
        enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        return [h.cpu() for h in out.hidden_states], enc

    def patched_run(seqs, clean_hs_all, layer_idx):
        enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        clean_hs = clean_hs_all[layer_idx].to(device)
        if layer_idx == 0:
            hook_module = model.embed_tokens
            def hook(mod, inp, out):
                L = min(out.shape[1], clean_hs.shape[1])
                out[:, :L, :] = clean_hs[:, :L, :]; return out
        else:
            hook_module = model.layers[layer_idx - 1]
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    h = out[0]; L = min(h.shape[1], clean_hs.shape[1])
                    new = h.clone(); new[:, :L, :] = clean_hs[:, :L, :]
                    return (new,) + out[1:]
                L = min(out.shape[1], clean_hs.shape[1])
                new = out.clone(); new[:, :L, :] = clean_hs[:, :L, :]
                return new
        handle = hook_module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                p_out = model(**enc, output_hidden_states=True)
        finally:
            handle.remove()
        h = p_out.hidden_states[args.target_layer]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return ((h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).cpu().numpy()

    bs = 4
    n_layers = model.config.num_hidden_layers + 1
    effects = {li: [] for li in range(n_layers)}
    noise_effects = {li: [] for li in range(n_layers)}

    t0 = time.time()
    for start in range(0, len(rows), bs):
        batch = rows[start:start+bs]
        clean_seqs = [r["clean"] for r in batch]
        label_idx = np.asarray(
            [probe.classes_.index(r["label"]) for r in batch], dtype=np.int64
        )
        clean_hs, clean_enc = get_hidden(clean_seqs)
        clean_pooled_l7 = clean_hs[args.target_layer]
        mask = clean_enc["attention_mask"].unsqueeze(-1).float().cpu()
        denom = mask.sum(dim=1).clamp(min=1.0)
        clean_pool = (clean_pooled_l7 * mask).sum(dim=1) / denom
        clean_logits = probe.logits(clean_pool.numpy())

        corrupt_seqs = [r["corrupt_rand"] for r in batch]
        corrupt_hs, corrupt_enc = get_hidden(corrupt_seqs)
        mask_c = corrupt_enc["attention_mask"].unsqueeze(-1).float().cpu()
        denom_c = mask_c.sum(dim=1).clamp(min=1.0)
        corr_pool_l7 = corrupt_hs[args.target_layer]
        corr_pool = (corr_pool_l7 * mask_c).sum(dim=1) / denom_c
        corr_logits = probe.logits(corr_pool.numpy())
        base_gap = np.array([clean_logits[i, label_idx[i]] - corr_logits[i, label_idx[i]]
                             for i in range(len(batch))])

        for lidx in range(n_layers):
            pooled = patched_run(corrupt_seqs, clean_hs, lidx)
            logits = probe.logits(pooled)
            recov = np.array([logits[i, label_idx[i]] - corr_logits[i, label_idx[i]]
                              for i in range(len(batch))])
            safe = np.where(np.abs(base_gap) < 1e-6, 1e-6, base_gap)
            effects[lidx].extend((recov / safe).tolist())

            pooled_n = patched_run(clean_seqs, corrupt_hs, lidx)
            logits_n = probe.logits(pooled_n)
            damage = np.array([clean_logits[i, label_idx[i]] - logits_n[i, label_idx[i]]
                              for i in range(len(batch))])
            noise_effects[lidx].extend((damage / safe).tolist())
        if (start // bs) % 10 == 0:
            el = time.time() - t0
            rate = (start + bs) / max(1e-9, el)
            remain = (len(rows) - start - bs) / max(1e-9, rate)
            print(f"  [{start+bs}/{len(rows)}] {rate:.2f} seq/s  eta {remain/60:.1f} min")

    def summarize(d):
        return {
            str(lidx): {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
            for lidx, vals in d.items() if vals
        }

    result = dict(
        control="random_non_motif_position",
        n_samples=len(rows),
        target_layer=args.target_layer,
        denoise=summarize(effects),
        noise=summarize(noise_effects),
    )
    out_path = OUT / "splice_patching_negative_control.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")
    print("\nSummary (noise direction, per layer):")
    for l in sorted(result["noise"], key=int):
        r = result["noise"][l]
        print(f"  L{int(l):2d}: mean={r['mean']:+.2f}")


if __name__ == "__main__":
    main()
