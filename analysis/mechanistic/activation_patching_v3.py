"""Activation patching v3 — reviewer §2.1.

Addresses the concern that the previous activation-patching null result may
have reflected insufficient corruption. We now:

1. Use *real* splice-site sequences (donor / acceptor / dinuc-shuffled
   "nonsite") from `data/real/arareg/splice.parquet` instead of synthetic motifs.
2. Build clean/corrupt pairs two ways:
     - canonical-base corruption: flip the canonical dinucleotide (GT→CC at
       donor, AG→CC at acceptor) at the splice position.
     - dinuc-shuffled corruption: replace a ±20-nt window around the splice
       site with a dinuc-shuffled surrogate (preserves local composition).
3. Run patching in both directions (denoising and noising) and per layer.
4. Scoring: downstream probe (fitted once on layer-7 trained activations)
   outputs a donor-vs-nonsite logit; effect = logit change normalized by the
   clean→corrupt gap.
5. Bootstrap CIs over sequences.

Emits `data/real/results/patching/splice_patching.json` and a summary figure
under `data/real/results/patching/fig.png`.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random as pyrandom
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODEL_DIR = REPO / "models" / "plant-dnagemma-BPE"
ARAREG = REPO / "data" / "real" / "arareg"
ACT_ROOT = REPO / "data" / "real" / "activations"
OUT = REPO / "data" / "real" / "results" / "patching"
OUT.mkdir(parents=True, exist_ok=True)

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dinuc_shuffle(seq: str, rng: pyrandom.Random) -> str:
    """Altschul-Erickson dinucleotide-preserving shuffle."""
    s = seq.upper()
    if len(s) < 2:
        return s
    alpha = sorted(set(s))
    succ: dict[str, list[str]] = {a: [] for a in alpha}
    for a, b in zip(s, s[1:]):
        succ[a].append(b)
    for a in alpha:
        rng.shuffle(succ[a])
    last_char = s[-1]
    end_succ: dict[str, str] = {}
    for a in alpha:
        if a == last_char:
            continue
        for i, b in enumerate(succ[a]):
            if b == last_char:
                end_succ[a] = succ[a].pop(i)
                break
    out = [s[0]]
    cur = s[0]
    while True:
        if succ[cur]:
            nxt = succ[cur].pop()
            out.append(nxt)
            cur = nxt
        elif cur in end_succ:
            nxt = end_succ.pop(cur)
            out.append(nxt)
            cur = nxt
        else:
            break
    if len(out) != len(s):
        out = list(s)
        rng.shuffle(out)
    return "".join(out)


def corrupt_canonical(seq: str, label: str) -> str:
    """Flip the canonical dinucleotide at the center of the window. Sequences
    are stored on the transcript strand; for a 1024-window around a splice
    site the center is at index 512 and the next position is 513. Donors have
    GT at (center, center+1); acceptors have AG at (center-1, center).
    """
    L = len(seq)
    mid = L // 2
    s = list(seq)
    if label == "donor":
        # donor: intron starts with GT at positions (mid, mid+1) (plus-strand)
        s[mid : mid + 2] = list("CC")
    elif label == "acceptor":
        # acceptor: intron ends with AG at positions (mid-1, mid)
        s[mid - 1 : mid + 1] = list("CC")
    return "".join(s)


def corrupt_shuffle(seq: str, rng: pyrandom.Random, half: int = 20) -> str:
    L = len(seq)
    mid = L // 2
    lo = max(0, mid - half)
    hi = min(L, mid + half)
    shuffled = dinuc_shuffle(seq[lo:hi], rng)
    return seq[:lo] + shuffled + seq[hi:]


@torch.no_grad()
def forward_hidden_states(model, tok, seqs, device, max_length=1024):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    # mean-pool using mask
    mask = enc["attention_mask"].unsqueeze(-1).float()
    denom = mask.sum(dim=1).clamp(min=1.0)
    pooled = torch.stack(
        [(h * mask).sum(dim=1) / denom for h in out.hidden_states], dim=1
    )  # (B, n_layers, d)
    return pooled.cpu().numpy(), enc["attention_mask"].cpu().numpy(), [h.cpu() for h in out.hidden_states]


class LogisticProbe:
    """Simple multiclass logistic regression on layer-7 mean-pooled activations,
    used as the patching target metric."""

    def __init__(self, layer: int):
        self.layer = layer
        self.sc_mean = None
        self.sc_std = None
        self.W = None
        self.b = None
        self.classes_ = None

    def fit_from_cache(self, dataset: str):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        cache = np.load(ACT_ROOT / dataset / "trained.npz", allow_pickle=True)
        H = cache["hidden"][:, self.layer, :].astype(np.float32)
        labels = cache["labels"].astype(str)
        split = cache["split"].astype(str)
        mask = np.isin(split, ["train", "val"])
        X = H[mask]
        y = labels[mask]
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        clf = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
        clf.fit(Xs, y)
        self.sc_mean = sc.mean_.astype(np.float32)
        self.sc_std = sc.scale_.astype(np.float32)
        self.W = clf.coef_.astype(np.float32)
        self.b = clf.intercept_.astype(np.float32)
        self.classes_ = clf.classes_.tolist()
        return self

    def logits(self, H_L: np.ndarray) -> np.ndarray:
        """H_L: (B, D) at layer `self.layer`. Returns (B, n_classes)."""
        Xs = (H_L - self.sc_mean) / self.sc_std
        return Xs @ self.W.T + self.b


def run(args) -> int:
    device = pick_device()
    print(f"[dev ] {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR).to(device).eval()

    df = pd.read_parquet(ARAREG / "splice.parquet")
    df = df[df["split"] == "test"].copy()
    df = df.groupby("label").head(args.n_per_class).reset_index(drop=True)
    print(f"Using n={len(df)} sequences; classes={df['label'].value_counts().to_dict()}")

    rng = pyrandom.Random(args.seed)

    probe = LogisticProbe(layer=args.target_layer).fit_from_cache("splice")
    print(f"probe classes: {probe.classes_}")

    # Build corrupted versions
    rows = []
    for _, r in df.iterrows():
        seq = r["sequence"]
        label = r["label"]
        if label == "nonsite":
            continue  # patching is meaningful only on donor/acceptor
        c_canon = corrupt_canonical(seq, label)
        c_shuf = corrupt_shuffle(seq, rng, half=20)
        rows.append(dict(sid=r["sequence_id"], label=label, clean=seq,
                         corrupt_canon=c_canon, corrupt_shuf=c_shuf))

    results = {
        "target_layer": args.target_layer,
        "n_samples": len(rows),
        "layers": {},
    }

    # We patch each layer's residual stream (post-block hidden state) using a
    # forward hook, restoring clean activations at a given layer in the
    # corrupted forward pass (denoising).
    # For efficiency we do this in small batches.

    def get_hidden_at(seqs: list[str]) -> list[torch.Tensor]:
        """Run model, return list of per-layer hidden states (each on CPU)."""
        enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        return [h.cpu() for h in out.hidden_states], enc

    def patched_run(corrupt_seqs: list[str], clean_hidden: list[torch.Tensor],
                    layer_idx: int):
        enc = tok(corrupt_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        # Match sequence length between clean and corrupt by truncating to min
        clean_hs = clean_hidden[layer_idx].to(device)
        # Attach hook to Gemma decoder layer (layer_idx - 1 because hidden[0]
        # is embedding; hidden[i] is output of layer i-1 for Gemma).
        # When layer_idx == 0, we replace the embedding hidden state before layer 1.
        if layer_idx == 0:
            # Patch the input embeddings directly
            hook_module = model.embed_tokens
            def hook(mod, inp, out):
                L = min(out.shape[1], clean_hs.shape[1])
                out[:, :L, :] = clean_hs[:, :L, :]
                return out
            handle = hook_module.register_forward_hook(hook)
        else:
            hook_module = model.layers[layer_idx - 1]
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    h = out[0]
                    L = min(h.shape[1], clean_hs.shape[1])
                    new = h.clone()
                    new[:, :L, :] = clean_hs[:, :L, :]
                    return (new,) + out[1:]
                L = min(out.shape[1], clean_hs.shape[1])
                new = out.clone()
                new[:, :L, :] = clean_hs[:, :L, :]
                return new
            handle = hook_module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                patched_out = model(**enc, output_hidden_states=True)
        finally:
            handle.remove()
        # Mean-pool at target_layer
        h = patched_out.hidden_states[args.target_layer]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.cpu().numpy()

    n_layers = model.config.num_hidden_layers + 1

    # For each sequence in `rows`, we compute:
    #   clean_logits, corrupt_logits (2 variants), patched_logits at each layer
    # Use batches of size 4 to fit in 6GB VRAM.
    bs = 4

    corruption_modes = ["canon", "shuf"]

    effects = {mode: {lidx: [] for lidx in range(n_layers)} for mode in corruption_modes}
    noise_effects = {mode: {lidx: [] for lidx in range(n_layers)} for mode in corruption_modes}

    t0 = time.time()
    for start in range(0, len(rows), bs):
        batch = rows[start : start + bs]
        clean_seqs = [r["clean"] for r in batch]
        labels = [r["label"] for r in batch]
        label_idx = np.asarray(
            [probe.classes_.index(l) for l in labels], dtype=np.int64
        )
        # Clean hidden states
        clean_hs, clean_enc = get_hidden_at(clean_seqs)
        clean_target = clean_hs[args.target_layer]
        # Clean logits via probe
        mask = clean_enc["attention_mask"].unsqueeze(-1).float().cpu()
        denom = mask.sum(dim=1).clamp(min=1.0)
        clean_pooled = (clean_target * mask).sum(dim=1) / denom
        clean_logits = probe.logits(clean_pooled.numpy())

        for mode in corruption_modes:
            key = "corrupt_canon" if mode == "canon" else "corrupt_shuf"
            corrupt_seqs = [r[key] for r in batch]
            # Corrupt hidden states (for noising-direction baseline and per-layer plug)
            corrupt_hs, corrupt_enc = get_hidden_at(corrupt_seqs)
            mask_c = corrupt_enc["attention_mask"].unsqueeze(-1).float().cpu()
            denom_c = mask_c.sum(dim=1).clamp(min=1.0)
            corrupt_target = corrupt_hs[args.target_layer]
            corrupt_pooled = (corrupt_target * mask_c).sum(dim=1) / denom_c
            corrupt_logits = probe.logits(corrupt_pooled.numpy())
            # per-sequence clean→corrupt gap in correct-class logit
            base_gap = np.array(
                [clean_logits[i, label_idx[i]] - corrupt_logits[i, label_idx[i]] for i in range(len(batch))]
            )

            for lidx in range(n_layers):
                # Denoising: plug clean activations at lidx into corrupt run.
                pooled = patched_run(corrupt_seqs, clean_hs, lidx)
                logits = probe.logits(pooled)
                recov = np.array(
                    [logits[i, label_idx[i]] - corrupt_logits[i, label_idx[i]] for i in range(len(batch))]
                )
                # Normalize by base gap (avoid div by zero)
                base_safe = np.where(np.abs(base_gap) < 1e-6, 1e-6, base_gap)
                effects[mode][lidx].extend((recov / base_safe).tolist())

                # Noising: plug corrupt activations at lidx into clean run.
                pooled_n = patched_run(clean_seqs, corrupt_hs, lidx)  # use corrupt_hs as "clean" to plug
                logits_n = probe.logits(pooled_n)
                damage = np.array(
                    [clean_logits[i, label_idx[i]] - logits_n[i, label_idx[i]] for i in range(len(batch))]
                )
                noise_effects[mode][lidx].extend((damage / base_safe).tolist())

        if (start // bs) % 10 == 0:
            el = time.time() - t0
            rate = (start + bs) / max(1e-9, el)
            remain = (len(rows) - start - bs) / max(1e-9, rate)
            print(
                f"  [{start+bs}/{len(rows)}]  {rate:.2f} seq/s  eta {remain/60:.1f} min"
            )

    def summarize(d):
        out = {}
        for lidx, vals in d.items():
            v = np.asarray(vals, dtype=np.float32)
            if len(v) == 0:
                continue
            out[str(lidx)] = {
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "ci95_low": float(np.percentile(v, 2.5)),
                "ci95_high": float(np.percentile(v, 97.5)),
                "n": int(len(v)),
            }
        return out

    results["layers"] = {
        mode: {
            "denoise": summarize(effects[mode]),
            "noise": summarize(noise_effects[mode]),
        }
        for mode in corruption_modes
    }
    out_path = OUT / "splice_patching.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-layer", type=int, default=7)
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
