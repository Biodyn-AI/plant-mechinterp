"""Per-attention-head ablation study — reviewer §2.2.

For the trained Plant-DnaGemma on the splice or TSS task:

1. Train a linear probe once at layer 7 on the frozen cache (train+val).
2. For each of the 12 × 12 = 144 heads:
     - Zero out that head's contribution (mask attention weights → 0) during
       the forward pass.
     - Recompute layer-7 activations on the test set.
     - Measure test accuracy drop relative to the baseline (no ablation).
3. Also ablate entire MLP blocks per layer.

Writes data/real/results/mechanistic/head_ablations_{task}.json with per-head
effects and a ranked list.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODEL_DIR = REPO / "models" / "plant-dnagemma-BPE"
ARAREG = REPO / "data" / "real" / "arareg"
ACT_ROOT = REPO / "data" / "real" / "activations"
OUT = REPO / "data" / "real" / "results" / "mechanistic"
OUT.mkdir(parents=True, exist_ok=True)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_probe(dataset: str, layer: int) -> tuple[LogisticRegression, StandardScaler, list]:
    cache = np.load(ACT_ROOT / dataset / "trained.npz", allow_pickle=True)
    H = cache["hidden"][:, layer, :].astype(np.float32)
    y = cache["labels"].astype(str)
    split = cache["split"].astype(str)
    mask = np.isin(split, ["train", "val"])
    sc = StandardScaler().fit(H[mask])
    Xs = sc.transform(H[mask])
    clf = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
    clf.fit(Xs, y[mask])
    classes = clf.classes_.tolist()
    return clf, sc, classes


def evaluate(clf, sc, classes, H_test, y_test):
    Xs = sc.transform(H_test)
    pred = clf.predict(Xs)
    return float(accuracy_score(y_test, pred)), float(
        f1_score(y_test, pred, average="macro")
    )


def install_head_ablation(model, target_layer: int, target_head: int):
    """Patch the target layer's attention output projection so that the
    contribution of a single head is zeroed. We intercept the attention
    output by hooking the `o_proj` call: the head dimension is on the last
    axis before o_proj; we zero that slice.
    """
    # Gemma's GemmaAttention: out = o_proj(concat(heads)).
    # We hook the self_attn module; compute output manually then replace head.
    layer = model.layers[target_layer]
    attn = layer.self_attn
    head_dim = model.config.head_dim  # 256 for plant-dnagemma-BPE
    n_heads = model.config.num_attention_heads
    # Hook target: we'll wrap o_proj to receive (B, T, n_heads*head_dim)
    orig_fwd = attn.o_proj.forward
    def new_fwd(x, *args, **kwargs):
        # x: (B, T, n_heads * head_dim)
        B, T, D = x.shape
        x = x.reshape(B, T, n_heads, head_dim).clone()
        x[:, :, target_head, :] = 0.0
        x = x.reshape(B, T, D)
        return orig_fwd(x, *args, **kwargs)
    attn.o_proj.forward = new_fwd
    def restore():
        attn.o_proj.forward = orig_fwd
    return restore


def install_mlp_ablation(model, target_layer: int):
    layer = model.layers[target_layer]
    orig_fwd = layer.mlp.forward
    def new_fwd(x, *args, **kwargs):
        return torch.zeros_like(x)
    layer.mlp.forward = new_fwd
    def restore():
        layer.mlp.forward = orig_fwd
    return restore


@torch.no_grad()
def extract_pooled(model, tok, seqs, device, layer):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[layer]
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="splice", choices=["splice", "tss", "region_type"])
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    device = pick_device()
    print(f"[dev ] {device}")
    model = AutoModel.from_pretrained(MODEL_DIR).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Load splice test set
    df = pd.read_parquet(ARAREG / f"{args.dataset}.parquet")
    df = df[df["split"] == "test"].copy()
    # Balanced subsample
    df = df.groupby("label").head(args.n_test // df["label"].nunique()).reset_index(drop=True)
    print(f"Evaluating on {len(df)} test sequences")

    clf, sc, classes = train_probe(args.dataset, args.layer)
    print(f"Probe classes: {classes}")

    # Baseline (no ablation)
    seqs = df["sequence"].tolist()
    y_test = df["label"].astype(str).to_numpy()
    bs = args.batch
    def run_all():
        pooled = []
        for i in range(0, len(seqs), bs):
            pooled.append(extract_pooled(model, tok, seqs[i : i + bs], device, args.layer))
        return np.concatenate(pooled, axis=0)
    H = run_all()
    base_acc, base_f1 = evaluate(clf, sc, classes, H, y_test)
    print(f"Baseline test acc={base_acc:.3f} f1={base_f1:.3f}")

    results = {
        "dataset": args.dataset,
        "layer": args.layer,
        "n_test": len(df),
        "baseline_acc": base_acc,
        "baseline_f1": base_f1,
        "heads": [],
        "mlps": [],
    }

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    t0 = time.time()
    for li in range(n_layers):
        for hi in range(n_heads):
            restore = install_head_ablation(model, li, hi)
            try:
                H2 = run_all()
                acc, f1 = evaluate(clf, sc, classes, H2, y_test)
            finally:
                restore()
            drop = base_acc - acc
            results["heads"].append(
                {"layer": li, "head": hi, "acc": acc, "f1": f1, "acc_drop": drop}
            )
            print(f"  L{li:2d}H{hi:2d}: acc={acc:.3f} (Δ={drop:+.3f})")
        # After completing a layer, ablate its MLP
        restore = install_mlp_ablation(model, li)
        try:
            H2 = run_all()
            acc, f1 = evaluate(clf, sc, classes, H2, y_test)
        finally:
            restore()
        drop = base_acc - acc
        results["mlps"].append({"layer": li, "acc": acc, "f1": f1, "acc_drop": drop})
        print(f"  L{li:2d} MLP: acc={acc:.3f} (Δ={drop:+.3f})")
        el = time.time() - t0
        rate = ((li + 1) * (n_heads + 1)) / max(1e-9, el)
        remain = ((n_layers - li - 1) * (n_heads + 1)) / max(1e-9, rate)
        print(f"  layer {li+1}/{n_layers} done — ETA {remain/60:.1f} min")

    # Ranking
    heads_sorted = sorted(results["heads"], key=lambda d: -d["acc_drop"])
    results["top_heads"] = heads_sorted[:20]

    out_path = OUT / f"head_ablations_{args.dataset}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
