"""Extract mean-pooled hidden-state activations from Plant-DnaGemma.

For each parquet dataset under data/real/arareg/ (and optionally multispecies),
produces a cache in data/real/activations/{dataset}/{tag}.npz with keys:
    hidden     : (n_seq, n_layers=13, hidden=768) float32
    labels     : (n_seq,) object
    sequence_ids: (n_seq,) object
    gc         : (n_seq,) float32
    split      : (n_seq,) object

`tag` in {"trained", "random"} — the random-init variant is the published
control baseline. Both share the same tokenizer and architecture.

Usage:
    python analysis/probing/extract_activations.py \
        --dataset region_type --tag trained --batch 16
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODEL_DIR = REPO / "models" / "plant-dnagemma-BPE"
ARAREG = REPO / "data" / "real" / "arareg"
MSPEC = REPO / "data" / "real" / "multispecies"
OUT_ROOT = REPO / "data" / "real" / "activations"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "region_type": ARAREG / "region_type.parquet",
    "splice": ARAREG / "splice.parquet",
    "tss": ARAREG / "tss.parquet",
    "promoter": ARAREG / "promoter.parquet",
    "multispecies": MSPEC / "windows.parquet",
    "multispecies_gc_matched": MSPEC / "windows_gc_matched.parquet",
    "multispecies_heldout": MSPEC / "windows_heldout.parquet",
}


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(tag: str) -> tuple[AutoModel, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tag == "trained":
        m = AutoModel.from_pretrained(MODEL_DIR)
    elif tag == "random":
        # Same architecture, untrained weights (matches paper's "random" baseline).
        cfg = AutoConfig.from_pretrained(MODEL_DIR)
        m = AutoModel.from_config(cfg)
    else:
        raise ValueError(f"unknown tag {tag}")
    m.eval()
    return m, tok


def balance_subsample(df: pd.DataFrame, label_col: str, n: int, seed: int = 0) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    classes = df[label_col].unique()
    per_class = max(1, n // len(classes))
    pieces = []
    for c in classes:
        sub = df[df[label_col] == c]
        k = min(per_class, len(sub))
        pieces.append(sub.sample(n=k, random_state=seed))
    out = pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


@torch.no_grad()
def extract(
    dataset: str,
    tag: str,
    batch_size: int = 16,
    max_seq: int | None = None,
    max_length: int = 1024,
    label_col: str = "label",
) -> None:
    src = DATASETS[dataset]
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_parquet(src)

    # Optional balanced sub-sample (for speed on very large datasets).
    if max_seq is not None and len(df) > max_seq:
        df = balance_subsample(df, label_col=label_col, n=max_seq)

    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.npz"
    if out_path.exists():
        print(f"[skip] {out_path} already exists ({out_path.stat().st_size/1e6:.1f} MB)")
        return

    device = pick_device()
    print(f"[load] model tag={tag}")
    model, tok = load_model(tag)
    model = model.to(device)

    n = len(df)
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding
    hidden = model.config.hidden_size
    print(f"[run ] dataset={dataset} tag={tag} n={n} n_layers={n_layers} hidden={hidden} device={device}")

    arr = np.empty((n, n_layers, hidden), dtype=np.float32)
    labels = df[label_col].astype(str).to_numpy() if label_col in df.columns else np.array(["?"] * n)
    ids = df["sequence_id"].to_numpy() if "sequence_id" in df.columns else np.arange(n).astype(str)
    gc = df["gc"].to_numpy().astype(np.float32) if "gc" in df.columns else np.zeros(n, np.float32)
    split = df["split"].to_numpy() if "split" in df.columns else np.array(["?"] * n)

    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        seqs = df["sequence"].iloc[start:end].tolist()
        enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        for li, hs in enumerate(out.hidden_states):
            pooled = (hs * mask).sum(dim=1) / denom  # (B, hidden)
            arr[start:end, li] = pooled.detach().to("cpu", torch.float32).numpy()
        if (start // batch_size) % 50 == 0:
            el = time.time() - t0
            rate = (end) / max(1e-9, el)
            remain = (n - end) / max(1e-9, rate)
            print(f"  [{end:6d}/{n:6d}] {rate:5.1f} seq/s, eta {remain/60:.1f} min")

    np.savez_compressed(
        out_path,
        hidden=arr,
        labels=labels,
        sequence_ids=ids,
        gc=gc,
        split=split,
    )
    print(f"[save] {out_path}: {arr.shape} -> {out_path.stat().st_size/1e6:.1f} MB  "
          f"total time {(time.time()-t0)/60:.1f} min")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--tag", choices=["trained", "random"], default="trained")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=None)
    ap.add_argument("--label-col", default="label")
    args = ap.parse_args()
    extract(
        dataset=args.dataset,
        tag=args.tag,
        batch_size=args.batch,
        max_seq=args.max_seq,
        label_col=args.label_col,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
