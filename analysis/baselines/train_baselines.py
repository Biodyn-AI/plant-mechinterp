"""Train learned-from-scratch baselines on AraReg tasks (reviewer §3.1).

Baselines:
- CNN1: 1-layer CNN on one-hot DNA (64 filters, width 8, global max-pool).
- CNN3: 3-layer DanQ-style CNN (64-128-256 filters, with max-pool + dropout).
- BiLSTM: 1-layer bi-LSTM (hidden 128) on one-hot DNA.
- KmerMLP: MLP on k-mer frequency features (k ∈ {3, 4, 5, 6} concatenated).

Training protocol per dataset (real AraReg parquet):
- Use the frozen chrom-based splits: train (+val), test.
- 5 random-weight seeds per model.
- Early stopping on val accuracy with patience=5.
- Save all per-epoch and test metrics to data/real/results/baselines/{dataset}.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ARAREG = REPO / "data" / "real" / "arareg"
OUT = REPO / "data" / "real" / "results" / "baselines"
OUT.mkdir(parents=True, exist_ok=True)

PARQUET = {
    "region_type": ARAREG / "region_type.parquet",
    "splice": ARAREG / "splice.parquet",
    "tss": ARAREG / "tss.parquet",
    "promoter": ARAREG / "promoter.parquet",
}

BASE_COMP = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def onehot(seq: str, L: int) -> np.ndarray:
    """Returns (4, L) one-hot; unknown Ns are all-zero columns."""
    arr = np.zeros((4, L), dtype=np.float32)
    for i, ch in enumerate(seq[:L]):
        j = BASE_COMP.get(ch.upper(), 4)
        if j < 4:
            arr[j, i] = 1.0
    return arr


def kmer_vec(seq: str, k: int) -> np.ndarray:
    alpha = "ACGT"
    idx = {ch: i for i, ch in enumerate(alpha)}
    L = 4 ** k
    counts = np.zeros(L, dtype=np.float32)
    if len(seq) < k:
        return counts
    for i in range(len(seq) - k + 1):
        km = seq[i : i + k]
        if any(ch not in idx for ch in km):
            continue
        j = 0
        for ch in km:
            j = j * 4 + idx[ch]
        counts[j] += 1
    tot = counts.sum()
    if tot > 0:
        counts /= tot
    return counts


class CNN1(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv = nn.Conv1d(4, 64, kernel_size=8, padding=0)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        h = F.relu(self.conv(x))
        h = h.max(dim=-1).values
        return self.fc(h)


class CNN3(nn.Module):
    """DanQ/Basset-ish: three Conv1d blocks with max-pool + dropout + MLP head."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.c1 = nn.Conv1d(4, 64, 8, padding=3)
        self.c2 = nn.Conv1d(64, 128, 8, padding=3)
        self.c3 = nn.Conv1d(128, 256, 8, padding=3)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        h = self.pool(F.relu(self.c1(x)))
        h = self.pool(F.relu(self.c2(h)))
        h = self.pool(F.relu(self.c3(h)))
        h = h.max(dim=-1).values
        h = self.drop(h)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class BiLSTM(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4, hidden_size=hidden, num_layers=1, batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):  # x: (B, 4, L)
        x = x.transpose(1, 2)  # (B, L, 4)
        h, _ = self.lstm(x)
        h = h.max(dim=1).values
        return self.fc(h)


class KmerMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor


def iter_batches(X, y, batch_size, shuffle, rng):
    idx = np.arange(len(y))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = idx[i : i + batch_size]
        yield X[sel], y[sel]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def balance_subsample(df: pd.DataFrame, label_col: str, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    classes = df[label_col].unique()
    per_class = max(1, n // len(classes))
    pieces = []
    for c in classes:
        sub = df[df[label_col] == c]
        k = min(per_class, len(sub))
        pieces.append(sub.sample(n=k, random_state=seed))
    return pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device()

    if model_name == "cnn1":
        model = CNN1(n_classes)
    elif model_name == "cnn3":
        model = CNN3(n_classes)
    elif model_name == "bilstm":
        model = BiLSTM(n_classes)
    elif model_name == "kmer_mlp":
        model = KmerMLP(X_train.shape[1], n_classes)
    else:
        raise ValueError(model_name)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    rng = np.random.default_rng(seed)
    best_val = -1.0
    best_state = None
    patience_left = patience
    history = []

    def _eval(X_t, y_t):
        model.eval()
        with torch.no_grad():
            total, correct, loss = 0, 0, 0.0
            for i in range(0, len(y_t), 256):
                xb = X_t[i : i + 256]
                yb = y_t[i : i + 256]
                logits = model(xb)
                loss += F.cross_entropy(logits, yb, reduction="sum").item()
                correct += (logits.argmax(-1) == yb).sum().item()
                total += yb.numel()
            return correct / max(1, total), loss / max(1, total)

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        losses = []
        for xb, yb in iter_batches(
            X_train_t, y_train_t, batch_size=batch_size, shuffle=True, rng=rng
        ):
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_acc, val_loss = _eval(X_val_t, y_val_t)
        history.append(
            dict(epoch=ep, train_loss=float(np.mean(losses)), val_acc=val_acc,
                 val_loss=val_loss, time=time.time() - t0)
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_loss = _eval(X_test_t, y_test_t)

    # Macro F1 on test
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(y_test_t), 256):
            logits = model(X_test_t[i : i + 256])
            preds.append(logits.argmax(-1).cpu().numpy())
        preds = np.concatenate(preds)
    from sklearn.metrics import f1_score
    test_f1 = float(f1_score(y_test, preds, average="macro"))

    return dict(
        model=model_name,
        seed=seed,
        best_val_acc=float(best_val),
        test_acc=float(test_acc),
        test_f1=test_f1,
        test_loss=float(test_loss),
        history=history,
    )


def build_features(df: pd.DataFrame, kind: str, seq_len: int = 1024) -> np.ndarray:
    if kind == "onehot":
        return np.stack([onehot(s, seq_len) for s in df["sequence"].tolist()])
    if kind == "kmer":
        feats = [
            np.concatenate([kmer_vec(s, k) for k in (3, 4, 5, 6)])
            for s in df["sequence"].tolist()
        ]
        return np.stack(feats)
    raise ValueError(kind)


def run_dataset(
    dataset: str,
    models: list[str],
    *,
    subsample: int | None,
    seeds: list[int],
    epochs: int,
    batch_size: int,
) -> dict:
    print(f"\n=== dataset={dataset} ===")
    df = pd.read_parquet(PARQUET[dataset])
    df["label"] = df["label"].astype(str)
    if subsample is not None:
        df = balance_subsample(df, "label", subsample, seed=0)
    # Ensure class balance and split coverage
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if len(val_df) == 0 and len(train_df) > 0:
        # Random 10% of train → val if chrom split gave empty val
        val_df = train_df.sample(frac=0.1, random_state=0)
        train_df = train_df.drop(val_df.index)
    cls = sorted(df["label"].unique())
    c2i = {c: i for i, c in enumerate(cls)}
    y_tr = train_df["label"].map(c2i).to_numpy(dtype=np.int64)
    y_va = val_df["label"].map(c2i).to_numpy(dtype=np.int64)
    y_te = test_df["label"].map(c2i).to_numpy(dtype=np.int64)
    print(f"  classes={cls}  train={len(y_tr)} val={len(y_va)} test={len(y_te)}")

    # Pre-build features once per kind
    features_cache: dict[str, dict[str, np.ndarray]] = {}

    results: dict = {"dataset": dataset, "classes": cls, "runs": []}
    for m in models:
        kind = "kmer" if m == "kmer_mlp" else "onehot"
        if kind not in features_cache:
            print(f"  building {kind} features")
            t0 = time.time()
            features_cache[kind] = {
                "train": build_features(train_df, kind),
                "val": build_features(val_df, kind),
                "test": build_features(test_df, kind),
            }
            print(f"  {kind} features built in {time.time()-t0:.1f}s")
        feats = features_cache[kind]
        for seed in seeds:
            t0 = time.time()
            r = train_model(
                m,
                feats["train"], y_tr,
                feats["val"], y_va,
                feats["test"], y_te,
                n_classes=len(cls),
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
            )
            r["dataset"] = dataset
            r["time"] = time.time() - t0
            results["runs"].append(r)
            print(f"    {m} seed={seed}: val_acc={r['best_val_acc']:.3f} "
                  f"test_acc={r['test_acc']:.3f} test_f1={r['test_f1']:.3f} "
                  f"({r['time']:.1f}s)")

    # Aggregate
    summary = {}
    for m in models:
        runs = [r for r in results["runs"] if r["model"] == m]
        if not runs:
            continue
        accs = [r["test_acc"] for r in runs]
        f1s = [r["test_f1"] for r in runs]
        summary[m] = {
            "test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs, ddof=1)),
            "test_acc_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            "test_f1_mean": float(np.mean(f1s)),
            "test_f1_std": float(np.std(f1s, ddof=1)),
            "n_seeds": len(runs),
        }
    results["summary"] = summary

    out_path = OUT / f"{dataset}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"  wrote {out_path}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(PARQUET.keys()))
    ap.add_argument(
        "--models", nargs="+",
        default=["cnn1", "cnn3", "bilstm", "kmer_mlp"],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    run_dataset(
        args.dataset, args.models,
        subsample=args.subsample, seeds=args.seeds,
        epochs=args.epochs, batch_size=args.batch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
