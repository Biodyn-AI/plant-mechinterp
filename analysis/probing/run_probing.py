"""Run probing on real-data activations with baselines.

For each dataset under data/real/activations/{dataset}/, computes:

1. Per-layer 5-fold stratified CV accuracy and macro-F1 on a balanced training
   subset (train+val splits). Reported with mean ± 1 SD and 95% bootstrap CI.
2. Chromosome-based held-out test accuracy (using the frozen split column).
3. Same for both 'trained' and 'random' variants.
4. Baselines computed directly from sequence features:
   - k-mer frequency logistic regression (k ∈ {3,4,5})
   - GC-content only logistic regression
   - Shuffle label control
5. Bootstrap p-values for "trained vs random", "trained vs best-kmer".

Outputs data/real/results/probing/{dataset}.json and a CSV summary.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ACT_ROOT = REPO / "data" / "real" / "activations"
ARAREG = REPO / "data" / "real" / "arareg"
MSPEC = REPO / "data" / "real" / "multispecies"
OUT_DIR = REPO / "data" / "real" / "results" / "probing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET = {
    "region_type": ARAREG / "region_type.parquet",
    "splice": ARAREG / "splice.parquet",
    "tss": ARAREG / "tss.parquet",
    "promoter": ARAREG / "promoter.parquet",
    "multispecies": MSPEC / "windows.parquet",
    "multispecies_gc_matched": MSPEC / "windows_gc_matched.parquet",
    "multispecies_heldout": MSPEC / "windows_heldout.parquet",
}


def kmer_counts(seq: str, k: int) -> np.ndarray:
    """Sparse-ish k-mer frequency vector over 4^k alphabet (A,C,G,T)."""
    alpha = "ACGT"
    idx = {ch: i for i, ch in enumerate(alpha)}
    L = 4 ** k
    counts = np.zeros(L, dtype=np.float32)
    if len(seq) < k:
        return counts
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if any(ch not in idx for ch in kmer):
            continue
        j = 0
        for ch in kmer:
            j = j * 4 + idx[ch]
        counts[j] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def _cv_logreg(X, y, *, seed=0, C=1.0, n_splits=5, max_iter=2000):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        Xte = sc.transform(Xte)
        clf = LogisticRegression(
            C=C, max_iter=max_iter, solver="lbfgs", n_jobs=1
        )
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return np.asarray(accs), np.asarray(f1s)


def _test_set_eval(X_trainval, y_trainval, X_test, y_test):
    sc = StandardScaler().fit(X_trainval)
    X_trainval = sc.transform(X_trainval)
    X_test = sc.transform(X_test)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", n_jobs=1)
    clf.fit(X_trainval, y_trainval)
    pred = clf.predict(X_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        ia = rng.integers(0, len(a), size=len(a))
        ib = rng.integers(0, len(b), size=len(b))
        diffs.append(a[ia].mean() - b[ib].mean())
    diffs = np.asarray(diffs)
    return dict(
        mean_diff=float(a.mean() - b.mean()),
        ci95_low=float(np.percentile(diffs, 2.5)),
        ci95_high=float(np.percentile(diffs, 97.5)),
        p_two_sided=float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    )


def run_dataset(dataset: str, subsample: int | None = None, seed: int = 0) -> dict:
    print(f"\n=== dataset={dataset} ===")
    trained = np.load(ACT_ROOT / dataset / "trained.npz", allow_pickle=True)
    random_cache = ACT_ROOT / dataset / "random.npz"
    random = np.load(random_cache, allow_pickle=True) if random_cache.exists() else None

    labels = trained["labels"].astype(str)
    split = trained["split"].astype(str)
    ids = trained["sequence_ids"].astype(str)
    gc = trained["gc"].astype(np.float32)

    # Build label encoder (string → int)
    cls = sorted(np.unique(labels).tolist())
    cls_to_idx = {c: i for i, c in enumerate(cls)}
    y = np.asarray([cls_to_idx[l] for l in labels], dtype=np.int64)

    # Sub-sample (balanced) for cross-validation speed while keeping chromosome splits.
    rng = np.random.default_rng(seed)
    if subsample is not None and len(y) > subsample:
        per_class = max(10, subsample // len(cls))
        keep = []
        for c in range(len(cls)):
            idx = np.where(y == c)[0]
            if len(idx) > per_class:
                keep.extend(rng.choice(idx, size=per_class, replace=False).tolist())
            else:
                keep.extend(idx.tolist())
        keep = np.asarray(sorted(keep))
    else:
        keep = np.arange(len(y))

    print(f"  n_classes={len(cls)}  n_total={len(y)}  n_used={len(keep)}  "
          f"classes={dict(zip(cls, np.bincount(y, minlength=len(cls)).tolist()))}")

    trainval_mask = np.isin(split[keep], ["train", "val"])
    test_mask = split[keep] == "test"
    trainval_idx = keep[trainval_mask]
    test_idx = keep[test_mask]
    print(f"  train+val: {len(trainval_idx)}, test: {len(test_idx)}")

    results: dict = {
        "dataset": dataset,
        "classes": cls,
        "n_total": int(len(y)),
        "n_used": int(len(keep)),
        "n_trainval": int(len(trainval_idx)),
        "n_test": int(len(test_idx)),
        "layers": {"trained": {}, "random": {}},
        "baselines": {},
    }

    hidden = trained["hidden"]  # (N, L, D)
    L = hidden.shape[1]
    layer_acc_trained = {}
    for li in range(L):
        X = hidden[trainval_idx, li]
        yy = y[trainval_idx]
        t0 = time.time()
        accs, f1s = _cv_logreg(X, yy, seed=seed, C=1.0)
        info = {
            "cv_acc_mean": float(accs.mean()),
            "cv_acc_std": float(accs.std(ddof=1)),
            "cv_acc_ci95": [
                float(np.percentile(accs, 2.5)),
                float(np.percentile(accs, 97.5)),
            ],
            "cv_f1_mean": float(f1s.mean()),
            "cv_f1_std": float(f1s.std(ddof=1)),
            "cv_accs": accs.tolist(),
            "cv_f1s": f1s.tolist(),
        }
        if len(test_idx) > 0:
            tacc, tf1 = _test_set_eval(X, yy, hidden[test_idx, li], y[test_idx])
            info["test_acc"] = float(tacc)
            info["test_f1"] = float(tf1)
        results["layers"]["trained"][str(li)] = info
        layer_acc_trained[li] = accs
        print(f"    trained L{li:2d}: acc {accs.mean():.3f}±{accs.std(ddof=1):.3f}  "
              f"f1 {f1s.mean():.3f}  ({time.time()-t0:.1f}s)")

    # Random model
    layer_acc_random = {}
    if random is not None:
        hr = random["hidden"]
        for li in range(hr.shape[1]):
            X = hr[trainval_idx, li]
            yy = y[trainval_idx]
            accs, f1s = _cv_logreg(X, yy, seed=seed, C=1.0)
            info = {
                "cv_acc_mean": float(accs.mean()),
                "cv_acc_std": float(accs.std(ddof=1)),
                "cv_accs": accs.tolist(),
                "cv_f1_mean": float(f1s.mean()),
                "cv_f1_std": float(f1s.std(ddof=1)),
            }
            if len(test_idx) > 0:
                tacc, tf1 = _test_set_eval(X, yy, hr[test_idx, li], y[test_idx])
                info["test_acc"] = float(tacc)
                info["test_f1"] = float(tf1)
            results["layers"]["random"][str(li)] = info
            layer_acc_random[li] = accs
            print(f"    random  L{li:2d}: acc {accs.mean():.3f}±{accs.std(ddof=1):.3f}")

    # Baselines: sequences -> features
    df = pd.read_parquet(PARQUET[dataset])
    # Drop duplicate sequence_ids BEFORE reindexing — some parquet builds have
    # a few duplicates (same window built twice for different transcripts).
    df = df.drop_duplicates("sequence_id")
    df = df.set_index("sequence_id").loc[ids].reset_index()
    # Sanity check: labels from parquet must match labels from cache.
    assert (df["label"].astype(str).values == labels).all(), \
        "parquet labels and cache labels do not align after reindex"
    # Match selected subsample
    df_sel = df.iloc[keep].reset_index(drop=True)

    for k in (3, 4, 5):
        t0 = time.time()
        Xk = np.stack([kmer_counts(s, k) for s in df_sel["sequence"].tolist()])
        X_tv = Xk[trainval_mask]
        y_tv = y[keep][trainval_mask]
        accs, f1s = _cv_logreg(X_tv, y_tv, seed=seed, C=1.0)
        info = {
            "cv_acc_mean": float(accs.mean()),
            "cv_acc_std": float(accs.std(ddof=1)),
            "cv_accs": accs.tolist(),
            "cv_f1_mean": float(f1s.mean()),
            "cv_f1_std": float(f1s.std(ddof=1)),
        }
        if test_mask.any():
            tacc, tf1 = _test_set_eval(X_tv, y_tv, Xk[test_mask], y[keep][test_mask])
            info["test_acc"] = float(tacc)
            info["test_f1"] = float(tf1)
        results["baselines"][f"{k}mer"] = info
        print(f"    {k}-mer: cv_acc {accs.mean():.3f}  ({time.time()-t0:.1f}s)")

    # GC baseline
    Xg = df_sel["gc"].to_numpy().astype(np.float32).reshape(-1, 1)
    X_tv = Xg[trainval_mask]
    y_tv = y[keep][trainval_mask]
    accs, f1s = _cv_logreg(X_tv, y_tv, seed=seed, C=1.0)
    info = {
        "cv_acc_mean": float(accs.mean()),
        "cv_acc_std": float(accs.std(ddof=1)),
        "cv_accs": accs.tolist(),
        "cv_f1_mean": float(f1s.mean()),
        "cv_f1_std": float(f1s.std(ddof=1)),
    }
    if test_mask.any():
        tacc, tf1 = _test_set_eval(X_tv, y_tv, Xg[test_mask], y[keep][test_mask])
        info["test_acc"] = float(tacc)
        info["test_f1"] = float(tf1)
    results["baselines"]["gc"] = info
    print(f"    gc    : cv_acc {accs.mean():.3f}")

    # Label-shuffle control
    rng2 = np.random.default_rng(seed + 1)
    y_shuf = y.copy()
    rng2.shuffle(y_shuf)
    X_tv = hidden[trainval_idx, int(L // 2)]  # middle layer
    accs, f1s = _cv_logreg(X_tv, y_shuf[trainval_idx], seed=seed, C=1.0)
    results["baselines"]["label_shuffle"] = {
        "cv_acc_mean": float(accs.mean()),
        "cv_acc_std": float(accs.std(ddof=1)),
        "cv_accs": accs.tolist(),
    }
    print(f"    shuffled labels @ mid layer: cv_acc {accs.mean():.3f}")

    # Effect-size tests
    if layer_acc_random:
        best_t_layer = max(layer_acc_trained, key=lambda li: layer_acc_trained[li].mean())
        best_r_layer = max(layer_acc_random, key=lambda li: layer_acc_random[li].mean())
        results["tests"] = {
            "trained_best_layer": int(best_t_layer),
            "random_best_layer": int(best_r_layer),
            "trained_vs_random_best": _bootstrap_diff(
                layer_acc_trained[best_t_layer], layer_acc_random[best_r_layer]
            ),
        }
        best_kmer = max(
            ((k, results["baselines"][k]["cv_accs"]) for k in ("3mer", "4mer", "5mer")),
            key=lambda kv: np.mean(kv[1]),
        )
        results["tests"]["trained_vs_best_kmer"] = {
            "best_kmer": best_kmer[0],
            **_bootstrap_diff(
                np.asarray(layer_acc_trained[best_t_layer]),
                np.asarray(best_kmer[1]),
            ),
        }

    out_path = OUT_DIR / f"{dataset}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"  wrote {out_path}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(PARQUET.keys()))
    ap.add_argument("--subsample", type=int, default=None,
                    help="Max #seqs to use for probing CV (balanced).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_dataset(args.dataset, subsample=args.subsample, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
