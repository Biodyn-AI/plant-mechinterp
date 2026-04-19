"""GC-matched AraReg probing — reviewer §3.2.

For each AraReg task, construct a GC-matched subset where every class has an
overlapping GC distribution, then repeat 5-fold CV probing at layer 11
(best from original). Report trained, random-init, 4-mer, and GC baselines
side-by-side with the unmatched versions.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ACT = REPO / "data" / "real" / "activations"
ARAREG = REPO / "data" / "real" / "arareg"
OUT = REPO / "data" / "real" / "results" / "probing"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "region_type": "region_type.parquet",
    "splice":      "splice.parquet",
    "tss":         "tss.parquet",
    "promoter":    "promoter.parquet",
}


def kmer_counts(seq, k):
    alpha = "ACGT"; idx = {c: i for i, c in enumerate(alpha)}
    L = 4 ** k
    out = np.zeros(L, dtype=np.float32)
    for i in range(len(seq) - k + 1):
        km = seq[i:i+k]
        if any(c not in idx for c in km):
            continue
        j = 0
        for c in km:
            j = j*4 + idx[c]
        out[j] += 1
    tot = out.sum()
    if tot > 0:
        out /= tot
    return out


def gc_match(df: pd.DataFrame, label_col: str, bin_width: float = 0.01,
             per_class: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Keep GC bins where every class has at least one sequence, then sample
    min(per-class count) from each class per bin."""
    rng = random.Random(seed)
    df = df.copy()
    df["_bin"] = (df["gc"] / bin_width).round().astype(int)
    keep_idx: list[int] = []
    classes = sorted(df[label_col].unique().tolist())
    for _, by_bin in df.groupby("_bin"):
        counts = by_bin[label_col].value_counts()
        if len(counts) < len(classes):
            continue
        k = int(counts.min())
        if k == 0:
            continue
        for c in classes:
            sub = by_bin[by_bin[label_col] == c]
            keep = rng.sample(list(sub.index), min(k, len(sub)))
            keep_idx.extend(keep)
    out = df.loc[sorted(keep_idx)].drop(columns=["_bin"])
    if per_class is not None:
        pieces = []
        for c in classes:
            sub = out[out[label_col] == c]
            if len(sub) > per_class:
                sub = sub.sample(n=per_class, random_state=seed)
            pieces.append(sub)
        out = pd.concat(pieces).reset_index(drop=True)
    return out


def cv_logreg(X, y, seed=0, C=1.0, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        p = clf.predict(Xte)
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    return np.asarray(accs), np.asarray(f1s)


def run_one(dataset: str) -> dict:
    print(f"\n=== {dataset} ===")
    trained = np.load(ACT / dataset / "trained.npz", allow_pickle=True)
    random_c = ACT / dataset / "random.npz"
    rnd = np.load(random_c, allow_pickle=True) if random_c.exists() else None

    full = pd.read_parquet(ARAREG / DATASETS[dataset])
    cache_ids = trained["sequence_ids"].astype(str)
    cache_id_set = set(cache_ids.tolist())
    df = full[full["sequence_id"].isin(cache_id_set)].copy()
    # Drop any duplicates in the parquet (can arise if overlap windows produced
    # identical ids during build).
    df = df.drop_duplicates("sequence_id").copy()
    # Keep only ids that actually exist in df (after dedup).
    common_ids = [i for i in cache_ids if i in set(df["sequence_id"].tolist())]
    df = df.set_index("sequence_id").reindex(common_ids).reset_index()
    # Map cache positions to df rows.
    sid2pos_cache = {s: i for i, s in enumerate(cache_ids)}
    df["_cache_pos"] = df["sequence_id"].map(sid2pos_cache)
    df_matched = gc_match(df, "label")
    print(f"  matched {len(df_matched)}/{len(df)}; per class: {df_matched['label'].value_counts().to_dict()}")
    print(f"  GC stats (matched):")
    for c in sorted(df_matched["label"].unique()):
        sub = df_matched[df_matched["label"] == c]
        print(f"    {c}: mean={sub['gc'].mean():.3f}  std={sub['gc'].std():.3f}  n={len(sub)}")
    pos = df_matched["_cache_pos"].to_numpy()
    labels = trained["labels"].astype(str)[pos]
    cls = sorted(np.unique(labels).tolist())
    c2i = {c: i for i, c in enumerate(cls)}
    y = np.asarray([c2i[l] for l in labels])

    res = {"dataset": dataset, "n_matched": int(len(pos)),
           "n_classes": len(cls), "classes": cls,
           "per_class_counts": {c: int((labels == c).sum()) for c in cls}}

    H = trained["hidden"][pos]
    n_layers = H.shape[1]
    per_layer = {}
    best_acc, best_layer = -1, 0
    for li in range(n_layers):
        accs, f1s = cv_logreg(H[:, li], y, seed=0)
        per_layer[str(li)] = {
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std(ddof=1)),
            "f1_mean": float(f1s.mean()),
        }
        if accs.mean() > best_acc:
            best_acc, best_layer = float(accs.mean()), li
    res["trained_per_layer"] = per_layer
    res["trained_best_layer"] = int(best_layer)
    res["trained_best_acc"] = best_acc
    print(f"  trained best L{best_layer}: {best_acc:.3f}")

    if rnd is not None:
        HR = rnd["hidden"][pos]
        r_per = {}
        br_acc, br_layer = -1, 0
        for li in range(HR.shape[1]):
            accs, _ = cv_logreg(HR[:, li], y, seed=0)
            r_per[str(li)] = {"acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1))}
            if accs.mean() > br_acc:
                br_acc, br_layer = float(accs.mean()), li
        res["random_per_layer"] = r_per
        res["random_best_layer"] = int(br_layer)
        res["random_best_acc"] = br_acc
        print(f"  random  best L{br_layer}: {br_acc:.3f}")

    # k-mer + GC
    seqs = df_matched["sequence"].tolist()
    for k in (3, 4, 5):
        Xk = np.stack([kmer_counts(s, k) for s in seqs])
        accs, _ = cv_logreg(Xk, y, seed=0)
        res[f"{k}mer"] = {"acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1))}
        print(f"  {k}-mer: {accs.mean():.3f}")
    Xg = df_matched["gc"].to_numpy().reshape(-1, 1).astype(np.float32)
    accs, _ = cv_logreg(Xg, y, seed=0)
    res["gc_only"] = {"acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1))}
    print(f"  gc_only: {accs.mean():.3f}")

    return res


def main():
    all_res = {}
    for ds in DATASETS:
        all_res[ds] = run_one(ds)
    out = OUT / "arareg_gc_matched.json"
    with out.open("w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
