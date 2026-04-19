"""Add ROC-AUC and calibration metrics to existing probing results.

Reads data/real/activations/<ds>/trained.npz, trains the same layer-wise
logistic regression as run_probing, and on the chromosome held-out test set
computes:
    - ROC-AUC (OVR for multi-class; direct for binary)
    - Expected Calibration Error (ECE, 10 bins)
    - Brier score (binary only)

Writes data/real/results/probing/<ds>_auc.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ACT_ROOT = REPO / "data" / "real" / "activations"
OUT = REPO / "data" / "real" / "results" / "probing"


def ece(probs, labels, n_bins=10):
    """Expected Calibration Error with equal-width bins on max-probability."""
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accs = (preds == labels).astype(np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    N = len(labels)
    for b in range(n_bins):
        m = (confs >= bins[b]) & (confs < bins[b + 1])
        if b == n_bins - 1:
            m |= confs == bins[b + 1]
        if m.sum() == 0:
            continue
        err += m.sum() / N * abs(accs[m].mean() - confs[m].mean())
    return float(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["region_type", "splice", "tss", "promoter"])
    args = ap.parse_args()
    results = {}
    for ds in args.datasets:
        p = ACT_ROOT / ds / "trained.npz"
        if not p.exists():
            continue
        cache = np.load(p, allow_pickle=True)
        H = cache["hidden"]
        y = cache["labels"].astype(str)
        split = cache["split"].astype(str)
        tv = np.isin(split, ["train", "val"])
        te = split == "test"
        classes = sorted(np.unique(y).tolist())
        c2i = {c: i for i, c in enumerate(classes)}
        y_int = np.asarray([c2i[c] for c in y])
        n_layers = H.shape[1]
        ds_res = {"classes": classes, "n_test": int(te.sum()), "layers": {}}
        for li in range(n_layers):
            X = H[:, li]
            sc = StandardScaler().fit(X[tv])
            Xs = sc.transform(X)
            clf = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
            clf.fit(Xs[tv], y_int[tv])
            probs = clf.predict_proba(Xs[te])
            preds = probs.argmax(axis=1)
            acc = (preds == y_int[te]).mean()
            try:
                if len(classes) == 2:
                    auc = roc_auc_score(y_int[te], probs[:, 1])
                    brier = brier_score_loss(y_int[te], probs[:, 1])
                else:
                    auc = roc_auc_score(y_int[te], probs, multi_class="ovr", average="macro")
                    brier = float("nan")
            except Exception:
                auc, brier = float("nan"), float("nan")
            e = ece(probs, y_int[te])
            ds_res["layers"][str(li)] = dict(
                test_acc=float(acc),
                test_auc=float(auc),
                brier=float(brier) if brier == brier else None,
                ece=float(e),
            )
        best_l = max(ds_res["layers"], key=lambda li: ds_res["layers"][li]["test_auc"])
        ds_res["best_auc_layer"] = int(best_l)
        ds_res["best_auc"] = ds_res["layers"][best_l]["test_auc"]
        results[ds] = ds_res
        row = ds_res["layers"][best_l]
        print(f"{ds:15s} best-AUC layer L{best_l}: "
              f"acc={row['test_acc']:.3f}  AUC={row['test_auc']:.3f}  "
              f"ECE={row['ece']:.3f}")
    out = OUT / "auc_calibration.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
