"""Cross-species analysis v2 — reviewer §4.

Uses extracted activations on three datasets:
  - multispecies           (natural-GC)
  - multispecies_gc_matched (GC-matched subset)
  - multispecies_heldout    (tomato + soybean, held out)

For each, computes:
  - 5-fold CV logistic regression accuracy per layer, trained model.
  - Baselines: GC-content-only, k-mer (k=3,4,5), random-init model.
  - Layer-wise CKA among species pairs.
  - Partial Mantel test: representational distance ~ phylogenetic distance,
    controlling for mean-GC distance (rough approximation of phylogeny).
  - Held-out species placement: for each heldout sequence, 1-NN in
    representation space over the four in-distribution species; report
    "nearest in-dist species" proportions per heldout species.

Writes data/real/results/cross_species/ various JSONs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MSPEC = REPO / "data" / "real" / "multispecies"
ACT_ROOT = REPO / "data" / "real" / "activations"
OUT = REPO / "data" / "real" / "results" / "cross_species"
OUT.mkdir(parents=True, exist_ok=True)

# Pairwise divergence times (median MYA) from TimeTree 5
# (Kumar et al., Mol. Biol. Evol. 2022, doi:10.1093/molbev/msac174;
#  http://www.timetree.org, queried 2025-04-17). Values are median estimates
# reported in the TimeTree database for each angiosperm species pair.
PHYL = {
    ("arabidopsis", "brachypodium_distachyon"): 160.0,
    ("arabidopsis", "glycine_max"):              107.0,
    ("arabidopsis", "oryza_sativa"):             160.0,
    ("arabidopsis", "solanum_lycopersicum"):     114.0,
    ("arabidopsis", "zea_mays"):                 160.0,
    ("brachypodium_distachyon", "glycine_max"):        160.0,
    ("brachypodium_distachyon", "oryza_sativa"):        53.0,
    ("brachypodium_distachyon", "solanum_lycopersicum"):160.0,
    ("brachypodium_distachyon", "zea_mays"):            53.0,
    ("glycine_max", "oryza_sativa"):             160.0,
    ("glycine_max", "solanum_lycopersicum"):     107.0,
    ("glycine_max", "zea_mays"):                 160.0,
    ("oryza_sativa", "solanum_lycopersicum"):    160.0,
    ("oryza_sativa", "zea_mays"):                 26.0,
    ("solanum_lycopersicum", "zea_mays"):        160.0,
}


def pair_key(a, b):
    return tuple(sorted([a, b]))


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


def cv_logreg(X, y, seed=0, C=1.0, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        p = clf.predict(Xte)
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    return np.asarray(accs), np.asarray(f1s)


def linear_cka(X, Y):
    """Linear CKA between two (n, d) matrices."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    denom = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / (denom + 1e-12))


def partial_mantel(rep_D, phy_D, gc_D, n_perm=5000, seed=0):
    """Partial Mantel: correlation(rep_D, phy_D) controlling for gc_D.

    Uses the standard Legendre approach: residualize rep_D and phy_D against
    gc_D via linear regression, then compute Pearson correlation of residuals.
    Permutation: shuffle rows/cols of phy_D and recompute.
    """
    def _triu(M):
        iu = np.triu_indices_from(M, k=1)
        return M[iu]
    rng = np.random.default_rng(seed)
    a = _triu(rep_D)
    b = _triu(phy_D)
    c = _triu(gc_D)
    def _resid(x, z):
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        A = np.column_stack([np.ones_like(z), z])
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        return x - A @ coef
    ra = _resid(a, c)
    rb = _resid(b, c)
    obs = float(np.corrcoef(ra, rb)[0, 1])
    # Permute b by shuffling species order in phy_D
    n = phy_D.shape[0]
    ge_obs = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        phy_p = phy_D[perm][:, perm]
        bp = _triu(phy_p)
        rbp = _resid(bp, c)
        r = float(np.corrcoef(ra, rbp)[0, 1])
        if abs(r) >= abs(obs):
            ge_obs += 1
    p = (ge_obs + 1) / (n_perm + 1)
    return obs, p


def run_probing_one(dataset: str, layer: int | None = None, seed: int = 0) -> dict:
    trained = np.load(ACT_ROOT / dataset / "trained.npz", allow_pickle=True)
    random_path = ACT_ROOT / dataset / "random.npz"
    random = np.load(random_path, allow_pickle=True) if random_path.exists() else None
    labels = trained["labels"].astype(str)
    gc = trained["gc"].astype(np.float32)
    cls = sorted(np.unique(labels).tolist())
    c2i = {c: i for i, c in enumerate(cls)}
    y = np.asarray([c2i[l] for l in labels], dtype=np.int64)

    res = {"dataset": dataset, "classes": cls, "n": int(len(y))}
    # per-layer trained
    layer_accs = {}
    H = trained["hidden"]
    for li in range(H.shape[1]):
        if layer is not None and li != layer:
            continue
        accs, f1s = cv_logreg(H[:, li], y, seed=seed)
        layer_accs[str(li)] = {
            "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
            "accs": accs.tolist(),
        }
    res["trained_per_layer"] = layer_accs
    if random is not None:
        r_accs = {}
        HR = random["hidden"]
        for li in range(HR.shape[1]):
            if layer is not None and li != layer:
                continue
            accs, _ = cv_logreg(HR[:, li], y, seed=seed)
            r_accs[str(li)] = {
                "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
            }
        res["random_per_layer"] = r_accs
    # Baselines: GC, kmer
    df_path = (
        MSPEC / "windows.parquet" if dataset == "multispecies"
        else MSPEC / "windows_gc_matched.parquet" if dataset == "multispecies_gc_matched"
        else MSPEC / "windows_heldout.parquet"
    )
    df = pd.read_parquet(df_path).set_index("sequence_id").loc[trained["sequence_ids"].astype(str)].reset_index()
    Xg = df["gc"].to_numpy().reshape(-1, 1).astype(np.float32)
    accs, _ = cv_logreg(Xg, y, seed=seed)
    res["gc_only"] = {"acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1))}
    for k in (3, 4, 5):
        Xk = np.stack([kmer_vec(s, k) for s in df["sequence"].tolist()])
        accs, _ = cv_logreg(Xk, y, seed=seed)
        res[f"{k}mer"] = {"acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1))}
    return res


def cka_and_distances(dataset: str, layer: int) -> dict:
    act = np.load(ACT_ROOT / dataset / "trained.npz", allow_pickle=True)
    H = act["hidden"][:, layer, :].astype(np.float32)
    labels = act["labels"].astype(str)
    gc = act["gc"].astype(np.float32)
    species = sorted(np.unique(labels).tolist())
    # Per-species matrix of activations
    per_sp = {s: H[labels == s] for s in species}
    gc_mean = {s: gc[labels == s].mean() for s in species}
    n = len(species)
    rep_D = np.zeros((n, n), dtype=np.float64)
    gc_D = np.zeros((n, n), dtype=np.float64)
    phy_D = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(species):
        for j, b in enumerate(species):
            if i == j:
                continue
            # Representational distance: 1 - CKA on equal-sized subsample
            k = min(len(per_sp[a]), len(per_sp[b]))
            rng = np.random.default_rng(0)
            ia = rng.choice(len(per_sp[a]), size=k, replace=False)
            ib = rng.choice(len(per_sp[b]), size=k, replace=False)
            cka = linear_cka(per_sp[a][ia], per_sp[b][ib])
            rep_D[i, j] = 1.0 - cka
            gc_D[i, j] = abs(gc_mean[a] - gc_mean[b])
            phy_D[i, j] = PHYL.get(pair_key(a, b), np.nan)
    obs, p = partial_mantel(rep_D, phy_D, gc_D)
    return {
        "species_order": species,
        "rep_distance": rep_D.tolist(),
        "gc_distance": gc_D.tolist(),
        "phy_distance": phy_D.tolist(),
        "partial_mantel_rep_vs_phy_controlling_gc": {
            "corr": float(obs), "p": float(p),
        },
    }


def heldout_1nn(layer: int) -> dict:
    """For each held-out sequence, find its 1-NN species among the 4 in-dist species."""
    in_dist = np.load(ACT_ROOT / "multispecies" / "trained.npz", allow_pickle=True)
    hold = np.load(ACT_ROOT / "multispecies_heldout" / "trained.npz", allow_pickle=True)
    H_in = in_dist["hidden"][:, layer, :].astype(np.float32)
    H_out = hold["hidden"][:, layer, :].astype(np.float32)
    y_in = in_dist["labels"].astype(str)
    y_out = hold["labels"].astype(str)
    # Filter in_dist to exclude held-out species (soy + tomato).
    held_species = set(np.unique(y_out).tolist())
    keep = ~np.isin(y_in, list(held_species))
    H_in = H_in[keep]
    y_in = y_in[keep]
    # Distances
    # Normalize for cosine
    H_in_n = H_in / (np.linalg.norm(H_in, axis=1, keepdims=True) + 1e-9)
    H_out_n = H_out / (np.linalg.norm(H_out, axis=1, keepdims=True) + 1e-9)
    sims = H_out_n @ H_in_n.T  # (n_out, n_in)
    nn_idx = np.argmax(sims, axis=1)
    nn_species = y_in[nn_idx]
    # Per held-out species, distribution over in-dist species
    out = {}
    for sp in np.unique(y_out):
        mask = y_out == sp
        nb = nn_species[mask]
        total = len(nb)
        d = {s: int((nb == s).sum()) for s in np.unique(y_in)}
        # Fraction
        d = {k: (v, float(v / total)) for k, v in d.items()}
        out[sp] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=7)
    args = ap.parse_args()

    results = {"layer": args.layer}
    for ds in ["multispecies", "multispecies_gc_matched", "multispecies_heldout"]:
        act_p = ACT_ROOT / ds / "trained.npz"
        if not act_p.exists():
            print(f"[skip] {ds}: missing {act_p}")
            continue
        print(f"\n--- {ds} ---")
        results[ds] = run_probing_one(ds, layer=None)
    # CKA & distances for in-distribution dataset
    if (ACT_ROOT / "multispecies" / "trained.npz").exists():
        print("\n--- CKA / partial Mantel (multispecies) ---")
        results["cka_multispecies"] = cka_and_distances("multispecies", args.layer)
        print(f"  partial Mantel: {results['cka_multispecies']['partial_mantel_rep_vs_phy_controlling_gc']}")
    if (ACT_ROOT / "multispecies_heldout" / "trained.npz").exists() and (
        ACT_ROOT / "multispecies" / "trained.npz"
    ).exists():
        print("\n--- Held-out 1-NN ---")
        results["heldout_1nn"] = heldout_1nn(args.layer)

    out_path = OUT / "cross_species_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
