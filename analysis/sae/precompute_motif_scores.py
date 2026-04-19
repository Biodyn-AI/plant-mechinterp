"""Precompute max-log-odds scores for every (sequence × motif) pair on real
data. Runs independently of SAE training so we can do it in parallel with the
baseline compute on MPS.

Writes data/real/results/sae_enrichment/{dataset}_motif_scores.npy (N, M).
Also writes motif metadata CSV.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from motif_enrichment import (
    BASE_IDX,
    motif_max_scores_batch,
    onehot_batch,
    parse_meme,
)

REPO = HERE.parent.parent
ACT_ROOT = REPO / "data" / "real" / "activations"
ARAREG = REPO / "data" / "real" / "arareg"
OUT = REPO / "data" / "real" / "results" / "sae_enrichment"
OUT.mkdir(parents=True, exist_ok=True)
MOTIFS = REPO / "data" / "real" / "raw" / "motifs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="region_type")
    ap.add_argument("--meme-file", default=str(
        MOTIFS / "JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt"))
    args = ap.parse_args()

    act = np.load(ACT_ROOT / args.dataset / "trained.npz", allow_pickle=True)
    ids = act["sequence_ids"].astype(str)
    df = pd.read_parquet(ARAREG / f"{args.dataset}.parquet").set_index("sequence_id").loc[ids]
    seqs = df["sequence"].tolist()
    L = len(seqs[0])
    print(f"[oh  ] encoding {len(seqs)} × {L} nt ...")
    t0 = time.time()
    one_hot = onehot_batch(seqs, L)
    print(f"[oh  ] done {time.time()-t0:.1f}s   size={one_hot.nbytes/1e9:.2f} GB")

    motifs = parse_meme(Path(args.meme_file))
    meta = pd.DataFrame([
        {"idx": i, "name": m["name"], "alt": m["alt"], "width": m["width"]}
        for i, m in enumerate(motifs)
    ])
    meta.to_csv(OUT / "motif_metadata.csv", index=False)
    print(f"[mot ] {len(motifs)} motifs")

    N, M = len(seqs), len(motifs)
    scores = np.zeros((N, M), dtype=np.float32)
    t0 = time.time()
    for mi, mot in enumerate(motifs):
        scores[:, mi] = motif_max_scores_batch(one_hot, mot["log_odds"])
        if mi % 50 == 0:
            el = time.time() - t0
            rate = (mi + 1) / max(1e-9, el)
            eta = (M - mi - 1) / max(1e-9, rate)
            print(f"  motif {mi+1}/{M}  ({rate:.1f}/s, eta {eta/60:.1f} min)")
    out = OUT / f"{args.dataset}_motif_scores.npy"
    np.save(out, scores)
    print(f"wrote {out}  ({scores.shape}, {scores.nbytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
