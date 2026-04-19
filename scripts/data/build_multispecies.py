"""Build multi-species dataset for §4 cross-species v2.

- 2 000 random 1-kb windows per species across 6 plant genomes (configurable).
- Windows are sampled from the primary nuclear chromosomes only (drops Pt/Mt
  and unplaced scaffolds).
- Drops windows with >5% N.
- GC-matched subset (kernel-density matching on mean GC) is produced in a
  separate parquet so the §4.3 control can be run directly.
- Train/val/test split is by-chromosome within each species when >=3
  chromosomes, otherwise 70/15/15 by random seed.

Outputs:
    data/real/multispecies/windows.parquet         # all windows
    data/real/multispecies/windows_gc_matched.parquet
    data/real/multispecies/windows_heldout.parquet # 2 held-out species
    data/real/splits/multispecies_splits.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RAW = REPO / "data" / "real" / "raw"
OUT = REPO / "data" / "real" / "multispecies"
SPLITS = REPO / "data" / "real" / "splits"
OUT.mkdir(parents=True, exist_ok=True)
SPLITS.mkdir(parents=True, exist_ok=True)

# species slug → (FASTA filename in raw/<slug>/, list of primary chrom names)
SPECIES = {
    "arabidopsis": (
        "arabidopsis/TAIR10.dna.toplevel.fa",
        ["1", "2", "3", "4", "5"],
        False,  # held-in
    ),
    "oryza_sativa": (
        "oryza_sativa/Oryza_sativa.IRGSP-1.0.dna.toplevel.fa",
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        False,
    ),
    "zea_mays": (
        "zea_mays/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa",
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        False,
    ),
    "brachypodium_distachyon": (
        "brachypodium_distachyon/"
        "Brachypodium_distachyon.Brachypodium_distachyon_v3.0.dna.toplevel.fa",
        ["1", "2", "3", "4", "5"],
        False,
    ),
    # held-out for generalization test
    "solanum_lycopersicum": (
        "solanum_lycopersicum/Solanum_lycopersicum.SL3.0.dna.toplevel.fa",
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        True,
    ),
    "glycine_max": (
        "glycine_max/Glycine_max.Glycine_max_v2.1.dna.toplevel.fa",
        [str(i) for i in range(1, 21)],
        True,
    ),
}

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(COMPLEMENT)[::-1]


def gc(s: str) -> float:
    s = s.upper()
    n = len(s) - s.count("N")
    if n == 0:
        return float("nan")
    return (s.count("G") + s.count("C")) / n


def sample_windows(
    fa_path: Path,
    chroms: list[str],
    *,
    n: int,
    window: int,
    rng: random.Random,
    max_n_frac: float = 0.05,
) -> list[dict]:
    print(f"[fa  ] loading {fa_path}")
    seqs = {}
    for rec in SeqIO.parse(str(fa_path), "fasta"):
        rid = rec.id
        if rid in chroms:
            seqs[rid] = str(rec.seq).upper()
    print(f"  chromosomes kept: {list(seqs.keys())}")

    per_chrom = max(1, n // max(1, len(seqs)))
    records: list[dict] = []
    for chrom, seq in seqs.items():
        L = len(seq)
        attempts = 0
        collected = 0
        while collected < per_chrom and attempts < per_chrom * 50:
            attempts += 1
            start = rng.randint(1, max(1, L - window))
            end = start + window - 1
            s = seq[start - 1 : end]
            if s.count("N") / len(s) > max_n_frac:
                continue
            strand = rng.choice(["+", "-"])
            if strand == "-":
                s = revcomp(s)
            records.append(
                dict(
                    sequence_id=f"{fa_path.stem}_{chrom}_{start}_{end}_{strand}",
                    sequence=s,
                    gc=gc(s),
                    length=len(s),
                    chrom=chrom,
                    start=start,
                    end=end,
                    strand=strand,
                )
            )
            collected += 1
    # Top-up to n with random extra draws across chroms if we came up short
    while len(records) < n:
        chrom = rng.choice(list(seqs.keys()))
        L = len(seqs[chrom])
        start = rng.randint(1, max(1, L - window))
        end = start + window - 1
        s = seqs[chrom][start - 1 : end]
        if s.count("N") / len(s) > max_n_frac:
            continue
        strand = rng.choice(["+", "-"])
        if strand == "-":
            s = revcomp(s)
        records.append(
            dict(
                sequence_id=f"{fa_path.stem}_{chrom}_{start}_{end}_{strand}",
                sequence=s,
                gc=gc(s),
                length=len(s),
                chrom=chrom,
                start=start,
                end=end,
                strand=strand,
            )
        )
    return records[:n]


def by_chrom_split(df: pd.DataFrame, rng: random.Random) -> pd.Series:
    """Per-species chromosome split: 70% train / 15% val / 15% test chroms."""
    splits = pd.Series(index=df.index, dtype="object")
    for species, sub in df.groupby("species"):
        chroms = sorted(sub["chrom"].unique(), key=str)
        rng_local = random.Random(hash(species) & 0xFFFFFFFF)
        rng_local.shuffle(chroms)
        k = len(chroms)
        n_train = max(1, int(0.7 * k))
        n_val = max(1, int(0.15 * k))
        train = set(chroms[:n_train])
        val = set(chroms[n_train : n_train + n_val])
        test = set(chroms[n_train + n_val :])
        if not test:  # ensure at least one test chrom
            test = {chroms[-1]}
            val = val - test
        for idx in sub.index:
            c = sub.at[idx, "chrom"]
            if c in test:
                splits.at[idx] = "test"
            elif c in val:
                splits.at[idx] = "val"
            else:
                splits.at[idx] = "train"
    return splits


def gc_match(df: pd.DataFrame, species_col: str, gc_col: str, per_species: int, rng: random.Random) -> pd.DataFrame:
    """KDE-like matching: bin GC into 0.01-wide bins; keep the species-balanced
    intersection — for each bin take min count across species, sample that many
    from each species. Yields up to `per_species` per species."""
    df = df.copy()
    df["_bin"] = (df[gc_col] * 100).round().astype("Int64")
    species = sorted(df[species_col].unique())
    keep_idx = []
    for bin_val, by_bin in df.groupby("_bin"):
        counts = by_bin[species_col].value_counts()
        if len(counts) < len(species):
            continue
        k = int(counts.min())
        if k == 0:
            continue
        for sp in species:
            sub = by_bin[by_bin[species_col] == sp]
            idx = rng.sample(list(sub.index), min(k, len(sub)))
            keep_idx.extend(idx)
    out = df.loc[sorted(keep_idx)].drop(columns=["_bin"])
    # Subsample to per_species per species if oversized
    pieces = []
    for sp, sub in out.groupby(species_col):
        if len(sub) > per_species:
            sub = sub.sample(n=per_species, random_state=rng.randint(0, 1 << 30))
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species", type=int, default=2000)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    all_records = []
    for species, (fa_rel, chroms, heldout) in SPECIES.items():
        fa_path = RAW / fa_rel
        if not fa_path.exists():
            print(f"[skip] {species}: missing {fa_path}")
            continue
        recs = sample_windows(
            fa_path, chroms, n=args.per_species, window=args.window, rng=rng
        )
        for r in recs:
            r["species"] = species
            r["held_out"] = heldout
            r["label"] = species
            r["source"] = "ensembl_plants_r58"
            all_records.append(r)

    df = pd.DataFrame(all_records)
    df["split"] = by_chrom_split(df, rng)
    df.to_parquet(OUT / "windows.parquet", index=False)
    print(f"\nwrote {OUT/'windows.parquet'}: {len(df)} rows")
    print(df.groupby("species").agg(n=("sequence_id", "count"), gc_mean=("gc", "mean")))

    held_df = df[df["held_out"]].copy()
    held_df.to_parquet(OUT / "windows_heldout.parquet", index=False)
    print(f"wrote {OUT/'windows_heldout.parquet'}: {len(held_df)} rows (held-out species)")

    in_df = df[~df["held_out"]].copy()
    matched = gc_match(in_df, "species", "gc", args.per_species, rng)
    matched.to_parquet(OUT / "windows_gc_matched.parquet", index=False)
    print(f"wrote {OUT/'windows_gc_matched.parquet'}: {len(matched)} rows (GC-matched, in-distribution)")
    if len(matched):
        print(matched.groupby("species").agg(n=("sequence_id", "count"), gc_mean=("gc", "mean")))

    df[["sequence_id", "species", "chrom", "split", "gc", "held_out"]].to_csv(
        SPLITS / "multispecies_splits.csv", index=False
    )
    print(f"wrote {SPLITS/'multispecies_splits.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
