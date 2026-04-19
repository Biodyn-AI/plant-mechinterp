"""Cross-reference SAE motif matches against PlantTFDB Arabidopsis TF list.

Merges the JASPAR-derived feature→motif CSV with the PlantTFDB TF table
(TF_ID, Gene_ID, Family) so each significant feature has a documented
Arabidopsis TF family attribution. Output table is the centerpiece for the
paper's biological interpretation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "data" / "real" / "results" / "sae_enrichment"
MOTIFS = REPO / "data" / "real" / "raw" / "motifs"


def canonical(name: str) -> str:
    """Normalize a JASPAR motif/alt name to a canonical key for matching."""
    s = str(name).strip().upper()
    s = s.replace("_", "").replace("-", "").replace(".", "").replace(" ", "")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-tag", default="topk_exp16_L7_seed0")
    args = ap.parse_args()

    # Best per feature (already in OUT)
    best = pd.read_csv(OUT / f"{args.sae_tag}_best_per_feature.csv")
    # PlantTFDB list
    tfdb = pd.read_csv(MOTIFS / "Ath_TF_list.txt", sep="\t")
    # Keep first occurrence per gene
    tfdb["gene_upper"] = tfdb["Gene_ID"].str.upper()
    tfdb_by_gene = tfdb.drop_duplicates("gene_upper").set_index("gene_upper")

    # JASPAR motif_alt is usually the TF gene symbol; try matching by gene symbol
    # first, then loose family-name match from a lookup dict.

    # Family abbreviation lookup from common TF names
    family_map = {
        "BPC": "BBR-BPC",
        "DOF": "Dof",
        "MYB": "MYB",
        "NAC": "NAC",
        "ERF": "ERF",
        "LEC": "B3",
        "AHL": "AT-hook",
        "WRKY": "WRKY",
        "JUB": "NAC",  # JUB1 is a NAC TF
        "GRF": "GRF",
        "ZHD": "HD-ZIP",
        "NTL": "NAC",
        "SPL": "SBP",
        "HY5": "bZIP",
    }

    rows = []
    for _, r in best.iterrows():
        alt = str(r["motif_alt"])
        alt_c = canonical(alt)
        family = None
        gene_id = None
        # Strategy 1: direct gene-name prefix match in PlantTFDB
        for prefix, fam in family_map.items():
            if alt_c.startswith(prefix):
                family = fam
                break
        # Strategy 2: look for AT-style gene IDs
        if alt.startswith("AT") and "G" in alt:
            gene = alt.split(".")[0].upper()
            if gene in tfdb_by_gene.index:
                hit = tfdb_by_gene.loc[gene]
                family = hit["Family"]
                gene_id = hit["Gene_ID"]
        rows.append(
            dict(
                feature=int(r["feature"]),
                motif=r["motif"],
                motif_alt=alt,
                jaspar_q=float(r["q"]),
                planttfdb_gene=gene_id,
                planttfdb_family=family,
            )
        )
    out_df = pd.DataFrame(rows)
    merged = best.merge(out_df, on=["feature", "motif", "motif_alt"], suffixes=("", "_xref"))
    path = OUT / f"{args.sae_tag}_best_with_planttfdb.csv"
    merged.to_csv(path, index=False)
    print(f"wrote {path}")

    sig = merged[merged["q"] < 0.05]
    fam_counts = sig["planttfdb_family"].value_counts(dropna=False)
    print(f"\n{len(sig)}/{len(merged)} features have q<0.05 best motif.")
    print("\nPlantTFDB family assignments for significant matches:")
    for fam, cnt in fam_counts.items():
        print(f"  {str(fam)}: {cnt}")


if __name__ == "__main__":
    main()
