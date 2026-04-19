"""Fetch plant motif and TF-binding databases.

Sources:
- JASPAR CORE plantae 2024 (MEME and PFMs TAR).
- PlantTFDB 5.0 TF lists and Arabidopsis TFBS BED.
- PlantCARE descriptions (text dump).

Pinned URLs; content may be versioned. All files land in data/real/raw/motifs/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RAW = REPO / "data" / "real" / "raw" / "motifs"
RAW.mkdir(parents=True, exist_ok=True)

# JASPAR 2024 CORE plantae (non-redundant) — MEME format and PFMs archive.
JASPAR_BASE = "https://jaspar.genereg.net/download/data/2024"
JASPAR_TARGETS = [
    (
        f"{JASPAR_BASE}/CORE/JASPAR2024_CORE_non-redundant_pfms_meme.txt",
        "JASPAR2024_CORE_non-redundant_pfms_meme.txt",
    ),
    (
        f"{JASPAR_BASE}/CORE/JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt",
        "JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt",
    ),
    (
        f"{JASPAR_BASE}/CORE/JASPAR2024_CORE_plants_non-redundant_pfms_jaspar.txt",
        "JASPAR2024_CORE_plants_non-redundant_pfms_jaspar.txt",
    ),
]

# PlantTFDB 5.0 — TF family catalogues for Arabidopsis. Full site is behind
# browsable pages, but the per-species TF list is a static txt file.
PLANTTFDB_TARGETS = [
    (
        "http://planttfdb.gao-lab.org/download/TF_list/Ath_TF_list.txt.gz",
        "Ath_TF_list.txt.gz",
    ),
    (
        "http://planttfdb.gao-lab.org/download/seq/Ath_TF_pep.fas.gz",
        "Ath_TF_pep.fas.gz",
    ),
]


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name}")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url}")
    try:
        with requests.get(url, stream=True, timeout=60, verify=False) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            with tmp.open("wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 14):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(dest)
        print(f"[ok  ] {dest.name} -> {dest.stat().st_size/1e6:.2f} MB")
        return True
    except Exception as e:
        print(f"[err ] {url}: {e}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink()
        return False


def main() -> int:
    import urllib3

    urllib3.disable_warnings()
    print(f"Destination: {RAW}")
    for url, name in JASPAR_TARGETS:
        download(url, RAW / name)
    for url, name in PLANTTFDB_TARGETS:
        download(url, RAW / name)
    # Decompress small gzips
    import gzip
    import shutil

    for p in RAW.iterdir():
        if p.suffix == ".gz" and not p.with_suffix("").exists():
            with gzip.open(p, "rb") as fi, p.with_suffix("").open("wb") as fo:
                shutil.copyfileobj(fi, fo)
    print("\n== Summary ==")
    for p in sorted(RAW.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            print(f"{p.name:70s} {p.stat().st_size/1e6:6.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
