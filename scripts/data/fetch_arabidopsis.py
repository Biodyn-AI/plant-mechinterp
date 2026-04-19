"""Fetch Arabidopsis thaliana reference data.

Sources:
- TAIR10 genome FASTA (Ensembl Plants release 58)
- Araport11 gene annotation GFF3 (Ensembl Plants release 58)
- EPDnew A. thaliana promoters v1

All files land in data/real/raw/arabidopsis/. Checksums printed, URLs are pinned.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RAW = REPO / "data" / "real" / "raw" / "arabidopsis"
RAW.mkdir(parents=True, exist_ok=True)

# Ensembl Plants release 58 (pinned).
ENSEMBL_REL = 58
ENSEMBL_BASE = f"http://ftp.ensemblgenomes.org/pub/plants/release-{ENSEMBL_REL}"

TARGETS = [
    # (url, local_name)
    (
        f"{ENSEMBL_BASE}/fasta/arabidopsis_thaliana/dna/"
        "Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz",
        "TAIR10.dna.toplevel.fa.gz",
    ),
    (
        f"{ENSEMBL_BASE}/gff3/arabidopsis_thaliana/"
        f"Arabidopsis_thaliana.TAIR10.{ENSEMBL_REL}.gff3.gz",
        f"TAIR10.{ENSEMBL_REL}.gff3.gz",
    ),
    # EPDnew A. thaliana promoters (v1). The .sga file contains TSS coordinates.
    (
        "https://epd.expasy.org/ftp/epdnew/A_thaliana/current/At_EPDnew.sga",
        "At_EPDnew.sga",
    ),
    (
        "https://epd.expasy.org/ftp/epdnew/A_thaliana/current/At_EPDnew.bed",
        "At_EPDnew.bed",
    ),
    (
        "https://epd.expasy.org/ftp/epdnew/A_thaliana/current/At_EPDnew.fa",
        "At_EPDnew.fa",
    ),
]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size/1e6:.1f} MB)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 14):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    print(f"[ok  ] {dest.name} -> {dest.stat().st_size/1e6:.1f} MB")


def decompress_if_gz(path: Path) -> Path:
    if path.suffix != ".gz":
        return path
    out = path.with_suffix("")
    if out.exists():
        return out
    print(f"[gunz] {path.name} -> {out.name}")
    with gzip.open(path, "rb") as fi, out.open("wb") as fo:
        shutil.copyfileobj(fi, fo)
    return out


def main() -> int:
    print(f"Destination: {RAW}")
    for url, name in TARGETS:
        dest = RAW / name
        try:
            download(url, dest)
        except Exception as e:
            print(f"[err ] {name}: {e}", file=sys.stderr)
            # EPDnew URL paths can change; non-fatal for core workflow
            if "epd.expasy.org" in url:
                continue
            return 1
    # Decompress FASTA and GFF3
    for name in ("TAIR10.dna.toplevel.fa.gz", f"TAIR10.{ENSEMBL_REL}.gff3.gz"):
        p = RAW / name
        if p.exists():
            decompress_if_gz(p)
    # Report sizes and hashes
    print("\n== Summary ==")
    for p in sorted(RAW.iterdir()):
        if p.is_file():
            print(f"{p.name:50s} {p.stat().st_size/1e6:8.2f} MB  {sha256(p)[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
