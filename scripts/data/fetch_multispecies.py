"""Fetch genomes + GFF3 annotations for 5 additional plant species from Ensembl Plants r58.

Species pinned here (with Ensembl assembly names used in URL paths):
- Oryza sativa Japonica       (IRGSP-1.0)        - rice, monocot
- Zea mays                    (Zm-B73-REFERENCE-NAM-5.0) - maize, monocot (large)
- Solanum lycopersicum        (SL3.0)            - tomato, eudicot
- Glycine max                 (Glycine_max_v2.1) - soybean, eudicot (large)
- Brachypodium distachyon     (Brachypodium_distachyon_v3.0) - monocot (small)

Downloads the "dna.toplevel" FASTA and the release-58 GFF3. Run independently
from fetch_arabidopsis.py.
"""
from __future__ import annotations

import gzip
import hashlib
import shutil
import sys
from pathlib import Path

import requests
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RAW = REPO / "data" / "real" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

ENSEMBL_REL = 58
BASE = f"http://ftp.ensemblgenomes.org/pub/plants/release-{ENSEMBL_REL}"

# Each entry: (dir_slug, ensembl_species, assembly_name, notes)
SPECIES = [
    ("oryza_sativa", "oryza_sativa", "IRGSP-1.0", "rice"),
    ("zea_mays", "zea_mays", "Zm-B73-REFERENCE-NAM-5.0", "maize (~2.3 GB FASTA)"),
    ("solanum_lycopersicum", "solanum_lycopersicum", "SL3.0", "tomato"),
    ("glycine_max", "glycine_max", "Glycine_max_v2.1", "soybean"),
    (
        "brachypodium_distachyon",
        "brachypodium_distachyon",
        "Brachypodium_distachyon_v3.0",
        "Brachypodium",
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url}")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
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
        return True
    except Exception as e:
        print(f"[err ] {url}: {e}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink()
        return False


def decompress_if_gz(path: Path) -> Path:
    if path.suffix != ".gz":
        return path
    out = path.with_suffix("")
    if out.exists():
        return out
    with gzip.open(path, "rb") as fi, out.open("wb") as fo:
        shutil.copyfileobj(fi, fo)
    # Drop the .gz once we have the uncompressed version to save disk.
    path.unlink()
    print(f"[gunz] {path.name} -> {out.name}")
    return out


def fetch_species(slug: str, ensembl_species: str, assembly: str, note: str) -> None:
    print(f"\n--- {slug} ({note}) ---")
    out_dir = RAW / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    fa_name = f"{ensembl_species.capitalize()}.{assembly}.dna.toplevel.fa.gz"
    gff_name = f"{ensembl_species.capitalize()}.{assembly}.{ENSEMBL_REL}.gff3.gz"
    fa_url = f"{BASE}/fasta/{ensembl_species}/dna/{fa_name}"
    gff_url = f"{BASE}/gff3/{ensembl_species}/{gff_name}"

    ok_fa = download(fa_url, out_dir / fa_name)
    ok_gff = download(gff_url, out_dir / gff_name)
    if ok_fa:
        decompress_if_gz(out_dir / fa_name)
    if ok_gff:
        decompress_if_gz(out_dir / gff_name)


def main() -> int:
    print(f"Destination: {RAW}")
    for slug, species, assembly, note in SPECIES:
        fetch_species(slug, species, assembly, note)
    print("\n== Summary ==")
    for species_dir in sorted(RAW.iterdir()):
        if not species_dir.is_dir():
            continue
        for p in sorted(species_dir.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                print(f"{species_dir.name}/{p.name:55s} {p.stat().st_size/1e6:8.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
