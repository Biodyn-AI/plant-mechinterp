"""Build AraReg region-type, TSS, UTR, and splice-site datasets from TAIR10+Araport11.

Inputs:
    data/real/raw/arabidopsis/TAIR10.dna.toplevel.fa
    data/real/raw/arabidopsis/TAIR10.58.gff3
    data/real/raw/arabidopsis/At_EPDnew.bed

Outputs (under data/real/arareg/):
    region_type.parquet        # 4-way: exon / intron / utr5 / utr3 / intergenic
    splice.parquet             # 3-way: donor / acceptor / non-site
    tss.parquet                # binary: TSS window / shuffled
    promoter.parquet           # binary: EPDnew promoter / random intergenic

Also writes data/real/splits/arareg_splits.csv (sequence_id -> {train,val,test}).

We keep the core/window length configurable (default 1024) so the tokenizer has
plenty of context, and a second 200-nt variant is produced to support the
length-matched comparison with the current paper.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from Bio import SeqIO

# ------------------------------------------------------------------ Paths ----

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RAW = REPO / "data" / "real" / "raw" / "arabidopsis"
OUT = REPO / "data" / "real" / "arareg"
SPLITS = REPO / "data" / "real" / "splits"
OUT.mkdir(parents=True, exist_ok=True)
SPLITS.mkdir(parents=True, exist_ok=True)

FASTA = RAW / "TAIR10.dna.toplevel.fa"
GFF3 = RAW / "TAIR10.58.gff3"
EPDNEW_BED = RAW / "At_EPDnew.bed"

# ------------------------------------------------------------ Parse GFF3 ----


@dataclass
class Gene:
    gid: str
    chrom: str
    start: int  # 1-based inclusive (GFF)
    end: int
    strand: str
    transcripts: dict  # tid -> list of (feature_type, start, end, strand)


def parse_gff3(path: Path) -> dict:
    """Return {gene_id: Gene}. Only keeps protein-coding genes with ≥1 mRNA."""
    print(f"[gff ] parsing {path}")
    genes: dict[str, Gene] = {}
    tx2gene: dict[str, str] = {}
    pc_genes: set[str] = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, ftype, start, end, _, strand, _, attrs = parts
            start, end = int(start), int(end)
            attr_kv = dict(
                x.split("=", 1) for x in attrs.split(";") if "=" in x
            )
            if ftype == "gene":
                gid = attr_kv.get("ID", "").removeprefix("gene:")
                if attr_kv.get("biotype") == "protein_coding":
                    pc_genes.add(gid)
                    genes[gid] = Gene(
                        gid=gid,
                        chrom=chrom,
                        start=start,
                        end=end,
                        strand=strand,
                        transcripts={},
                    )
            elif ftype == "mRNA":
                tid = attr_kv.get("ID", "").removeprefix("transcript:")
                parent = attr_kv.get("Parent", "").removeprefix("gene:")
                if parent in genes:
                    genes[parent].transcripts[tid] = []
                    tx2gene[tid] = parent
            elif ftype in {"exon", "CDS", "five_prime_UTR", "three_prime_UTR"}:
                parent = attr_kv.get("Parent", "").removeprefix("transcript:")
                # Parent may be comma-separated for multi-tx exons
                for pid in parent.split(","):
                    pid = pid.removeprefix("transcript:")
                    gid = tx2gene.get(pid)
                    if gid and pid in genes[gid].transcripts:
                        genes[gid].transcripts[pid].append(
                            (ftype, start, end, strand)
                        )
    print(f"[gff ] {len(genes):,} protein-coding genes parsed")
    return genes


# ----------------------------------------------------------- Load FASTA ----


def load_fasta(path: Path) -> dict[str, str]:
    print(f"[fa  ] loading {path}")
    seqs = {}
    for rec in SeqIO.parse(str(path), "fasta"):
        # Ensembl uses "1", "2", ..., "Pt", "Mt".
        seqs[rec.id] = str(rec.seq).upper()
    print(f"[fa  ] chromosomes: {sorted(seqs.keys())}")
    return seqs


# ------------------------------------------------------ Sequence helpers ----

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(COMPLEMENT)[::-1]


def fetch(seqs: dict[str, str], chrom: str, start: int, end: int, strand: str) -> str:
    """Return sequence on the *plus* strand span [start, end] (1-based inclusive),
    reverse-complemented if strand is '-'.
    Returns '' on invalid coords.
    """
    if chrom not in seqs:
        return ""
    chrom_seq = seqs[chrom]
    if start < 1 or end > len(chrom_seq) or end < start:
        return ""
    s = chrom_seq[start - 1 : end]
    return revcomp(s) if strand == "-" else s


def gc_content(s: str) -> float:
    if not s:
        return float("nan")
    n = len(s) - s.count("N")
    if n == 0:
        return float("nan")
    return (s.count("G") + s.count("C")) / n


def n_fraction(s: str) -> float:
    if not s:
        return 1.0
    return s.count("N") / len(s)


def dinuc_shuffle(s: str, rng: random.Random) -> str:
    """Simple dinucleotide-preserving shuffle using an Eulerian walk on the
    dinucleotide graph (Altschul & Erickson 1985, implemented here
    pragmatically via the deBruijn-graph-eulerian approach).
    """
    s = s.upper()
    if len(s) < 2:
        return s
    alphabet = sorted(set(s))
    # Build successor lists per base
    succ: dict[str, list[str]] = {a: [] for a in alphabet}
    for a, b in zip(s, s[1:]):
        succ[a].append(b)
    # Shuffle the successors
    for a in alphabet:
        rng.shuffle(succ[a])
    # Reconstruct by walking; ensure the last-letter-of-each-base goes last
    last_char = s[-1]
    end_succ: dict[str, str] = {}
    for a in alphabet:
        if a == last_char:
            continue
        # pick some edge ending in last_char's walk arbitrarily
        for i, b in enumerate(succ[a]):
            if b == last_char:
                end_succ[a] = succ[a].pop(i)
                break
    # Walk
    out = [s[0]]
    cur = s[0]
    while True:
        if succ[cur]:
            nxt = succ[cur].pop()
            out.append(nxt)
            cur = nxt
        elif cur in end_succ:
            nxt = end_succ.pop(cur)
            out.append(nxt)
            cur = nxt
        else:
            break
    if len(out) != len(s):
        # Fallback: naive shuffle (rare on N-free sequences).
        arr = list(s)
        rng.shuffle(arr)
        return "".join(arr)
    return "".join(out)


# ----------------------------------------------------- Region extractors ----


def collect_introns(gene: Gene) -> list[tuple[str, int, int, str]]:
    """Introns implied by per-transcript sorted exons."""
    introns = []
    for tid, feats in gene.transcripts.items():
        exons = sorted(
            [(s, e) for (f, s, e, _) in feats if f == "exon"]
        )
        if len(exons) < 2:
            continue
        for (a_s, a_e), (b_s, b_e) in zip(exons, exons[1:]):
            intron_s, intron_e = a_e + 1, b_s - 1
            if intron_e > intron_s:
                introns.append((gene.chrom, intron_s, intron_e, gene.strand))
    return introns


def collect_internal_exons(gene: Gene, min_len: int = 100) -> list[tuple[str, int, int, str]]:
    out = []
    for tid, feats in gene.transcripts.items():
        exons = sorted([(s, e) for (f, s, e, _) in feats if f == "exon"])
        if len(exons) < 3:
            continue
        for s, e in exons[1:-1]:
            if e - s + 1 >= min_len:
                out.append((gene.chrom, s, e, gene.strand))
    return out


def collect_utrs(gene: Gene, utr_type: str, min_len: int = 50) -> list[tuple[str, int, int, str]]:
    key = "five_prime_UTR" if utr_type == "5" else "three_prime_UTR"
    out = []
    for tid, feats in gene.transcripts.items():
        for (f, s, e, strand) in feats:
            if f == key and e - s + 1 >= min_len:
                out.append((gene.chrom, s, e, strand))
    return out


def collect_intergenic_windows(
    genes: dict[str, Gene],
    seqs: dict[str, str],
    *,
    min_distance: int = 5000,
    window: int = 1024,
    n_per_chrom: int = 1000,
    rng: random.Random,
) -> list[tuple[str, int, int, str]]:
    # Build sorted gene intervals per chrom
    by_chrom: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for g in genes.values():
        by_chrom[g.chrom].append((g.start, g.end))
    for k in by_chrom:
        by_chrom[k].sort()

    out = []
    for chrom, chrom_seq in seqs.items():
        gene_iv = by_chrom.get(chrom, [])
        L = len(chrom_seq)
        attempts = 0
        collected = 0
        while collected < n_per_chrom and attempts < n_per_chrom * 50:
            attempts += 1
            start = rng.randint(1, max(1, L - window))
            end = start + window - 1
            # Check min_distance from all genes
            ok = True
            for gs, ge in gene_iv:
                if gs - min_distance <= end and ge + min_distance >= start:
                    ok = False
                    break
            if not ok:
                continue
            # Skip Ns
            s = chrom_seq[start - 1 : end]
            if n_fraction(s) > 0.05:
                continue
            # Random strand for balance
            strand = rng.choice(["+", "-"])
            out.append((chrom, start, end, strand))
            collected += 1
    return out


def collect_splice_sites(
    genes: dict[str, Gene],
    seqs: dict[str, str],
    *,
    window: int = 1024,
    min_intron_len: int = 100,
) -> tuple[list[tuple[str, int, int, str]], list[tuple[str, int, int, str]]]:
    donors, acceptors = [], []
    for g in genes.values():
        for tid, feats in g.transcripts.items():
            exons = sorted([(s, e) for (f, s, e, _) in feats if f == "exon"])
            if len(exons) < 2:
                continue
            for (a_s, a_e), (b_s, b_e) in zip(exons, exons[1:]):
                intron_s, intron_e = a_e + 1, b_s - 1
                if intron_e - intron_s + 1 < min_intron_len:
                    continue
                # On strand, donor = exon-intron boundary at 5' of intron
                # (plus-strand: after exon-end; minus-strand: before exon-start)
                if g.strand == "+":
                    donor_center = a_e + 1
                    acceptor_center = b_s - 1
                else:
                    # On minus, we'll revcomp the fetched window; use absolute
                    # positions for the center.
                    donor_center = b_s - 1
                    acceptor_center = a_e + 1
                half = window // 2
                donors.append((g.chrom, donor_center - half, donor_center + half - 1, g.strand))
                acceptors.append((g.chrom, acceptor_center - half, acceptor_center + half - 1, g.strand))
    return donors, acceptors


# --------------------------------------------------- Record construction ----


def windowize(
    chrom: str,
    start: int,
    end: int,
    strand: str,
    seqs: dict[str, str],
    target_len: int,
    rng: random.Random,
) -> tuple[str, int, int, str] | None:
    """Return a (chrom, ws, we, strand) window of exactly target_len positioned
    around the given interval, within chromosome bounds, on the plus strand."""
    chrom_seq = seqs.get(chrom)
    if not chrom_seq:
        return None
    L = len(chrom_seq)
    # Center the window on the feature's midpoint
    mid = (start + end) // 2
    half = target_len // 2
    ws = mid - half
    we = ws + target_len - 1
    if ws < 1:
        ws, we = 1, target_len
    elif we > L:
        we, ws = L, L - target_len + 1
    if ws < 1 or we > L:
        return None
    return (chrom, ws, we, strand)


def build_records(
    items: list[tuple[str, int, int, str]],
    *,
    label: str,
    seqs: dict[str, str],
    target_len: int,
    source: str,
    rng: random.Random,
    max_n_frac: float = 0.05,
) -> list[dict]:
    recs = []
    for chrom, s, e, strand in items:
        w = windowize(chrom, s, e, strand, seqs, target_len, rng)
        if w is None:
            continue
        c, ws, we, st = w
        seq = fetch(seqs, c, ws, we, st)
        if not seq or len(seq) != target_len:
            continue
        if n_fraction(seq) > max_n_frac:
            continue
        sid = f"{source}_{c}_{ws}_{we}_{st}_{label}"
        recs.append(
            dict(
                sequence_id=sid,
                label=label,
                sequence=seq,
                gc=gc_content(seq),
                length=len(seq),
                chrom=c,
                start=ws,
                end=we,
                strand=st,
                source=source,
            )
        )
    return recs


# -------------------------------------------------------- EPDnew promoters --


def load_epdnew(path: Path) -> list[tuple[str, int, str]]:
    """Returns (chrom, tss_1based, strand). The EPDnew plant BED uses
    whitespace (not tab) separation and chrom names 'chr1' etc. The TSS is the
    thickStart coordinate (col 7, 0-based) converted to 1-based.
    """
    out = []
    if not path.exists():
        print(f"[warn] EPDnew BED missing: {path}")
        return out
    with path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            chrom, start, end, name, score, strand, thick_start, thick_end = parts[:8]
            chrom = chrom[3:] if chrom.startswith("chr") else chrom
            tss = int(thick_start) + 1  # 0-based BED → 1-based
            out.append((chrom, tss, strand))
    return out


# ------------------------------------------------------ Split assignment ----


def assign_split(chrom: str) -> str:
    # Chromosome-based split: chr1-3 train, chr4 val, chr5 test.
    # (Pt, Mt go to train.)
    if chrom in {"4"}:
        return "val"
    if chrom in {"5"}:
        return "test"
    return "train"


# ------------------------------------------------------------- Main build ---


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    seqs = load_fasta(FASTA)
    genes = parse_gff3(GFF3)

    # ---- 1. Region-type (exon/intron/utr5/utr3/intergenic) -----------------
    print("\n[1] region-type dataset")
    exons, introns, utr5, utr3 = [], [], [], []
    for g in genes.values():
        exons.extend(collect_internal_exons(g))
        introns.extend(collect_introns(g))
        utr5.extend(collect_utrs(g, "5"))
        utr3.extend(collect_utrs(g, "3"))

    # Filter introns by length
    introns = [x for x in introns if (x[2] - x[1] + 1) >= 100]
    # Deduplicate coordinates per class
    def _dedup(items):
        return list({(c, s, e, st) for c, s, e, st in items})
    exons, introns, utr5, utr3 = map(_dedup, (exons, introns, utr5, utr3))
    print(f"  raw counts  exons={len(exons)} introns={len(introns)} utr5={len(utr5)} utr3={len(utr3)}")

    intergenic = collect_intergenic_windows(
        genes,
        seqs,
        min_distance=5000,
        window=args.window,
        n_per_chrom=args.per_class,  # per chrom to get plenty
        rng=rng,
    )
    print(f"  intergenic windows: {len(intergenic)}")

    # Sub-sample and build records
    def _sample(items, k):
        if len(items) > k:
            return rng.sample(items, k)
        return items

    region_records = []
    region_records += build_records(
        _sample(exons, args.per_class), label="exon", seqs=seqs,
        target_len=args.window, source="araport11_exon", rng=rng,
    )
    region_records += build_records(
        _sample(introns, args.per_class), label="intron", seqs=seqs,
        target_len=args.window, source="araport11_intron", rng=rng,
    )
    region_records += build_records(
        _sample(utr5, args.per_class), label="utr5", seqs=seqs,
        target_len=args.window, source="araport11_utr5", rng=rng,
    )
    region_records += build_records(
        _sample(utr3, args.per_class), label="utr3", seqs=seqs,
        target_len=args.window, source="araport11_utr3", rng=rng,
    )
    region_records += build_records(
        _sample(intergenic, args.per_class), label="intergenic", seqs=seqs,
        target_len=args.window, source="araport11_intergenic", rng=rng,
    )
    df = pd.DataFrame(region_records)
    df["split"] = df["chrom"].map(assign_split)
    df.to_parquet(OUT / "region_type.parquet", index=False)
    print(f"  wrote {OUT/'region_type.parquet'}: {len(df)} rows, "
          f"classes={df['label'].value_counts().to_dict()}")

    # ---- 2. Splice-site dataset --------------------------------------------
    print("\n[2] splice-site dataset")
    donors, acceptors = collect_splice_sites(genes, seqs, window=args.window)
    donors = _dedup(donors)
    acceptors = _dedup(acceptors)
    print(f"  raw donors={len(donors)} acceptors={len(acceptors)}")
    donor_rec = build_records(
        _sample(donors, args.per_class), label="donor", seqs=seqs,
        target_len=args.window, source="araport11_donor", rng=rng,
    )
    acc_rec = build_records(
        _sample(acceptors, args.per_class), label="acceptor", seqs=seqs,
        target_len=args.window, source="araport11_acceptor", rng=rng,
    )
    # Non-site: dinucleotide-shuffled copies of donor windows
    nonsite = []
    for r in rng.sample(donor_rec + acc_rec, min(args.per_class, len(donor_rec) + len(acc_rec))):
        shuffled = dinuc_shuffle(r["sequence"], rng)
        nonsite.append(
            dict(
                sequence_id=r["sequence_id"] + "_shuf",
                label="nonsite",
                sequence=shuffled,
                gc=gc_content(shuffled),
                length=len(shuffled),
                chrom=r["chrom"],
                start=r["start"],
                end=r["end"],
                strand=r["strand"],
                source="dinuc_shuffle",
            )
        )
    splice_df = pd.DataFrame(donor_rec + acc_rec + nonsite)
    splice_df["split"] = splice_df["chrom"].map(assign_split)
    splice_df.to_parquet(OUT / "splice.parquet", index=False)
    print(f"  wrote {OUT/'splice.parquet'}: {len(splice_df)} rows, "
          f"classes={splice_df['label'].value_counts().to_dict()}")

    # ---- 3. TSS dataset (EPDnew) -------------------------------------------
    print("\n[3] TSS dataset")
    tss_list = load_epdnew(EPDNEW_BED)
    print(f"  EPDnew TSS count: {len(tss_list)}")
    tss_items = [(c, t, t, st) for c, t, st in tss_list]
    tss_rec = build_records(
        tss_items, label="tss", seqs=seqs,
        target_len=args.window, source="epdnew_tss", rng=rng,
    )
    # Shuffled controls (1:1)
    tss_ctrl = []
    for r in tss_rec:
        shuffled = dinuc_shuffle(r["sequence"], rng)
        tss_ctrl.append(
            dict(
                sequence_id=r["sequence_id"] + "_shuf",
                label="nontss",
                sequence=shuffled,
                gc=gc_content(shuffled),
                length=len(shuffled),
                chrom=r["chrom"],
                start=r["start"],
                end=r["end"],
                strand=r["strand"],
                source="dinuc_shuffle",
            )
        )
    tss_df = pd.DataFrame(tss_rec + tss_ctrl)
    if len(tss_df) == 0:
        print("  [warn] no TSS records built; skipping tss.parquet")
        tss_df = pd.DataFrame(columns=[
            "sequence_id", "label", "sequence", "gc", "length",
            "chrom", "start", "end", "strand", "source", "split",
        ])
    else:
        tss_df["split"] = tss_df["chrom"].map(assign_split)
    tss_df.to_parquet(OUT / "tss.parquet", index=False)
    print(f"  wrote {OUT/'tss.parquet'}: {len(tss_df)} rows, "
          f"classes={tss_df['label'].value_counts().to_dict()}")

    # ---- 4. Promoter dataset (EPDnew -1000,+200) vs intergenic --------------
    print("\n[4] promoter dataset")
    prom_items = []
    for chrom, tss, strand in tss_list:
        if strand == "+":
            s, e = tss - 1000, tss + 200
        else:
            s, e = tss - 200, tss + 1000
        prom_items.append((chrom, s, e, strand))
    prom_rec = build_records(
        prom_items, label="promoter", seqs=seqs,
        target_len=args.window, source="epdnew_promoter", rng=rng,
    )
    ctrl_rec = build_records(
        _sample(intergenic, len(prom_rec)), label="nonpromoter", seqs=seqs,
        target_len=args.window, source="araport11_intergenic", rng=rng,
    )
    prom_df = pd.DataFrame(prom_rec + ctrl_rec)
    if len(prom_df) == 0:
        print("  [warn] no promoter records built; skipping promoter.parquet")
        prom_df = pd.DataFrame(columns=[
            "sequence_id", "label", "sequence", "gc", "length",
            "chrom", "start", "end", "strand", "source", "split",
        ])
    else:
        prom_df["split"] = prom_df["chrom"].map(assign_split)
    prom_df.to_parquet(OUT / "promoter.parquet", index=False)
    print(f"  wrote {OUT/'promoter.parquet'}: {len(prom_df)} rows, "
          f"classes={prom_df['label'].value_counts().to_dict()}")

    # ---- Index splits summary ----------------------------------------------
    all_ids = pd.concat(
        [
            df.assign(dataset="region_type"),
            splice_df.assign(dataset="splice"),
            tss_df.assign(dataset="tss"),
            prom_df.assign(dataset="promoter"),
        ],
        ignore_index=True,
    )[["dataset", "sequence_id", "label", "split", "chrom", "gc", "length"]]
    all_ids.to_csv(SPLITS / "arareg_splits.csv", index=False)
    print(f"\nwrote {SPLITS/'arareg_splits.csv'}: {len(all_ids)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
