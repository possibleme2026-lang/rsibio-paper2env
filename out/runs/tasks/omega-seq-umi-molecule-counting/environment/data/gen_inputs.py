#!/usr/bin/env python3
"""Deterministic input generator for the Omega-seq molecule-counting task.

Writes, next to this file:
  annotation.gtf      gene/exon models (Drosophila-like genes + ERCC spike-ins)
  aligned.sam         UMI-tagged, cell-barcoded single-end alignments
  library_meta.tsv    cell barcode -> library name / input amount
  ercc_expected.tsv   ERCC expected molecule counts for library IN1000

No ground-truth molecule labels are written: the reference answer is defined
purely by applying the documented deduplication rules to aligned.sam. This
script is deleted from the image during the Docker build.
"""
import os
import random

SEED = 20260312
BASES = "ACGT"
HERE = os.path.dirname(os.path.abspath(__file__))

FLY_GENES = [
    ("dpp",  "chr2L", "+", 120000, 128000, [(120000, 120920), (123500, 124300), (127200, 128000)]),
    ("wg",   "chr2L", "-", 210000, 216500, [(210000, 210800), (213000, 213900), (215700, 216500)]),
    ("numb", "chr2L", "+", 305000, 311000, [(305000, 305700), (308000, 308900), (310100, 311000)]),
    ("pros", "chr3R", "+", 410000, 418000, [(410000, 410950), (414000, 414800), (417000, 418000)]),
    ("dpn",  "chr3R", "-", 505000, 511500, [(505000, 505900), (508500, 509400), (510700, 511500)]),
    ("ase",  "chr3R", "+", 620000, 626000, [(620000, 620800), (623000, 623900), (625100, 626000)]),
]
FLY_ABUND = {"dpp": 0.28, "wg": 0.14, "numb": 0.24,
             "pros": 0.11, "dpn": 0.13, "ase": 0.10}

LIBS = [
    ("AAACCTGAGAAACCAT-1", "NC1",    0.0,    1),
    ("AAACCTGAGAAACCAT-2", "IN10",   10.0,   45),
    ("AAACCTGAGAAACCAT-3", "IN100",  100.0,  260),
    ("AAACCTGAGAAACCAT-4", "IN1000", 1000.0, 900),
    ("AAACCTGAGAAACCAT-5", "NB1",    200.0,  420),
    ("AAACCTGAGAAACCAT-6", "NB2",    400.0,  640),
]
N_ERCC = 20
ERCC_EFF = 0.6


def ercc_expected():
    out = {}
    lo, hi = 2.0, 500.0
    for i in range(N_ERCC):
        out["ERCC-%05d" % (i + 1)] = round(lo * (hi / lo) ** (i / (N_ERCC - 1)), 3)
    return out


def gene_table():
    genes = []
    for name, chrom, strand, s, e, exons in FLY_GENES:
        genes.append(dict(name=name, chrom=chrom, strand=strand, start=s, end=e,
                          exons=exons, kind="fly"))
    for i in range(N_ERCC):
        st = 1000 + i * 3000
        genes.append(dict(name="ERCC-%05d" % (i + 1), chrom="chrERCC",
                          strand="+", start=st, end=st + 800,
                          exons=[(st, st + 800)], kind="ercc"))
    return genes


def write_gtf(genes, path):
    with open(path, "w") as fh:
        fh.write("# Omega-seq reference annotation (1-based, inclusive)\n")
        for g in genes:
            fh.write('%s\tomega\tgene\t%d\t%d\t.\t%s\t.\tgene_id "%s"; gene_name "%s";\n'
                     % (g["chrom"], g["start"], g["end"], g["strand"], g["name"], g["name"]))
            for (s, e) in g["exons"]:
                fh.write('%s\tomega\texon\t%d\t%d\t.\t%s\t.\tgene_id "%s";\n'
                         % (g["chrom"], s, e, g["strand"], g["name"]))


def main():
    rng = random.Random(SEED)
    genes = gene_table()
    byname = {g["name"]: g for g in genes}
    exp = ercc_expected()

    write_gtf(genes, os.path.join(HERE, "annotation.gtf"))
    with open(os.path.join(HERE, "library_meta.tsv"), "w") as fh:
        fh.write("cell_barcode\tlibrary\tinput_fg\n")
        for cb, name, fg, _n in LIBS:
            fh.write("%s\t%s\t%s\n" % (cb, name, ("%g" % fg)))
    with open(os.path.join(HERE, "ercc_expected.tsv"), "w") as fh:
        fh.write("ercc_id\texpected_molecules_IN1000\n")
        for k in sorted(exp):
            fh.write("%s\t%s\n" % (k, exp[k]))

    records = []

    def sample_end(g):
        if g["kind"] == "ercc":
            off = rng.choices([0, 1, 2, 3, 5, 8], weights=[40, 20, 15, 10, 10, 5])[0]
            return g["end"] - off
        if g["strand"] == "+":
            base = g["end"]
            off = rng.choices([0, 1, 2, 3, 5, 8, 13], weights=[35, 20, 15, 10, 8, 7, 5])[0]
            e = base - off
            if g["name"] == "numb" and rng.random() < 0.15:
                e = base + 40
            return e
        base = g["start"]
        off = rng.choices([0, 1, 2, 3, 5, 8, 13], weights=[35, 20, 15, 10, 8, 7, 5])[0]
        return base + off

    def make_reads(cell, g, umi, end, n):
        for _ in range(n):
            u = umi
            if rng.random() < 0.12:
                i = rng.randrange(8)
                u = umi[:i] + rng.choice([b for b in BASES if b != umi[i]]) + umi[i + 1:]
            clip = rng.choices([0, 1, 2, 3, 4], weights=[55, 15, 12, 10, 8])[0]
            aln = 90 - clip
            if g["strand"] == "+":
                pos = end - aln + 1
                cigar = ("%dS%dM" % (clip, aln)) if clip else ("%dM" % aln)
                flag = 0
            else:
                pos = end
                cigar = ("%dM%dS" % (aln, clip)) if clip else ("%dM" % aln)
                flag = 16
            q = rng.random()
            mapq = 60 if q >= 0.07 else (10 if q >= 0.02 else 0)
            seq = "".join(rng.choice(BASES) for _ in range(90))
            records.append((cell, flag, g["chrom"], pos, mapq, cigar, seq, u))

    for cb, _lib, fg, nfly in LIBS:
        flygenes = [g for g in genes if g["kind"] == "fly"]
        weights = [FLY_ABUND[g["name"]] for g in flygenes]
        for pi in rng.choices(range(len(flygenes)), weights=weights, k=nfly):
            g = flygenes[pi]
            umi = "".join(rng.choice(BASES) for _ in range(8))
            make_reads(cb, g, umi, sample_end(g), max(1, int(rng.gauss(5.5, 2.0))))
        if fg > 0:
            scale = fg / 1000.0
            for i in range(N_ERCC):
                g = byname["ERCC-%05d" % (i + 1)]
                lam = exp[g["name"]] * ERCC_EFF * scale
                if lam < 0.3:
                    continue
                nmol = max(0, int(round(rng.gauss(lam, lam * 0.25))))
                for _ in range(nmol):
                    umi = "".join(rng.choice(BASES) for _ in range(8))
                    make_reads(cb, g, umi, sample_end(g), max(1, int(rng.gauss(5.0, 1.8))))

    gaps = [("chr2L", 150000, 200000), ("chr2L", 240000, 290000),
            ("chr3R", 450000, 490000), ("chr3R", 540000, 600000),
            ("chrERCC", 70000, 90000)]
    libcells = [l[0] for l in LIBS]
    for _ in range(int(len(records) * 0.03)):
        chrom, lo, hi = rng.choice(gaps)
        pos = rng.randrange(lo, hi)
        cb = rng.choice(libcells)
        umi = "".join(rng.choice(BASES) for _ in range(8))
        seq = "".join(rng.choice(BASES) for _ in range(90))
        records.append((cb, 0, chrom, pos, 60, "90M", seq, umi))

    rng.shuffle(records)
    with open(os.path.join(HERE, "aligned.sam"), "w") as fh:
        fh.write("@HD\tVN:1.6\tSO:unsorted\n")
        for chrom, ln in (("chr2L", 350000), ("chr3R", 700000), ("chrERCC", 100000)):
            fh.write("@SQ\tSN:%s\tLN:%d\n" % (chrom, ln))
        for i, (cb, flag, chrom, pos, mapq, cigar, seq, umi) in enumerate(records):
            fh.write("R%07d\t%d\t%s\t%d\t%d\t%s\t*\t0\t0\t%s\t%s\tCB:Z:%s\tUB:Z:%s\n"
                     % (i, flag, chrom, pos, mapq, cigar, seq, "I" * len(seq), cb, umi))


if __name__ == "__main__":
    main()
