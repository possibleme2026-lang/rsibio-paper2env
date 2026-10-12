#!/usr/bin/env python3
"""Deterministic generator for the spatial ligand-receptor heritability task.

Usage:  python gen_inputs.py [outdir]        # default ./data

All randomness comes from a fixed seed, so the produced inputs are
reproducible byte for byte.  Only inputs are written here; no results.
"""
import os
import sys
import zlib
import numpy as np

SEED = 20260329
TISSUES = ["adipose", "brain", "heart", "liver", "lung"]
TRAITS = [("LDL", "liver"), ("CAD", "heart"), ("T2D", "adipose"),
          ("BMI", "adipose"), ("SCZ", "brain"), ("IPF", "lung")]
N_SNPS, N_GENES, N_LR, N_PAIRS = 800, 200, 100, 90
N_LOCI, PER_LOCUS = 100, 8
SIDE, N_MODULE = 30, 6
CLASSES = ["cytokine", "adhesion", "chemokine", "growth_factor", "notch"]
KSCALE, KWIN = 2.0e5, 1.0e6
SIGNAL = 0.40


def rng(tag):
    return np.random.default_rng((SEED + zlib.crc32(tag.encode("utf-8"))) % (2 ** 31))


def wtab(path, header, rows):
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")


def build_reference():
    r = rng("genes")
    gchr = r.integers(1, 23, N_GENES)
    gpos = r.integers(2_000_000, 240_000_000, N_GENES)
    o = np.lexsort((gpos, gchr))
    gchr, gpos = gchr[o], gpos[o]
    gid = ["G%04d" % (i + 1) for i in range(N_GENES)]

    r = rng("snps")
    lchr = r.integers(1, 23, N_LOCI)
    lpos = r.integers(2_000_000, 240_000_000, N_LOCI)
    sc, sp = [], []
    for i in range(N_LOCI):
        for d in np.sort(r.integers(-60_000, 60_001, PER_LOCUS)):
            sc.append(int(lchr[i]))
            sp.append(int(lpos[i]) + int(d))
    schr = np.array(sc)
    spos = np.array(sp)
    o = np.lexsort((spos, schr))
    schr, spos = schr[o], spos[o]
    sid = ["rs%07d" % (1000000 + i) for i in range(N_SNPS)]
    maf = np.round(r.uniform(0.05, 0.5, N_SNPS), 3)
    return gid, gchr, gpos, sid, schr, spos, maf


def build_ld(schr, spos):
    R = np.eye(N_SNPS)
    for i in range(N_SNPS):
        for j in range(i + 1, N_SNPS):
            if schr[i] != schr[j]:
                continue
            d = abs(int(spos[i]) - int(spos[j]))
            if d > 150_000:
                continue
            if i // PER_LOCUS == j // PER_LOCUS:
                r2 = 0.70 * np.exp(-d / 25_000.0)
            else:
                r2 = 0.15 * np.exp(-d / 40_000.0)
            if r2 >= 0.005:
                R[i, j] = R[j, i] = r2
    return R


def build_pairs():
    r = rng("pairs")
    lr = np.sort(r.choice(N_GENES, N_LR, replace=False))
    pairs = []
    for p in range(N_PAIRS):
        a, b = r.choice(lr, 2, replace=False)
        pairs.append(("LR%03d" % (p + 1), int(a), int(b), CLASSES[p % len(CLASSES)]))
    return lr, pairs


def build_localization():
    r = rng("localization")
    A = np.zeros((N_GENES, len(TISSUES)))
    for g in range(N_GENES):
        for t in r.choice(len(TISSUES), int(r.integers(1, 4)), replace=False):
            A[g, t] = round(float(r.uniform(1.5, 5.0)), 3)
    return A


def build_modules(pairs):
    r = rng("modules")
    mod = r.integers(0, N_MODULE, N_GENES)
    for _, a, b, _ in pairs:
        k = int(r.integers(0, N_MODULE))
        mod[a] = k
        mod[b] = k
    return mod


def build_counts(tissue, ti, A, mod):
    r = rng("counts:" + tissue)
    xs, ys = np.meshgrid(np.arange(1, SIDE + 1), np.arange(1, SIDE + 1))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    centers = r.uniform(3.0, SIDE - 2.0, (N_MODULE, 2))
    field = np.zeros((coords.shape[0], N_GENES))
    for k in range(N_MODULE):
        d2 = ((coords[:, 0] - centers[k, 0]) ** 2 + (coords[:, 1] - centers[k, 1]) ** 2)
        kern = np.exp(-d2 / 50.0)
        for g in np.where((mod == k) & (A[:, ti] > 0))[0]:
            field[:, g] = kern * A[g, ti]
    base = 6.0 * r.lognormal(0.0, 0.7, N_GENES)
    lam = base[None, :] * (0.35 + field)
    return coords, r.poisson(lam)


def to_snp(vals, gchr, gpos, schr, spos):
    x = np.zeros(len(spos), dtype=float)
    for g in range(len(vals)):
        if vals[g] == 0.0:
            continue
        m = (gchr[g] == schr) & (np.abs(spos - gpos[g]) <= KWIN)
        if m.any():
            x[m] += vals[g] * np.exp(-np.abs(spos[m] - gpos[g]) / KSCALE)
    mu = x.mean()
    return x / mu if mu != 0 else x


def main(outdir):
    os.makedirs(os.path.join(outdir, "spatial"), exist_ok=True)
    gid, gchr, gpos, sid, schr, spos, maf = build_reference()
    R = build_ld(schr, spos)
    lr, pairs = build_pairs()
    A = build_localization()
    mod = build_modules(pairs)

    wtab(os.path.join(outdir, "genes.tsv"), ["gene_id", "chr", "tss"],
         [(gid[i], int(gchr[i]), int(gpos[i])) for i in range(N_GENES)])
    wtab(os.path.join(outdir, "snps.tsv"), ["snp_id", "chr", "pos", "maf"],
         [(sid[i], int(schr[i]), int(spos[i]), maf[i]) for i in range(N_SNPS)])
    wtab(os.path.join(outdir, "lr_pairs.tsv"),
         ["pair_id", "ligand", "receptor", "class"],
         [(pid, gid[a], gid[b], cl) for pid, a, b, cl in pairs])
    wtab(os.path.join(outdir, "ld_matrix.tsv"), ["snp_i", "snp_j", "r2"],
         [(sid[i], sid[j], round(float(R[i, j]), 5))
          for i in range(N_SNPS) for j in range(i + 1, N_SNPS) if R[i, j] > 0])

    rw = rng("intrinsic")
    wtab(os.path.join(outdir, "intrinsic_annotations.tsv"), ["gene_id", "weight"],
         [(gid[i], round(float(rw.uniform(0.4, 2.0)), 4)) for i in range(N_GENES)])

    for ti, t in enumerate(TISSUES):
        coords, counts = build_counts(t, ti, A, mod)
        hdr = ["spot_id", "x", "y"] + gid
        rows = []
        for s in range(coords.shape[0]):
            rows.append(["S%04d" % (s + 1), int(coords[s, 0]), int(coords[s, 1])]
                        + [int(v) for v in counts[s]])
        wtab(os.path.join(outdir, "spatial", "%s_counts.tsv" % t), hdr, rows)

    wtab(os.path.join(outdir, "trait_tissue.tsv"), ["trait", "matched_tissue"],
         [(tr, ti) for tr, ti in TRAITS])

    rg = rng("gwas")
    rows = []
    for tr, tis in TRAITS:
        ti = TISSUES.index(tis)
        pot = np.zeros(N_GENES)
        cntp = np.zeros(N_GENES)
        for _, a, b, _ in pairs:
            v = min(A[a, ti], A[b, ti])
            pot[a] += v
            cntp[a] += 1
            pot[b] += v
            cntp[b] += 1
        pot = np.where(cntp > 0, pot / np.maximum(cntp, 1.0), 0.0)
        c = np.zeros(N_GENES)
        for g in range(N_GENES):
            if pot[g] > 0 and rg.random() < 0.5:
                c[g] = pot[g]
        ls = R.dot(to_snp(c, gchr, gpos, schr, spos))
        mu = np.sqrt(np.maximum(SIGNAL * ls, 0.0))
        sgn = rg.choice([-1.0, 1.0], N_SNPS)
        z = rg.normal(0.0, 1.0, N_SNPS) + sgn * mu
        n = int(rg.integers(80_000, 400_000))
        for j in range(N_SNPS):
            rows.append((tr, sid[j], round(float(z[j]), 6), n))
    wtab(os.path.join(outdir, "gwas_sumstats.tsv"), ["trait", "snp_id", "z", "n"], rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
