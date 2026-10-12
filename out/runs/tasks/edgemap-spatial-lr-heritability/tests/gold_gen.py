#!/usr/bin/env python3
"""Protected reference: rebuilds the inputs deterministically and recomputes the
EdgeMap gold standard (per trait/tissue z-scores and gene prioritisation).

The generator block below is byte-identical to the build-time script that is
deleted from the image after the Docker build.
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


# ---------------------------------------------------------------- pipeline


def spot_norm(counts):
    tot = counts.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    return np.log1p(1e4 * counts / tot)


def edge_list(coords):
    pos = {}
    for i in range(coords.shape[0]):
        pos[(int(coords[i, 0]), int(coords[i, 1]))] = i
    e = []
    for (x, y), i in pos.items():
        for dx, dy in ((1, 0), (0, 1)):
            j = pos.get((x + dx, y + dy))
            if j is not None:
                e.append((i, j))
    return e


def morans_i(edges, a, n):
    W = 2.0 * len(edges)
    d = a - a.mean()
    den = float((d * d).sum())
    if den <= 0 or W == 0:
        return 0.0
    s = 0.0
    for i, j in edges:
        s += d[i] * d[j]
    return (n / W) * (2.0 * s) / den


def ols(X, y):
    XtX = X.T @ X
    beta = np.linalg.pinv(XtX) @ (X.T @ y)
    resid = y - X @ beta
    s2 = float(resid @ resid) / (X.shape[0] - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(XtX)))
    return beta, se


def _load(d):
    def rows(fn):
        with open(os.path.join(d, fn)) as fh:
            txt = fh.read().strip().split("\n")
        return [ln.split("\t") for ln in txt[1:]]

    snp = rows("snps.tsv")
    sid = [r[0] for r in snp]
    sidx = {s: i for i, s in enumerate(sid)}
    schr = np.array([int(r[1]) for r in snp])
    spos = np.array([int(r[2]) for r in snp])
    gen = rows("genes.tsv")
    gid = [r[0] for r in gen]
    gidx = {g: i for i, g in enumerate(gid)}
    gchr = np.array([int(r[1]) for r in gen])
    gpos = np.array([int(r[2]) for r in gen])
    pairs = [(gidx[r[1]], gidx[r[2]]) for r in rows("lr_pairs.tsv")]
    iw = np.array([float(r[1]) for r in rows("intrinsic_annotations.tsv")])

    R = np.eye(len(sid))
    for r in rows("ld_matrix.tsv"):
        i, j = sidx[r[0]], sidx[r[1]]
        R[i, j] = R[j, i] = float(r[2])

    z = {tr: np.zeros(len(sid)) for tr, _ in TRAITS}
    for r in rows("gwas_sumstats.tsv"):
        z[r[0]][sidx[r[1]]] = float(r[2])

    counts, coords = {}, {}
    for t in TISSUES:
        with open(os.path.join(d, "spatial", "%s_counts.tsv" % t)) as fh:
            lines = fh.read().strip().split("\n")
        counts[t] = np.array([[float(x) for x in ln.split("\t")[3:]]
                              for ln in lines[1:]])
        coords[t] = np.array([[int(x) for x in ln.split("\t")[1:3]]
                              for ln in lines[1:]], dtype=float)
    return dict(sid=sid, schr=schr, spos=spos, gid=gid, gchr=gchr, gpos=gpos,
                pairs=pairs, iw=iw, R=R, z=z, counts=counts, coords=coords)


def compute_gold(datadir):
    D = _load(datadir)
    gid = D["gid"]
    nG = len(gid)
    E, INT = {}, {}
    for t in TISSUES:
        X = spot_norm(D["counts"][t])
        ed = edge_list(D["coords"][t])
        acc = np.zeros(nG)
        cnt = np.zeros(nG)
        for a, b in D["pairs"]:
            val = morans_i(ed, np.sqrt(X[:, a] * X[:, b]), X.shape[0])
            acc[a] += val
            cnt[a] += 1
            acc[b] += val
            cnt[b] += 1
        E[t] = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0)
        INT[t] = D["iw"] * X.mean(axis=0)

    annE = {t: to_snp(E[t], D["gchr"], D["gpos"], D["schr"], D["spos"])
            for t in TISSUES}
    annI = {t: to_snp(INT[t], D["gchr"], D["gpos"], D["schr"], D["spos"])
            for t in TISSUES}

    edge = {}
    for tr, _mt in TRAITS:
        y = D["z"][tr] ** 2
        for t in TISSUES:
            le = D["R"] @ annE[t]
            li = D["R"] @ annI[t]
            X = np.column_stack([np.ones(len(y)), le, li])
            b, se = ols(X, y)
            sz, iz = float(b[1] / se[1]), float(b[2] / se[2])
            edge[(tr, t)] = dict(edge_tau=float(b[1]), intrinsic_tau=float(b[2]),
                                 edge_z=sz, intrinsic_z=iz, delta_z=sz - iz)

    y = D["z"]["LDL"] ** 2
    li = D["R"] @ annI["liver"]
    genes = sorted({g for a, b in D["pairs"] for g in (a, b)})
    gene = {}
    for g in genes:
        if E["liver"][g] == 0.0:
            gene[gid[g]] = dict(edge_score=0.0, z_g=0.0)
            continue
        v = np.zeros(nG)
        v[g] = E["liver"][g]
        lv = D["R"] @ to_snp(v, D["gchr"], D["gpos"], D["schr"], D["spos"])
        X = np.column_stack([np.ones(len(y)), lv, li])
        b, se = ols(X, y)
        gene[gid[g]] = dict(edge_score=float(E["liver"][g]), z_g=float(b[1] / se[1]))

    top10 = [k for k, _ in sorted(gene.items(), key=lambda kv: (-kv[1]["z_g"], kv[0]))[:10]]
    pattern = {}
    for tr, mt in TRAITS:
        others = [t for t in TISSUES if t != mt]
        pattern[tr] = bool(edge[(tr, mt)]["edge_z"]
                           > float(np.mean([edge[(tr, t)]["edge_z"] for t in others])))
    return dict(edge=edge, gene=gene, top10=top10, pattern=pattern)


if __name__ == "__main__":
    with_none = "/tmp/gold_inputs"
    os.makedirs(with_none, exist_ok=True)
    main(with_none)
    print(compute_gold(with_none)["top10"])
