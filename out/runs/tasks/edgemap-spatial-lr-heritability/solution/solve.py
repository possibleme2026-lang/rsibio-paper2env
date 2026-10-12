#!/usr/bin/env python3
"""Reference solution: full EdgeMap-style annotation pipeline (numpy only)."""
import os
import numpy as np

DATA = "/app/data"
OUT = "/app/output"
TISSUES = ["adipose", "brain", "heart", "liver", "lung"]
TRAITS = [("LDL", "liver"), ("CAD", "heart"), ("T2D", "adipose"),
          ("BMI", "adipose"), ("SCZ", "brain"), ("IPF", "lung")]
KSCALE, KWIN = 2.0e5, 1.0e6


def rows(fn):
    with open(os.path.join(DATA, fn)) as fh:
        txt = fh.read().strip().split("\n")
    return [ln.split("\t") for ln in txt[1:]]


def ols(X, y):
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    res = y - X @ b
    s2 = float(res @ res) / (X.shape[0] - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
    return b, se


def main():
    snp = rows("snps.tsv")
    sid = [r[0] for r in snp]
    sidx = {s: i for i, s in enumerate(sid)}
    schr = np.array([int(r[1]) for r in snp])
    spos = np.array([int(r[2]) for r in snp])
    gen = rows("genes.tsv")
    gid = [r[0] for r in gen]
    gchr = np.array([int(r[1]) for r in gen])
    gpos = np.array([int(r[2]) for r in gen])
    gidx = {g: i for i, g in enumerate(gid)}
    pairs = [(gidx[r[1]], gidx[r[2]]) for r in rows("lr_pairs.tsv")]
    iw = np.array([float(r[1]) for r in rows("intrinsic_annotations.tsv")])
    ng = len(gid)

    R = np.eye(len(sid))
    for r in rows("ld_matrix.tsv"):
        i, j = sidx[r[0]], sidx[r[1]]
        R[i, j] = R[j, i] = float(r[2])

    zs = {tr: np.zeros(len(sid)) for tr, _ in TRAITS}
    for r in rows("gwas_sumstats.tsv"):
        zs[r[0]][sidx[r[1]]] = float(r[2])

    def to_snp(v):
        x = np.zeros(len(sid))
        for g in range(ng):
            if v[g] == 0:
                continue
            m = (gchr[g] == schr) & (np.abs(spos - gpos[g]) <= KWIN)
            if m.any():
                x[m] += v[g] * np.exp(-np.abs(spos[m] - gpos[g]) / KSCALE)
        mu = x.mean()
        return x / mu if mu != 0 else x

    E, INT = {}, {}
    for t in TISSUES:
        with open(os.path.join(DATA, "spatial", "%s_counts.tsv" % t)) as fh:
            lines = fh.read().strip().split("\n")
        C = np.array([[float(x) for x in ln.split("\t")[3:]] for ln in lines[1:]])
        co = [[int(x) for x in ln.split("\t")[1:3]] for ln in lines[1:]]
        X = np.log1p(1e4 * C / C.sum(axis=1, keepdims=True))
        pos = {(co[i][0], co[i][1]): i for i in range(len(co))}
        edges = []
        for (x, y), i in pos.items():
            for dx, dy in ((1, 0), (0, 1)):
                j = pos.get((x + dx, y + dy))
                if j is not None:
                    edges.append((i, j))
        n = len(co)
        W = 2.0 * len(edges)
        acc = np.zeros(ng)
        cnt = np.zeros(ng)
        for a, b in pairs:
            d = np.sqrt(X[:, a] * X[:, b])
            d = d - d.mean()
            s = sum(d[i] * d[j] for i, j in edges)
            I = (n / W) * (2.0 * s) / float((d * d).sum())
            acc[a] += I
            cnt[a] += 1
            acc[b] += I
            cnt[b] += 1
        E[t] = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0)
        INT[t] = iw * X.mean(axis=0)

    annE = {t: to_snp(E[t]) for t in TISSUES}
    annI = {t: to_snp(INT[t]) for t in TISSUES}

    os.makedirs(OUT, exist_ok=True)
    out = ["trait\ttissue\tedge_tau\tintrinsic_tau\tedge_z\tintrinsic_z\tdelta_z"]
    for tr, _mt in TRAITS:
        y = zs[tr] ** 2
        for t in TISSUES:
            Xm = np.column_stack([np.ones(len(y)), R @ annE[t], R @ annI[t]])
            b, se = ols(Xm, y)
            ez, iz = b[1] / se[1], b[2] / se[2]
            out.append("%s\t%s\t%.8f\t%.8f\t%.6f\t%.6f\t%.6f"
                       % (tr, t, b[1], b[2], ez, iz, ez - iz))
    with open(os.path.join(OUT, "edge_z_scores.tsv"), "w") as fh:
        fh.write("\n".join(out) + "\n")

    y = zs["LDL"] ** 2
    li = R @ annI["liver"]
    genes = sorted({g for pr in pairs for g in pr})
    res = []
    for g in genes:
        if E["liver"][g] == 0.0:
            res.append((gid[g], 0.0, 0.0))
            continue
        v = np.zeros(ng)
        v[g] = E["liver"][g]
        Xm = np.column_stack([np.ones(len(y)), R @ to_snp(v), li])
        b, se = ols(Xm, y)
        res.append((gid[g], E["liver"][g], b[1] / se[1]))
    res.sort(key=lambda r: (-r[2], r[0]))
    out = ["gene\tedge_score\tz_g\tprioritized"]
    for i, (g, es, zv) in enumerate(res):
        out.append("%s\t%.8f\t%.6f\t%d" % (g, es, zv, 1 if i < 10 else 0))
    with open(os.path.join(OUT, "prioritized_genes.tsv"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
