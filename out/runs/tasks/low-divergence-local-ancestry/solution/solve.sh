#!/bin/bash
set -euo pipefail
mkdir -p /app/output/posteriors /app/output/paths
python3 - <<'PY'
import json
import math
import os

import numpy as np

DATA = "/app/data"
OUT = "/app/output"
LEVELS = ["fst_0002", "fst_0020", "fst_0150"]
SAMPLES = ["hap1", "hap2", "hap3", "hap4"]
GRID = list(range(1, 121))
PSEUDO = 0.5
ALPHA = 0.5


def load_map(path):
    ids = []
    cm = []
    with open(path) as fh:
        fh.readline()
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                ids.append(parts[0])
                cm.append(float(parts[2]))
    return ids, np.asarray(cm, dtype=float)


def load_matrix(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append([1 if c == "1" else 0 for c in s])
    return np.asarray(rows, dtype=np.int8)


def load_bits(path):
    with open(path) as fh:
        s = fh.read().strip()
    return np.asarray([1 if c == "1" else 0 for c in s], dtype=np.int8)


def load_truth(path):
    with open(path) as fh:
        s = fh.read().strip()
    return np.asarray([0 if c == "A" else 1 for c in s], dtype=np.int8)


def emission(x, pA, pB):
    e = np.empty((x.shape[0], 2), dtype=float)
    e[:, 0] = np.where(x == 1, pA, 1.0 - pA)
    e[:, 1] = np.where(x == 1, pB, 1.0 - pB)
    return e


def forward_backward(e, lam, dist):
    m = e.shape[0]
    q = 1.0 - np.exp(-lam * dist)
    alpha = np.empty((m, 2))
    scale = np.empty(m)
    a0 = 0.5 * e[0, 0]
    a1 = 0.5 * e[0, 1]
    s = a0 + a1
    scale[0] = s
    alpha[0, 0] = a0 / s
    alpha[0, 1] = a1 / s
    for l in range(1, m):
        ql = q[l - 1]
        p0 = alpha[l - 1, 0]
        p1 = alpha[l - 1, 1]
        a0 = (p0 * (1.0 - ql) + p1 * ql) * e[l, 0]
        a1 = (p0 * ql + p1 * (1.0 - ql)) * e[l, 1]
        s = a0 + a1
        scale[l] = s
        alpha[l, 0] = a0 / s
        alpha[l, 1] = a1 / s
    beta = np.empty((m, 2))
    beta[m - 1, 0] = 1.0
    beta[m - 1, 1] = 1.0
    for l in range(m - 2, -1, -1):
        ql = q[l]
        b0 = (1.0 - ql) * e[l + 1, 0] * beta[l + 1, 0] + ql * e[l + 1, 1] * beta[l + 1, 1]
        b1 = ql * e[l + 1, 0] * beta[l + 1, 0] + (1.0 - ql) * e[l + 1, 1] * beta[l + 1, 1]
        s = b0 + b1
        beta[l, 0] = b0 / s
        beta[l, 1] = b1 / s
    post = alpha * beta
    post = post / post.sum(axis=1, keepdims=True)
    return post, float(np.log(scale).sum())


def loglik_grid(e, dist, grid):
    lam = np.asarray(grid, dtype=float)
    q = 1.0 - np.exp(-np.outer(lam, dist))
    alpha = np.empty((lam.shape[0], 2))
    alpha[:, 0] = 0.5 * e[0, 0]
    alpha[:, 1] = 0.5 * e[0, 1]
    s = alpha.sum(axis=1)
    alpha[:, 0] /= s
    alpha[:, 1] /= s
    ll = np.log(s)
    for l in range(1, e.shape[0]):
        ql = q[:, l - 1]
        a0 = (alpha[:, 0] * (1.0 - ql) + alpha[:, 1] * ql) * e[l, 0]
        a1 = (alpha[:, 0] * ql + alpha[:, 1] * (1.0 - ql)) * e[l, 1]
        s = a0 + a1
        alpha[:, 0] = a0 / s
        alpha[:, 1] = a1 / s
        ll = ll + np.log(s)
    return ll


def viterbi(e, lam, dist):
    m = e.shape[0]
    q = 1.0 - np.exp(-lam * dist)
    d0 = math.log(0.5) + math.log(e[0, 0])
    d1 = math.log(0.5) + math.log(e[0, 1])
    psi = np.zeros((m, 2), dtype=np.int64)
    for l in range(1, m):
        ql = q[l - 1]
        lp = math.log(1.0 - ql)
        lq = math.log(ql)
        if d0 + lp >= d1 + lq:
            b0 = d0 + lp
            psi[l, 0] = 0
        else:
            b0 = d1 + lq
            psi[l, 0] = 1
        if d0 + lq >= d1 + lp:
            b1 = d0 + lq
            psi[l, 1] = 0
        else:
            b1 = d1 + lp
            psi[l, 1] = 1
        d0 = b0 + math.log(e[l, 0])
        d1 = b1 + math.log(e[l, 1])
    path = np.empty(m, dtype=np.int8)
    s = 0 if d0 >= d1 else 1
    path[m - 1] = s
    for l in range(m - 2, -1, -1):
        s = psi[l + 1, s]
        path[l] = s
    return path


def main():
    ids, cM = load_map(os.path.join(DATA, "map.tsv"))
    dist = np.diff(cM) / 100.0
    n = len(ids)
    per_chrom = []
    for lvl in LEVELS:
        d = os.path.join(DATA, lvl)
        A = load_matrix(os.path.join(d, "ref_A.haplotypes.txt"))
        B = load_matrix(os.path.join(d, "ref_B.haplotypes.txt"))
        pA = (A.sum(axis=0) + PSEUDO) / (A.shape[0] + 1.0)
        pB = (B.sum(axis=0) + PSEUDO) / (B.shape[0] + 1.0)
        os.makedirs(os.path.join(OUT, "posteriors", lvl), exist_ok=True)
        os.makedirs(os.path.join(OUT, "paths", lvl), exist_ok=True)
        for sample in SAMPLES:
            x = load_bits(os.path.join(d, "admixed_" + sample + ".txt"))
            truth = load_truth(os.path.join(d, "truth_" + sample + ".txt"))
            e = emission(x, pA, pB)
            lls = loglik_grid(e, dist, GRID)
            lam = int(GRID[int(np.argmax(lls))])
            post, ll = forward_backward(e, lam, dist)
            ppos = np.where(post[:, 0] >= post[:, 1], 0, 1).astype(np.int8)
            vit = viterbi(e, lam, dist)
            mle = np.where(e[:, 0] >= e[:, 1], 0, 1).astype(np.int8)
            with open(os.path.join(OUT, "posteriors", lvl, sample + ".tsv"), "w") as fh:
                print("snp_id", "p_A", "p_B", sep=chr(9), file=fh)
                for i in range(n):
                    print(ids[i], "%.8f" % post[i, 0], "%.8f" % post[i, 1], sep=chr(9), file=fh)
            with open(os.path.join(OUT, "paths", lvl, sample + ".posterior.txt"), "w") as fh:
                print("".join("A" if v == 0 else "B" for v in ppos), file=fh)
            with open(os.path.join(OUT, "paths", lvl, sample + ".viterbi.txt"), "w") as fh:
                print("".join("A" if v == 0 else "B" for v in vit), file=fh)
            acc_vit = float(np.mean(vit == truth))
            acc_mle = float(np.mean(mle == truth))
            per_chrom.append({
                "fst_level": lvl,
                "sample": sample,
                "n_sites": int(n),
                "lambda_hat": int(lam),
                "log_likelihood_at_lambda_hat": float(ll),
                "implied_admixture_time_generations": float(lam) / (2.0 * ALPHA * (1.0 - ALPHA)),
                "n_switches_posterior": int(np.sum(ppos[1:] != ppos[:-1])),
                "n_switches_viterbi": int(np.sum(vit[1:] != vit[:-1])),
                "accuracy_posterior": float(np.mean(ppos == truth)),
                "accuracy_viterbi": acc_vit,
                "accuracy_per_site_mle": acc_mle,
                "hmm_beats_per_site_mle": bool(acc_vit > acc_mle),
            })
    per_level = []
    n_feasible = 0
    for lvl in LEVELS:
        rows = [r for r in per_chrom if r["fst_level"] == lvl]
        msp = float(np.mean([r["n_switches_posterior"] for r in rows]))
        msv = float(np.mean([r["n_switches_viterbi"] for r in rows]))
        macv = float(np.mean([r["accuracy_viterbi"] for r in rows]))
        feasible = bool(macv >= 0.95)
        n_feasible += int(feasible)
        per_level.append({
            "fst_level": lvl,
            "mean_lambda_hat": float(np.mean([r["lambda_hat"] for r in rows])),
            "mean_switches_posterior": msp,
            "mean_switches_viterbi": msv,
            "tract_inflation_factor": msp / max(msv, 1.0),
            "mean_accuracy_posterior": float(np.mean([r["accuracy_posterior"] for r in rows])),
            "mean_accuracy_viterbi": macv,
            "mean_accuracy_per_site_mle": float(np.mean([r["accuracy_per_site_mle"] for r in rows])),
            "feasible": feasible,
        })
    out = {
        "model": {"lambda_grid": [GRID[0], GRID[-1]], "pseudocount": PSEUDO,
                  "alpha": ALPHA, "feasibility_threshold": 0.95},
        "per_chromosome": per_chrom,
        "per_fst_level": per_level,
        "summary": {"n_fst_levels": len(LEVELS), "n_samples_per_level": len(SAMPLES),
                    "n_feasible_fst_levels": n_feasible},
    }
    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", len(per_chrom), "chromosome records")


if __name__ == "__main__":
    main()
PY
