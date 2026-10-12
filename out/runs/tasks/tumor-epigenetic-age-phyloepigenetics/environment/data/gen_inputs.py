#!/usr/bin/env python3
"""Deterministic input generator for the tumour epigenetic-age benchmark task.

Writes beta_matrix.tsv, probe_annotation.csv, sample_metadata.csv,
clock_coefficients.csv and clock_model.json into --outdir.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np

SEED = 20260729
N_PROBES = 550
N_CLOCK = 353
N_CHRX = 35
N_CHRY = 5
N_PATIENTS = 60
ADULT_ANCHOR = 0.887
PLANT_DELTA = 0.45
N_PLANT = 6
EVENT_Q_CHRX = 0.10
EVENT_Q_OTHER = 0.025

CANCER = [
    ("UCEC", "EB", 0.32), ("LUSC", "33", 0.25), ("HNSC", "CN", 0.22),
    ("PRAD", "EJ", 0.18), ("KIRC", "B0", -0.25), ("THCA", "EL", -0.32),
    ("BRCA", "A2", 0.02), ("COAD", "AA", -0.02),
]

CHR_LEN = {
    "chr1": 249, "chr2": 243, "chr3": 198, "chr4": 190, "chr5": 181,
    "chr6": 171, "chr7": 159, "chr8": 146, "chr9": 141, "chr10": 135,
    "chr11": 135, "chr12": 133, "chr13": 114, "chr14": 107, "chr15": 102,
    "chr16": 90, "chr17": 83, "chr18": 80, "chr19": 59, "chr20": 64,
    "chr21": 47, "chr22": 51, "chrX": 156, "chrY": 57,
}

GENES = ["TP53", "EGFR", "MYC", "PTEN", "RB1", "KRAS", "BRCA1", "BRCA2", "CDH1",
         "APC", "MLH1", "VHL", "NF1", "PIK3CA", "ARID1A", "CTNNB1", "SMAD4",
         "STK11", "KEAP1", "NOTCH1", "CDKN2A", "GATA3", "FOXA1", "SOX2", "TERT",
         "IDH1", "IDH2", "SETD2", "BAP1", "PBRM1"]


def f_inv(age):
    age = float(age)
    if age >= 20.0:
        return math.log((age + 1.0) / 21.0)
    return (age - 20.0) / 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/app/data")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out = a.outdir
    os.makedirs(out, exist_ok=True)

    # ---------------- probe panel ----------------
    chrom = ["chrX"] * N_CHRX + ["chrY"] * N_CHRY
    names = [c for c in CHR_LEN if c not in ("chrX", "chrY")]
    w = np.array([CHR_LEN[c] for c in names], float)
    w = w / w.sum()
    for c, k in zip(names, rng.multinomial(N_PROBES - N_CHRX - N_CHRY, w)):
        chrom.extend([c] * int(k))
    chrom = list(chrom)
    rng.shuffle(chrom)
    probes = []
    for i, c in enumerate(chrom):
        probes.append({
            "probe_id": "cg%08d" % (10000000 + 17 * (i + 1)),
            "chromosome": c,
            "position": int(rng.integers(1, CHR_LEN[c] * 1000000)),
            "gene_symbol": GENES[i % len(GENES)] if rng.random() < 0.8 else "",
        })
    chrom_arr = np.array([p["chromosome"] for p in probes])
    probe_ids = [p["probe_id"] for p in probes]

    auto = [i for i in range(N_PROBES) if chrom_arr[i] not in ("chrX", "chrY")]
    clock_idx = auto[:N_CLOCK]
    is_clock = np.zeros(N_PROBES, bool)
    is_clock[clock_idx] = True
    bg_idx = [i for i in range(N_PROBES) if not is_clock[i]]

    # ---------------- clock model ----------------
    coef = np.round(rng.normal(0.0, 0.175, N_CLOCK), 6)
    h = np.abs(rng.normal(0.0, 1.0, N_CLOCK))
    scale = float(np.sum(np.abs(coef) * h))
    L = np.sign(coef) * h / scale
    base_clock = rng.uniform(0.25, 0.75, N_CLOCK)
    intercept = ADULT_ANCHOR - float(np.sum(coef * base_clock))
    reverse = rng.random(N_CLOCK) < 0.5
    base_bg = rng.uniform(0.20, 0.80, len(bg_idx))

    # ---------------- patients / samples ----------------
    patients = []
    for ct, tss, shift in CANCER:
        for k in range(N_PATIENTS):
            patients.append({
                "patient_id": "TCGA-%s-%04d" % (tss, 1000 + k),
                "cancer_type": ct,
                "age": int(np.clip(rng.normal(63.0, 12.0), 25, 90)),
                "sex": "male" if rng.random() < 0.5 else "female",
                "shift": shift,
            })
    n_pat = len(patients)
    n_samp = 2 * n_pat

    xt = np.zeros(n_samp)
    for i, p in enumerate(patients):
        xn = f_inv(p["age"]) + rng.normal(0.0, 0.30)
        xt[2 * i] = xn
        xt[2 * i + 1] = xn + p["shift"] + rng.normal(0.0, 0.06)

    Bp = np.zeros((n_samp, N_PROBES))
    Bp[:, clock_idx] = (base_clock[None, :]
                        + L[None, :] * (xt[:, None] - ADULT_ANCHOR)
                        + rng.normal(0.0, 0.015, (n_samp, N_CLOCK)))
    Bp[:, bg_idx] = base_bg[None, :] + rng.normal(0.0, 0.02, (n_samp, len(bg_idx)))

    # ---------------- planted trunk events (background probes only) -------
    bg_pos = {g: j for j, g in enumerate(bg_idx)}
    cand = [g for g in bg_idx
            if chrom_arr[g] not in ("chrX", "chrY")
            and 0.40 <= base_bg[bg_pos[g]] <= 0.60]
    planted = {}
    for ct, tss, shift in CANCER:
        planted[ct] = sorted(int(x) for x in rng.choice(cand, size=N_PLANT, replace=False))
    sign = {}
    for ct in planted:
        for g in planted[ct]:
            sign[(ct, g)] = 1.0 if rng.random() < 0.5 else -1.0

    q = np.where(chrom_arr == "chrX", EVENT_Q_CHRX, EVENT_Q_OTHER)
    q[chrom_arr == "chrY"] = 0.0

    rows_of = {}
    for i, p in enumerate(patients):
        rows_of.setdefault(p["cancer_type"], []).append(2 * i + 1)

    for ct in rows_of:
        plant = set(planted[ct])
        for r in rows_of[ct]:
            m = rng.random(N_PROBES) < q
            for g in plant:
                m[g] = False
            mag = rng.uniform(0.32, 0.60, N_PROBES)
            sgn = np.where(rng.random(N_PROBES) < 0.5, -1.0, 1.0)
            Bp[r, m] += (mag * sgn)[m]
            for g in plant:
                Bp[r, g] += sign[(ct, g)] * PLANT_DELTA

    # ---------------- sex effect on chrX ----------------
    chrx = chrom_arr == "chrX"
    for r in range(n_samp):
        if patients[r // 2]["sex"] == "female":
            Bp[r, chrx] += 0.08

    Bp = np.clip(Bp, 0.005, 0.995)
    is_rev = np.zeros(N_PROBES, bool)
    is_rev[clock_idx] = reverse
    beta = np.where(is_rev[None, :], 1.0 - Bp, Bp)

    # ---------------- missingness ----------------
    missing = np.zeros((n_samp, N_PROBES), bool)
    chry = chrom_arr == "chrY"
    for r in range(n_samp):
        if patients[r // 2]["sex"] == "female":
            missing[r, chry] = True

    planted_all = set()
    for ct in planted:
        planted_all.update(planted[ct])
    spare = [g for g in bg_idx
             if g not in planted_all and chrom_arr[g] not in ("chrX", "chrY")]
    extra = [int(x) for x in rng.choice(spare, size=8, replace=False)]
    for g in extra:
        f = rng.uniform(0.03, 0.08)
        missing[:, g] |= rng.random(n_samp) < f

    protected = planted_all | set(extra)
    for j in range(N_PROBES):
        if chry[j] or j in protected:
            continue
        missing[:, j] |= rng.random(n_samp) < 0.003

    # ---------------- write ----------------
    sample_ids = []
    for p in patients:
        sample_ids.append(p["patient_id"] + "-11A")
        sample_ids.append(p["patient_id"] + "-01A")

    with open(os.path.join(out, "beta_matrix.tsv"), "w") as fh:
        fh.write("sample_id\t" + "\t".join(probe_ids) + "\n")
        for r in range(n_samp):
            vals = ["NA" if missing[r, j] else "%.3f" % beta[r, j] for j in range(N_PROBES)]
            fh.write(sample_ids[r] + "\t" + "\t".join(vals) + "\n")

    with open(os.path.join(out, "probe_annotation.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["probe_id", "chromosome", "position", "gene_symbol"])
        for p in probes:
            wtr.writerow([p["probe_id"], p["chromosome"], p["position"], p["gene_symbol"]])

    with open(os.path.join(out, "sample_metadata.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["sample_id", "patient_id", "cancer_type", "tissue",
                      "chronological_age_years", "sex", "array_batch"])
        for i, p in enumerate(patients):
            batch = "B%d" % (1 + (i % 4))
            wtr.writerow([p["patient_id"] + "-11A", p["patient_id"], p["cancer_type"],
                          "normal", p["age"], p["sex"], batch])
            wtr.writerow([p["patient_id"] + "-01A", p["patient_id"], p["cancer_type"],
                          "tumor", p["age"], p["sex"], batch])

    with open(os.path.join(out, "clock_coefficients.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["probe_id", "coefficient", "transform"])
        for k, j in enumerate(clock_idx):
            wtr.writerow([probe_ids[j], "%.6f" % coef[k],
                          "one_minus_beta" if reverse[k] else "beta"])

    with open(os.path.join(out, "clock_model.json"), "w") as fh:
        json.dump({
            "intercept": round(intercept, 10),
            "beta_transform": "b' = beta if transform=='beta' else 1-beta",
            "age_transform": "DNAmAge = 21*exp(x)-1 if x>=0 else 20+20*x",
        }, fh, indent=2)

    print("wrote %d samples x %d probes into %s" % (n_samp, N_PROBES, out))


if __name__ == "__main__":
    main()
