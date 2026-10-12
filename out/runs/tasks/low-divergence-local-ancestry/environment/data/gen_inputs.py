#!/usr/bin/env python3
"""Deterministic generator for the low-divergence local-ancestry benchmark.

Model (population-frequency version of the Li-Stephens admixture simulator):

  * one 150 cM region, M = 3000 evenly spaced biallelic SNPs (map.tsv)
  * per-site ancestral frequency p0(l) ~ Uniform(0.05, 0.95)
  * p_A(l) = clip(p0(l) + delta, 0.01, 0.99),  p_B(l) = clip(p0(l) - delta, 0.01, 0.99)
    with delta calibrated by bisection so that
        FST = mean_l[(p_A-p_B)^2/4] / mean_l[pbar(1-pbar)]
    equals the level target
  * reference panels: 60 haplotypes per population, i.i.d. Bernoulli(p_s(l))
  * admixed haplotypes: two-state ancestry Markov chain along the map with
        P(switch between adjacent SNPs) = 1 - exp(-LAMBDA_TRUE * d_l)
    (d_l in Morgan), LAMBDA_TRUE = 40 switches per Morgan; alleles are then drawn
    i.i.d. Bernoulli(p_state(l)).
  * the same 4 ancestry paths are re-used at all three divergence levels.

Usage:  python gen_inputs.py --out /app/data --seed 20260730
"""
import argparse
import json
import os

import numpy as np

N_SITES = 3000
N_HAPLOTYPES = 60
N_SAMPLES = 4
MAP_CM = 150.0
LAMBDA_TRUE = 40.0
FST_TARGETS = (("fst_0002", 0.002), ("fst_0020", 0.020), ("fst_0150", 0.150))
SAMPLES = ("hap1", "hap2", "hap3", "hap4")


def _fst(p0, delta, lo=0.01, hi=0.99):
    pa = np.clip(p0 + delta, lo, hi)
    pb = np.clip(p0 - delta, lo, hi)
    pbar = 0.5 * (pa + pb)
    num = np.mean((pa - pb) ** 2 / 4.0)
    den = np.mean(pbar * (1.0 - pbar))
    return float(num / den)


def _calibrate(p0, target):
    lo, hi = 0.0, 0.5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _fst(p0, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build(seed=20260730):
    rng = np.random.default_rng(seed)
    cM = np.linspace(0.0, MAP_CM, N_SITES)
    pos = np.linspace(1, 150000000, N_SITES).astype(np.int64)
    snp_id = ["snp%05d" % (i + 1) for i in range(N_SITES)]
    dist = np.diff(cM) / 100.0
    p0 = rng.uniform(0.05, 0.95, N_SITES)

    switch_p = 1.0 - np.exp(-LAMBDA_TRUE * dist)
    truth = np.empty((N_SAMPLES, N_SITES), dtype=np.int8)
    for s in range(N_SAMPLES):
        state = int(rng.integers(0, 2))
        path = np.empty(N_SITES, dtype=np.int8)
        path[0] = state
        jumps = rng.random(N_SITES - 1) < switch_p
        for l in range(1, N_SITES):
            if jumps[l - 1]:
                state = 1 - state
            path[l] = state
        truth[s] = path

    levels = {}
    meta = []
    for name, target in FST_TARGETS:
        delta = _calibrate(p0, target)
        pa = np.clip(p0 + delta, 0.01, 0.99)
        pb = np.clip(p0 - delta, 0.01, 0.99)
        panel_a = (rng.random((N_HAPLOTYPES, N_SITES)) < pa[None, :]).astype(np.int8)
        panel_b = (rng.random((N_HAPLOTYPES, N_SITES)) < pb[None, :]).astype(np.int8)
        admixed = np.empty((N_SAMPLES, N_SITES), dtype=np.int8)
        for s in range(N_SAMPLES):
            p = np.where(truth[s] == 0, pa, pb)
            admixed[s] = (rng.random(N_SITES) < p).astype(np.int8)
        levels[name] = {"panel_a": panel_a, "panel_b": panel_b,
                        "admixed": admixed, "truth": truth}
        meta.append({"fst_level": name, "target_fst": target,
                     "realised_fst": _fst(p0, delta), "delta": float(delta)})
    return {"cM": cM, "pos": pos, "snp_id": snp_id, "levels": levels,
            "meta": meta, "lambda_true": LAMBDA_TRUE, "n_sites": N_SITES,
            "n_haplotypes": N_HAPLOTYPES, "samples": list(SAMPLES), "seed": seed}


def _write_haplotypes(path, mat):
    with open(path, "w") as fh:
        for row in mat:
            fh.write("".join("1" if v else "0" for v in row))
            fh.write(chr(10))


def write_dataset(out_dir, seed=20260730):
    data = build(seed)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "map.tsv"), "w") as fh:
        fh.write("snp_id" + chr(9) + "pos_bp" + chr(9) + "cM" + chr(10))
        for sid, p, c in zip(data["snp_id"], data["pos"], data["cM"]):
            fh.write("%s%s%d%s%.6f%s" % (sid, chr(9), p, chr(9), c, chr(10)))
    for name, lvl in data["levels"].items():
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        _write_haplotypes(os.path.join(d, "ref_A.haplotypes.txt"), lvl["panel_a"])
        _write_haplotypes(os.path.join(d, "ref_B.haplotypes.txt"), lvl["panel_b"])
        for i, s in enumerate(SAMPLES):
            with open(os.path.join(d, "admixed_%s.txt" % s), "w") as fh:
                fh.write("".join("1" if v else "0" for v in lvl["admixed"][i]) + chr(10))
            with open(os.path.join(d, "truth_%s.txt" % s), "w") as fh:
                fh.write("".join("A" if v == 0 else "B" for v in lvl["truth"][i]) + chr(10))
    manifest = {"seed": seed, "n_sites": data["n_sites"],
                "n_reference_haplotypes_per_population": data["n_haplotypes"],
                "map_length_cM": MAP_CM, "samples": list(SAMPLES),
                "switch_rate_lambda_true_per_Morgan": LAMBDA_TRUE,
                "fst_levels": data["meta"]}
    with open(os.path.join(out_dir, "simulation_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/app/data")
    ap.add_argument("--seed", type=int, default=20260730)
    args = ap.parse_args()
    write_dataset(args.out, args.seed)
    print("wrote dataset to", args.out)
