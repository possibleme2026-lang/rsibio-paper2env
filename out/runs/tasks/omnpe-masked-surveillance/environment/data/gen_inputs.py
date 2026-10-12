#!/usr/bin/env python3
"""Deterministic generator for the OM-NPE masked-surveillance task inputs.

Writes surveillance.csv, regions.csv and simulated_library.csv next to this
script. The true R0 of the six surveillance regions is deliberately NOT
written to disk, and this generator is deleted from the image after it runs.
"""
import os

import numpy as np
from scipy.integrate import solve_ivp

OUTDIR = os.path.dirname(os.path.abspath(__file__))
WEEKS = 52
SEED = 20260803
LIB_SEED = 13579

# name, population, R0, gamma, sigma   (beta = R0 * gamma)
REGION_PARAMS = [
    ("Tahiti", 192000, 2.75, 1.05, 2.10),
    ("Moorea", 17500, 1.60, 0.85, 1.70),
    ("Bora-Bora", 10500, 2.90, 1.25, 2.40),
    ("Raiatea", 12500, 1.75, 0.95, 1.90),
    ("Nuku_Hiva", 3000, 2.35, 1.15, 2.60),
    ("Tubuai", 2000, 1.55, 0.75, 1.55),
]


def simulate(beta, sigma, gamma, N, weeks=WEEKS):
    """Effective SEIR on a weekly grid; returns I at weeks 0..weeks-1."""
    def rhs(t, y):
        S, E, I, R = y
        inc = beta * S * I / N
        return [-inc, inc - sigma * E, sigma * E - gamma * I, gamma * I]

    t_eval = np.arange(weeks, dtype=float)
    sol = solve_ivp(rhs, (0.0, float(weeks - 1)),
                    [float(N) - 1.0, 0.0, 1.0, 0.0],
                    t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9)
    return sol.y[2]


def draw_mask(rng, min_obs):
    p = float(rng.uniform(0.6, 0.95))
    mask = rng.random(WEEKS) < p
    if int(mask.sum()) < min_obs:
        mask[rng.choice(WEEKS, min_obs, replace=False)] = True
    return mask


def compute_features(cases, mask, N):
    idx = np.where(mask)[0]
    y = cases[idx]
    peak_i = int(idx[int(np.argmax(y))])
    early = float(y[idx < 26].sum())
    late = float(y[idx >= 26].sum())
    total = float(y.sum())
    return {
        "peak_week": peak_i / float(WEEKS - 1),
        "log1p_peak": float(np.log1p(y.max())),
        "log1p_total": float(np.log1p(total)),
        "log1p_case_rate": float(np.log1p(total / N)),
        "obs_frac": float(mask.mean()),
        "late_frac": float(late / (early + late + 1.0)),
        "log10_N": float(np.log10(N)),
    }


FEATURE_COLS = ["peak_week", "log1p_peak", "log1p_total", "log1p_case_rate",
                "obs_frac", "late_frac", "log10_N"]


def main():
    rng = np.random.default_rng(SEED)
    with open(os.path.join(OUTDIR, "surveillance.csv"), "w") as fh:
        fh.write("region,week,cases\n")
        for name, N, r0, gamma, sigma in REGION_PARAMS:
            cases = simulate(r0 * gamma, sigma, gamma, N)
            noisy = np.maximum(
                0.0, np.round(cases * np.exp(rng.normal(0.0, 0.06, WEEKS))))
            mask = draw_mask(rng, 30)
            for w in range(WEEKS):
                if mask[w]:
                    fh.write("%s,%d,%d\n" % (name, w, int(noisy[w])))
                else:
                    fh.write("%s,%d,\n" % (name, w))

    with open(os.path.join(OUTDIR, "regions.csv"), "w") as fh:
        fh.write("region,population\n")
        for name, N, _, _, _ in REGION_PARAMS:
            fh.write("%s,%d\n" % (name, N))

    lrng = np.random.default_rng(LIB_SEED)
    rows = []
    for i in range(250):
        N = float(np.exp(lrng.uniform(np.log(2000.0), np.log(200000.0))))
        r0 = float(lrng.uniform(1.5, 3.0))
        gamma = float(lrng.uniform(0.6, 2.0))
        sigma = float(lrng.uniform(1.0, 3.5))
        cases = simulate(r0 * gamma, sigma, gamma, N)
        noisy = np.maximum(
            0.0, np.round(cases * np.exp(lrng.normal(0.0, 0.06, WEEKS))))
        mask = draw_mask(lrng, 20)
        feat = compute_features(noisy, mask, N)
        rows.append((i, feat, r0))

    with open(os.path.join(OUTDIR, "simulated_library.csv"), "w") as fh:
        fh.write("sim_id," + ",".join(FEATURE_COLS) + ",R0\n")
        for i, feat, r0 in rows:
            fh.write("s%04d," % i + ",".join(
                "%.6f" % feat[c] for c in FEATURE_COLS) + ",%.6f\n" % r0)

    print("wrote surveillance.csv, regions.csv, simulated_library.csv")


if __name__ == "__main__":
    main()
