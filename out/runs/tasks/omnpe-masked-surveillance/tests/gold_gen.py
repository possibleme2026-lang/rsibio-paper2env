"""Hidden gold data for the OM-NPE masked-surveillance task.

This module is verifier-side only. It knows the true generating R0 of the six
surveillance regions and can regenerate, from a protected seed, a set of
held-out synthetic regions used to test the agent's amortised predictor.
"""

import numpy as np
from scipy.integrate import solve_ivp

WEEKS = 52
HOLDOUT_SEED = 987654321
HOLDOUT_N = 20

# name, population, R0_true, gamma, sigma  (beta = R0_true * gamma)
REGION_PARAMS = [
    ("Tahiti", 192000, 2.75, 1.05, 2.10),
    ("Moorea", 17500, 1.60, 0.85, 1.70),
    ("Bora-Bora", 10500, 2.90, 1.25, 2.40),
    ("Raiatea", 12500, 1.75, 0.95, 1.90),
    ("Nuku_Hiva", 3000, 2.35, 1.15, 2.60),
    ("Tubuai", 2000, 1.55, 0.75, 1.55),
]

TRUE_R0 = {p[0]: p[2] for p in REGION_PARAMS}


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
    p = float(rng.uniform(0.45, 1.0))
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


def generate_holdout(n=HOLDOUT_N):
    """Return (feature dicts, true R0 values) for held-out synthetic regions."""
    rng = np.random.default_rng(HOLDOUT_SEED)
    feats, truths = [], []
    for _ in range(n):
        N = float(np.exp(rng.uniform(np.log(2000.0), np.log(200000.0))))
        r0 = float(rng.uniform(1.5, 3.0))
        gamma = float(rng.uniform(0.6, 2.0))
        sigma = float(rng.uniform(1.0, 3.5))
        cases = simulate(r0 * gamma, sigma, gamma, N)
        noisy = np.maximum(0.0, np.round(cases * np.exp(rng.normal(0.0, 0.06, WEEKS))))
        mask = draw_mask(rng, 20)
        feats.append(compute_features(noisy, mask, N))
        truths.append(r0)
    return feats, truths


if __name__ == "__main__":
    f, t = generate_holdout()
    print(len(f), "holdout regions; R0 range", round(min(t), 3), round(max(t), 3))
