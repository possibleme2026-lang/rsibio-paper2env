#!/bin/bash
set -euo pipefail
mkdir -p /app/output

# ---------------------------------------------------------------------------
# 1. Amortised predictor: a gradient-boosted regressor over the seven
#    surveillance summary statistics, trained on the simulation library.
# ---------------------------------------------------------------------------
cat > /app/output/amortized_predict.py <<'PYEOF'
#!/usr/bin/env python3
"""Amortised R0 predictor trained on the synthetic simulation library."""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

FEATURES = ["peak_week", "log1p_peak", "log1p_total", "log1p_case_rate",
            "obs_frac", "late_frac", "log10_N"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "amortized_model.pkl")


def train(library_path, model_path):
    df = pd.read_csv(library_path)
    model = GradientBoostingRegressor(
        random_state=0, n_estimators=400, max_depth=3, learning_rate=0.05,
        subsample=0.9,
    )
    model.fit(df[FEATURES].to_numpy(dtype=float),
              df["R0"].to_numpy(dtype=float))
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)


def predict(features_path):
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)
    with open(features_path) as fh:
        feats = json.load(fh)
    x = np.array([[float(feats[k]) for k in FEATURES]])
    print(json.dumps({"R0": float(model.predict(x)[0])}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--library", default="/app/data/simulated_library.csv")
    ap.add_argument("--features")
    args = ap.parse_args()
    if args.train:
        train(args.library, MODEL_PATH)
    elif args.features:
        predict(args.features)
    else:
        ap.error("pass --train or --features")
PYEOF

python /app/output/amortized_predict.py --train

# ---------------------------------------------------------------------------
# 2. Per-region effective-SEIR fit on the masked surveillance series.
# ---------------------------------------------------------------------------
cat > /tmp/fit_seir.py <<'PYEOF'
import json

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

WEEKS = 52
DATA = "/app/data"
OUT = "/app/output"


def simulate(beta, sigma, gamma, N, weeks=WEEKS):
    def rhs(t, y):
        S, E, I, R = y
        inc = beta * S * I / N
        return [-inc, inc - sigma * E, sigma * E - gamma * I, gamma * I]

    t_eval = np.arange(weeks, dtype=float)
    sol = solve_ivp(rhs, (0.0, float(weeks - 1)),
                    [float(N) - 1.0, 0.0, 1.0, 0.0],
                    t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9)
    return sol.y[2]


regions = pd.read_csv(f"{DATA}/regions.csv")
obs = pd.read_csv(f"{DATA}/surveillance.csv")

fit_rows = []
pred_rows = []
r0_values = []

for region in regions["region"]:
    N = float(regions.loc[regions["region"] == region, "population"].iloc[0])
    sub = obs[obs["region"] == region].sort_values("week")
    y = sub["cases"].to_numpy(dtype=float)
    mask = ~np.isnan(y)
    y_filled = np.nan_to_num(y)

    def residual(p, yv, m, n):
        pred = simulate(p[0], p[1], p[2], n)
        return pred[m] - yv[m]

    starts = ([2.0, 2.0, 1.0], [1.0, 1.5, 1.0], [3.0, 1.0, 1.5],
              [0.8, 3.0, 0.8], [1.5, 1.0, 2.0])
    best = None
    for x0 in starts:
        try:
            res = least_squares(
                residual, x0, args=(y_filled, mask, N),
                bounds=([0.05, 0.20, 0.20], [25.0, 8.0, 8.0]),
                method="trf", xtol=1e-10, ftol=1e-10,
            )
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res

    beta, sigma, gamma = (float(v) for v in best.x)
    pred = simulate(beta, sigma, gamma, N)
    rmse = float(np.sqrt(np.mean((pred[mask] - y_filled[mask]) ** 2)))
    r0 = beta / gamma
    r0_values.append(r0)

    fit_rows.append({"region": region, "beta": beta, "sigma": sigma,
                     "gamma": gamma, "R0": r0, "fit_rmse": rmse})
    for w in range(WEEKS):
        pred_rows.append({"region": region, "week": w,
                          "predicted_cases": float(pred[w])})

pd.DataFrame(fit_rows).to_csv(f"{OUT}/fits.csv", index=False)
pd.DataFrame(pred_rows).to_csv(f"{OUT}/predictions.csv", index=False)

arr = np.array(r0_values)
rng = np.random.default_rng(0)
boots = np.array([rng.choice(arr, size=arr.size, replace=True).mean()
                  for _ in range(4000)])
pooled = {
    "pooled_R0": float(arr.mean()),
    "pooled_R0_ci": [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))],
}
with open(f"{OUT}/pooled.json", "w") as fh:
    json.dump(pooled, fh, indent=2)
print(json.dumps(pooled))
PYEOF

python /tmp/fit_seir.py

ls -l /app/output
