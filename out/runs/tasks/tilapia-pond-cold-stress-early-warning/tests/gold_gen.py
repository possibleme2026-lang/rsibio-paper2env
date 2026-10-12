#!/usr/bin/env python3
"""Protected gold generator / hidden reference for the cold-stress task.

Only present during the verifier phase.  It
  1. regenerates the exact synthetic pond record from the protected seed
     (byte-compatible with environment/data/gen_inputs.py),
  2. runs a hidden reference implementation of the 2022-2023 fit / 2024 threshold
     / 2025 test protocol so the grader can derive calibrated floors.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 20260801
START = "2022-01-01 00:00"
END = "2025-12-31 18:00"
COLD_T = 20.0
LEADS = (6, 12, 18, 24)


def build_frame(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, END, freq="6h")
    n = len(idx)
    doy = idx.dayofyear.to_numpy(dtype=float)
    hod = idx.hour.to_numpy(dtype=float)

    innov = 0.42 + 0.34 * np.cos(2 * np.pi * (doy - 15.0) / 365.25)
    anom = np.zeros(n)
    for i in range(1, n):
        anom[i] = 0.945 * anom[i - 1] + innov[i] * rng.normal()

    air = (26.1 + 6.3 * np.cos(2 * np.pi * (doy - 200.0) / 365.25)
           + 4.1 * np.sin(2 * np.pi * (hod - 9.0) / 24.0)
           + anom + rng.normal(0.0, 0.40, n))

    water_anom = np.zeros(n)
    for i in range(1, n):
        water_anom[i] = 0.86 * water_anom[i - 1] + 0.15 * anom[i] + 0.75 * rng.normal()
    water = (26.1 + 6.3 * np.cos(2 * np.pi * (doy - 218.0) / 365.25)
             + 1.55 * np.sin(2 * np.pi * (hod - 13.0) / 24.0)
             + water_anom + rng.normal(0.0, 0.15, n))

    clear = np.clip(np.sin(np.pi * (hod - 6.0) / 12.0), 0.0, None)
    cloud = np.clip(0.85 - 0.12 * anom + rng.normal(0.0, 0.15, n), 0.15, 1.0)
    solar = 880.0 * clear ** 1.15 * cloud
    humidity = np.clip(79.0 - 1.1 * (air - 26.1) + rng.normal(0.0, 6.5, n), 28.0, 99.0)
    wind = np.clip(2.1 + 0.55 * np.abs(anom) + rng.normal(0.0, 0.9, n), 0.1, 11.0)

    df = pd.DataFrame({"timestamp": idx, "air_temp_c": air, "water_temp_c": water,
                       "solar_rad_wm2": solar, "humidity_pct": humidity, "wind_ms": wind})
    for col, rate in (("air_temp_c", 0.030), ("water_temp_c", 0.030),
                      ("solar_rad_wm2", 0.045), ("humidity_pct", 0.055), ("wind_ms", 0.060)):
        df.loc[rng.random(n) < rate, col] = np.nan
    return df


def make_features(df: pd.DataFrame):
    d = df.set_index("timestamp").sort_index()
    w = d["water_temp_c"]
    a = d["air_temp_c"]
    X = pd.DataFrame({
        "water_t": w,
        "water_l1": w.shift(1), "water_l2": w.shift(2), "water_l3": w.shift(3),
        "water_l4": w.shift(4), "water_l6": w.shift(6),
        "water_mean24": w.rolling(4, min_periods=2).mean(),
        "water_mean72": w.rolling(12, min_periods=4).mean(),
        "air_t": a, "air_l1": a.shift(1), "air_l2": a.shift(2), "air_l3": a.shift(3),
        "solar_t": d["solar_rad_wm2"], "humidity_t": d["humidity_pct"],
        "wind_t": d["wind_ms"],
        "hour_sin": np.sin(2 * np.pi * d.index.hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * d.index.hour / 24.0),
        "doy_sin": np.sin(2 * np.pi * d.index.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * d.index.dayofyear / 365.25),
    }, index=d.index)
    return d, X


def _new_model():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0)


def reference_run(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = build_frame()
    d, X = make_features(df)
    w = d["water_temp_c"]
    out = {}
    for L in LEADS:
        steps = L // 6
        y = w.shift(-steps)
        tr = (np.asarray(X.index.year) <= 2023) & y.notna().to_numpy()
        m = _new_model().fit(X.loc[tr], y.loc[tr])

        dev_idx = X.index[X.index.year == 2024]
        y_dev = y.reindex(dev_idx)
        p_dev = pd.Series(m.predict(X.reindex(dev_idx)), index=dev_idx)
        ok = y_dev.notna()
        best_t, best_f1 = COLD_T, -1.0
        for thr in np.arange(17.5, 21.001, 0.05):
            hit = (p_dev[ok] < thr) & (y_dev[ok] < COLD_T)
            tp = int(hit.sum())
            if tp == 0:
                continue
            fp = int(((p_dev[ok] < thr) & ~(y_dev[ok] < COLD_T)).sum())
            fn = int((~(p_dev[ok] < thr) & (y_dev[ok] < COLD_T)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn)
            if f1 > best_f1:
                best_f1, best_t = float(f1), float(thr)

        te_idx = X.index[X.index.year == 2025]
        y_te = y.reindex(te_idx)
        valid = y_te.notna() & w.reindex(te_idx).notna()
        p = pd.Series(m.predict(X.reindex(te_idx)), index=te_idx)[valid]
        t = y_te[valid]
        al = p < best_t
        cold = t < COLD_T
        tp = int((al & cold).sum())
        fp = int((al & ~cold).sum())
        fn = int((~al & cold).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        err = (p - t).to_numpy(dtype=float)
        out[L] = {"threshold": round(best_t, 3), "dev_f1": round(best_f1, 4),
                  "mae": float(np.mean(np.abs(err))),
                  "rmse": float(np.sqrt(np.mean(err ** 2))),
                  "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
                  "precision": float(prec), "recall": float(rec),
                  "n_cold": int(cold.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = build_frame()
    if args.out:
        df.to_csv(args.out, index=False, float_format="%.2f")
        print(f"wrote {len(df)} rows to {args.out}")
    else:
        print({L: round(v["mae"], 3) for L, v in reference_run(df).items()})


if __name__ == "__main__":
    main()
