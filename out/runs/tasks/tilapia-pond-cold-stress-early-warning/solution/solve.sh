#!/bin/bash
# Reference solution: lagged-feature gradient boosting + 2024 threshold tuning.
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

COLD, LEADS = 20.0, (6, 12, 18, 24)
df = pd.read_csv("/app/data/pond_temperatures.csv", parse_dates=["timestamp"]).set_index("timestamp").sort_index()
w, a = df["water_temp_c"], df["air_temp_c"]

X = pd.DataFrame({
    "water_t": w, "water_l1": w.shift(1), "water_l2": w.shift(2),
    "water_l3": w.shift(3), "water_l4": w.shift(4), "water_l6": w.shift(6),
    "water_mean24": w.rolling(4, min_periods=2).mean(),
    "water_mean72": w.rolling(12, min_periods=4).mean(),
    "air_t": a, "air_l1": a.shift(1), "air_l2": a.shift(2), "air_l3": a.shift(3),
    "solar_t": df["solar_rad_wm2"], "humidity_t": df["humidity_pct"], "wind_t": df["wind_ms"],
    "hour_sin": np.sin(2 * np.pi * df.index.hour / 24.0),
    "hour_cos": np.cos(2 * np.pi * df.index.hour / 24.0),
    "doy_sin": np.sin(2 * np.pi * df.index.dayofyear / 365.25),
    "doy_cos": np.cos(2 * np.pi * df.index.dayofyear / 365.25),
}, index=df.index)

def new_model():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=0)

origins = pd.date_range("2025-01-01 00:00", "2025-12-30 18:00", freq="6h")
origins = origins[w.reindex(origins).notna()]
dev_idx = X.index[X.index.year == 2024]
thresholds, rows = {}, []

for L in LEADS:
    steps = L // 6
    y = w.shift(-steps)
    tr = (np.asarray(X.index.year) <= 2023) & y.notna().to_numpy()
    model = new_model().fit(X.loc[tr], y.loc[tr])

    y_dev = y.reindex(dev_idx)
    p_dev = pd.Series(model.predict(X.reindex(dev_idx)), index=dev_idx)
    ok = y_dev.notna()
    best_t, best_f1 = COLD, -1.0
    for thr in np.arange(17.5, 21.001, 0.05):
        pred, true = p_dev[ok] < thr, y_dev[ok] < COLD
        tp = int((pred & true).sum())
        if tp == 0:
            continue
        fp = int((pred & ~true).sum())
        fn = int((~pred & true).sum())
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(thr)
    thresholds[L] = best_t

    tgt = origins + pd.Timedelta(hours=L)
    keep = w.reindex(tgt).notna().to_numpy()
    o, t = origins[keep], tgt[keep]
    pred = model.predict(X.reindex(o))
    rows.append(pd.DataFrame({"origin_time": o, "target_time": t, "lead_hours": L,
                              "water_temp_pred_c": np.round(pred, 3),
                              "alert_pred": (pred < best_t).astype(int)}))

preds = pd.concat(rows, ignore_index=True).sort_values(["origin_time", "lead_hours"])
preds.to_csv("/app/output/predictions_2025.csv", index=False)

preds["true"] = w.reindex(pd.DatetimeIndex(preds["target_time"])).to_numpy()
leads = {}
for L in LEADS:
    sub = preds[preds["lead_hours"] == L]
    t, p = sub["true"].to_numpy(float), sub["water_temp_pred_c"].to_numpy(float)
    err, al, cold = p - t, sub["alert_pred"].to_numpy() == 1, t < COLD
    tp, fp, fn = int((al & cold).sum()), int((al & ~cold).sum()), int((~al & cold).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    zone = (t >= 18.0) & (t <= 22.0)
    leads[str(L)] = {"threshold_c": round(thresholds[L], 3),
                     "mae_c": round(float(np.mean(np.abs(err))), 4),
                     "rmse_c": round(float(np.sqrt(np.mean(err ** 2))), 4),
                     "bias_c": round(float(np.mean(err)), 4),
                     "boundary_zone_bias_c": round(float(np.mean(err[zone])), 4) if zone.any() else None,
                     "f1": round(float(2 * prec * rec / max(prec + rec, 1e-9)), 4),
                     "precision": round(float(prec), 4), "recall": round(float(rec), 4),
                     "n_alerts": int(al.sum()), "n_cold_readings": int(cold.sum())}

s6 = preds[preds["lead_hours"] == 6]
pers = w.reindex(pd.DatetimeIndex(s6["target_time"]) - pd.Timedelta(hours=24)).to_numpy(float)
pers = np.where(np.isnan(pers), w.reindex(pd.DatetimeIndex(s6["origin_time"])).to_numpy(float), pers)

with open("/app/output/metrics.json", "w") as fh:
    json.dump({"model": "hist-gradient-boosting", "train_years": [2022, 2023], "dev_year": 2024,
               "test_year": 2025, "cold_stress_threshold_c": COLD,
               "persistence_mae_6h_c": round(float(np.mean(np.abs(pers - s6["true"].to_numpy(float)))), 4),
               "leads": leads}, fh, indent=2)
print("wrote", len(preds), "predictions")
PY
