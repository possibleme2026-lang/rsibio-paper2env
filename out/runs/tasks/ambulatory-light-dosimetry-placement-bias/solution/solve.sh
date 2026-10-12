#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import json
import numpy as np
import pandas as pd

CONTEXTS = ["commute", "home_evening", "indoor_day", "indoor_night", "outdoor_day", "outdoor_night"]
METRICS = ["CoM_hour", "IV", "TAT_1000", "TAT_250", "max_MEDI", "mean_log_MEDI", "median_MEDI"]

df = pd.read_csv("/app/data/dosimetry.csv", parse_dates=["datetime"])
valid = (
    (df["quality_flag"] == 0)
    & (df["glasses_MEDI"] > 0)
    & (df["chest_MEDI"] > 0)
    & (df["wrist_MEDI"] > 0)
)
counts = df[valid].groupby("participant_day").size()
all_days = df["participant_day"].unique()
excluded = sorted([d for d in all_days if counts.get(d, 0) < 6912])
included = [d for d in all_days if d not in excluded]
df_valid = df[valid & df["participant_day"].isin(included)].copy()
valid_sample_count = int(len(df_valid))

context_errors = {}
for placement in ["chest", "wrist"]:
    col = f"{placement}_MEDI"
    err = 100.0 * (df_valid[col] - df_valid["glasses_MEDI"]) / df_valid["glasses_MEDI"]
    context_errors[placement] = {}
    for ctx in CONTEXTS:
        mask = df_valid["context"] == ctx
        context_errors[placement][ctx] = {
            "median_pct_error": round(float(np.median(err[mask])), 4),
            "mean_pct_error": round(float(np.mean(err[mask])), 4),
        }


def metrics(sub, col):
    x = sub[col].values.astype(float)
    logx = np.log10(x + 1.0)
    n = len(x)
    out = {}
    out["mean_log_MEDI"] = float(np.mean(logx))
    out["median_MEDI"] = float(np.median(x))
    out["TAT_250"] = float(np.mean(x >= 250.0))
    out["TAT_1000"] = float(np.mean(x >= 1000.0))
    out["max_MEDI"] = float(np.max(x))
    if n > 1:
        diffs = np.diff(logx)
        var = np.var(logx, ddof=0)
        out["IV"] = float(np.sqrt(n * np.sum(diffs ** 2) / ((n - 1) * var))) if var > 0 else 0.0
    else:
        out["IV"] = 0.0
    hours = (
        sub["datetime"].dt.hour
        + sub["datetime"].dt.minute / 60.0
        + sub["datetime"].dt.second / 3600.0
    ).values
    theta = 2.0 * np.pi * hours / 24.0
    total = np.sum(x)
    if total > 0:
        cos_mean = np.sum(x * np.cos(theta)) / total
        sin_mean = np.sum(x * np.sin(theta)) / total
        out["CoM_hour"] = float((np.arctan2(sin_mean, cos_mean) * 24.0 / (2.0 * np.pi)) % 24.0)
    else:
        out["CoM_hour"] = float("nan")
    return out


daily_metric_bias = {}
for placement in ["chest", "wrist"]:
    daily_metric_bias[placement] = {}
    for metric in METRICS:
        abs_biases = []
        for day in included:
            sub = df_valid[df_valid["participant_day"] == day].sort_values("datetime")
            if len(sub) == 0:
                continue
            m_glasses = metrics(sub, "glasses_MEDI")
            m_placement = metrics(sub, f"{placement}_MEDI")
            if metric == "CoM_hour":
                diff = abs(m_placement[metric] - m_glasses[metric])
                if not np.isnan(diff):
                    abs_biases.append(min(diff, 24.0 - diff))
            else:
                ref = m_glasses[metric]
                val = m_placement[metric]
                if ref == 0:
                    if val == 0:
                        abs_biases.append(0.0)
                    else:
                        continue
                else:
                    abs_biases.append(abs(100.0 * (val - ref) / ref))
        daily_metric_bias[placement][metric] = {
            "median_abs_bias": round(float(np.median(abs_biases)), 4) if abs_biases else 0.0,
            "max_abs_bias": round(float(np.max(abs_biases)), 4) if abs_biases else 0.0,
            "n_days": len(abs_biases),
        }

metrics_exceeding_threshold = {}
for placement in ["chest", "wrist"]:
    exceeding = []
    for metric in METRICS:
        m = daily_metric_bias[placement][metric]["median_abs_bias"]
        thresh = 0.5 if metric == "CoM_hour" else 5.0
        if m > thresh:
            exceeding.append(metric)
    metrics_exceeding_threshold[placement] = sorted(exceeding)

result = {
    "excluded_participant_days": excluded,
    "valid_sample_count": valid_sample_count,
    "context_errors": context_errors,
    "daily_metric_bias": daily_metric_bias,
    "metrics_exceeding_threshold": metrics_exceeding_threshold,
}
with open("/app/output/placement_analysis.json", "w") as f:
    json.dump(result, f, indent=2)
PY
