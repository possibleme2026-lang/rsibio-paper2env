import json
import numpy as np
import pandas as pd

SEED = 20260728
N_PARTICIPANTS = 12
N_DAYS = 8
SAMPLES_PER_DAY = 8640
SITES = [f"site_{i:02d}" for i in range(1, 9)]
CONTEXTS = ["commute", "home_evening", "indoor_day", "indoor_night", "outdoor_day", "outdoor_night"]
METRICS = ["CoM_hour", "IV", "TAT_1000", "TAT_250", "max_MEDI", "mean_log_MEDI", "median_MEDI"]

ATT_CHEST = {
    "indoor_day": 0.90,
    "outdoor_day": 0.96,
    "commute": 0.92,
    "home_evening": 0.88,
    "indoor_night": 0.78,
    "outdoor_night": 0.76,
}
ATT_WRIST = {
    "indoor_day": 0.65,
    "outdoor_day": 0.45,
    "commute": 0.68,
    "home_evening": 0.62,
    "indoor_night": 0.60,
    "outdoor_night": 0.55,
}


def generate_dataframe():
    rng = np.random.default_rng(SEED)
    rows = []
    for p in range(N_PARTICIPANTS):
        pid = f"P{p+1:02d}"
        site = SITES[p % len(SITES)]
        p_eff_chest = rng.lognormal(mean=-0.02, sigma=0.08)
        p_eff_wrist = rng.lognormal(mean=-0.02, sigma=0.12)
        for d in range(N_DAYS):
            day_id = f"{pid}_D{d+1:02d}"
            d_eff_chest = rng.lognormal(mean=0.0, sigma=0.05)
            d_eff_wrist = rng.lognormal(mean=0.0, sigma=0.08)
            low_cov = ((p + d) % 12 == 0)
            for t in range(SAMPLES_PER_DAY):
                hour = t * 10.0 / 3600.0
                dt = pd.Timestamp("2026-07-01") + pd.Timedelta(days=d) + pd.Timedelta(seconds=10 * t)
                r = rng.random()
                if hour < 6:
                    ctx = "outdoor_night" if r < 0.3 else "indoor_night"
                elif hour < 9:
                    ctx = "commute"
                elif hour < 17:
                    if site in ["site_02", "site_04", "site_06", "site_08"] and r < 0.4:
                        ctx = "outdoor_day"
                    else:
                        ctx = "indoor_day"
                elif hour < 22:
                    ctx = "home_evening"
                else:
                    ctx = "indoor_night"
                if ctx == "outdoor_day":
                    base = rng.lognormal(mean=7.5, sigma=0.4)
                elif ctx == "indoor_day":
                    base = rng.lognormal(mean=4.5, sigma=0.6)
                elif ctx == "commute":
                    base = rng.lognormal(mean=5.5, sigma=0.7)
                elif ctx == "home_evening":
                    base = rng.lognormal(mean=3.5, sigma=0.7)
                elif ctx == "outdoor_night":
                    base = rng.lognormal(mean=0.0, sigma=0.5)
                else:
                    base = rng.lognormal(mean=-0.5, sigma=0.5)
                if "night" in ctx and rng.random() < 0.01:
                    base *= rng.lognormal(mean=2.0, sigma=0.5)
                glasses = max(0.05, base)
                att_c = ATT_CHEST[ctx]
                att_w = ATT_WRIST[ctx]
                noise_c = rng.lognormal(mean=0, sigma=0.12)
                noise_w = rng.lognormal(mean=0, sigma=0.18)
                chest = max(0.01, glasses * att_c * p_eff_chest * d_eff_chest * noise_c)
                wrist = max(0.01, glasses * att_w * p_eff_wrist * d_eff_wrist * noise_w)
                quality = 1 if (low_cov and t % 2 == 0) else 0
                if rng.random() < 0.01:
                    quality = 1
                if quality == 1:
                    glasses = chest = wrist = -1.0
                rows.append((dt, pid, site, day_id, ctx, quality, glasses, chest, wrist))
    df = pd.DataFrame(
        rows,
        columns=[
            "datetime",
            "participant_id",
            "site",
            "participant_day",
            "context",
            "quality_flag",
            "glasses_MEDI",
            "chest_MEDI",
            "wrist_MEDI",
        ],
    )
    return df


def compute_metrics(sub, col):
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
        if var > 0:
            out["IV"] = float(np.sqrt(n * np.sum(diffs ** 2) / ((n - 1) * var)))
        else:
            out["IV"] = 0.0
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
        com = np.arctan2(sin_mean, cos_mean) * 24.0 / (2.0 * np.pi)
        out["CoM_hour"] = float(com % 24.0)
    else:
        out["CoM_hour"] = float("nan")
    return out


def compute_gold(df=None):
    if df is None:
        df = generate_dataframe()
    valid = (
        (df["quality_flag"] == 0)
        & (df["glasses_MEDI"] > 0)
        & (df["chest_MEDI"] > 0)
        & (df["wrist_MEDI"] > 0)
    )
    counts = df[valid].groupby("participant_day").size()
    all_days = df["participant_day"].unique()
    excluded = sorted([d for d in all_days if counts.get(d, 0) < 0.8 * SAMPLES_PER_DAY])
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
            if mask.sum() > 0:
                context_errors[placement][ctx] = {
                    "median_pct_error": float(np.median(err[mask])),
                    "mean_pct_error": float(np.mean(err[mask])),
                }
            else:
                context_errors[placement][ctx] = {"median_pct_error": 0.0, "mean_pct_error": 0.0}
    daily_metric_bias = {}
    for placement in ["chest", "wrist"]:
        daily_metric_bias[placement] = {}
        for metric in METRICS:
            abs_biases = []
            for day in included:
                sub = df_valid[df_valid["participant_day"] == day].sort_values("datetime")
                if len(sub) == 0:
                    continue
                m_glasses = compute_metrics(sub, "glasses_MEDI")
                m_placement = compute_metrics(sub, f"{placement}_MEDI")
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
            if abs_biases:
                daily_metric_bias[placement][metric] = {
                    "median_abs_bias": float(np.median(abs_biases)),
                    "max_abs_bias": float(np.max(abs_biases)),
                    "n_days": int(len(abs_biases)),
                }
            else:
                daily_metric_bias[placement][metric] = {
                    "median_abs_bias": 0.0,
                    "max_abs_bias": 0.0,
                    "n_days": 0,
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
    return {
        "excluded_participant_days": excluded,
        "valid_sample_count": valid_sample_count,
        "context_errors": context_errors,
        "daily_metric_bias": daily_metric_bias,
        "metrics_exceeding_threshold": metrics_exceeding_threshold,
    }


if __name__ == "__main__":
    print(json.dumps(compute_gold(), indent=2))
