#!/usr/bin/env python3
"""Deep, partial-credit grader for the OM-NPE masked-surveillance task.

Seven independent constraint groups are verified:
  fits_format, parameter_ranges, r0_recovery, trajectory_fit,
  pooled_posterior, amortised_generalisation, amortised_runtime.
The score is the weighted sum of the group scores.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gold_gen  # noqa: E402

DATA_DIR = "/app/data"
OUT_DIR = "/app/output"
LOG_DIR = "/logs/verifier"

WEEKS = 52
REGIONS = ["Tahiti", "Moorea", "Bora-Bora", "Raiatea", "Nuku_Hiva", "Tubuai"]
FIT_COLUMNS = ["region", "beta", "sigma", "gamma", "R0", "fit_rmse"]
PRED_COLUMNS = ["region", "week", "predicted_cases"]

WEIGHTS = {
    "fits_format": 0.10,
    "parameter_ranges": 0.10,
    "r0_recovery": 0.25,
    "trajectory_fit": 0.10,
    "pooled_posterior": 0.10,
    "amortised_generalisation": 0.30,
    "amortised_runtime": 0.05,
}


def load_fits():
    path = os.path.join(OUT_DIR, "fits.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def check_fits_format():
    path = os.path.join(OUT_DIR, "fits.csv")
    if not os.path.exists(path):
        return 0.0, "fits.csv not found"
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return 0.0, "fits.csv unreadable: %s" % exc
    if list(df.columns) != FIT_COLUMNS:
        return 0.0, "columns are %s" % list(df.columns)
    if len(df) != len(REGIONS):
        return 0.0, "expected %d rows, got %d" % (len(REGIONS), len(df))
    if set(df["region"].astype(str)) != set(REGIONS):
        return 0.0, "region labels are %s" % sorted(df["region"].astype(str))
    if df[FIT_COLUMNS[1:]].isnull().to_numpy().any():
        return 0.0, "missing numeric values"
    if not np.isfinite(df[FIT_COLUMNS[1:]].to_numpy(dtype=float)).all():
        return 0.0, "non-finite values"
    return 1.0, "6 regions with exact columns and finite values"


def check_parameter_ranges(fits):
    if fits is None:
        return 0.0, "fits.csv unavailable"
    per = {}
    for _, row in fits.iterrows():
        reg = str(row["region"])
        try:
            beta = float(row["beta"])
            sigma = float(row["sigma"])
            gamma = float(row["gamma"])
            r0 = float(row["R0"])
            rmse = float(row["fit_rmse"])
        except Exception:
            per[reg] = "non-numeric"
            continue
        problems = []
        if not (0.05 <= beta <= 25.0):
            problems.append("beta")
        if not (0.10 <= sigma <= 12.0):
            problems.append("sigma")
        if not (0.10 <= gamma <= 12.0):
            problems.append("gamma")
        if not (0.60 <= r0 <= 6.0):
            problems.append("R0-range")
        if gamma > 0 and abs(r0 - beta / gamma) > 1e-3 * max(1.0, abs(r0)):
            problems.append("R0!=beta/gamma")
        if not (rmse >= 0.0):
            problems.append("fit_rmse")
        per[reg] = "ok" if not problems else ",".join(problems)
    score = sum(1 for v in per.values() if v == "ok") / float(len(REGIONS))
    return score, json.dumps(per)


def check_r0_recovery(fits):
    if fits is None:
        return 0.0, "fits.csv unavailable"
    per = {}
    total = 0.0
    for _, row in fits.iterrows():
        reg = str(row["region"])
        truth = gold_gen.TRUE_R0.get(reg)
        if truth is None:
            per[reg] = "unknown region"
            continue
        try:
            est = float(row["R0"])
        except Exception:
            per[reg] = "non-numeric"
            continue
        rel = abs(est - truth) / truth
        if rel <= 0.15:
            pts = 1.0
        elif rel <= 0.30:
            pts = 0.5
        else:
            pts = 0.0
        total += pts
        per[reg] = {"est": round(est, 3), "true": truth, "rel_err": round(rel, 3), "pts": pts}
    return total / float(len(REGIONS)), json.dumps(per)


def check_trajectory_fit(fits):
    path = os.path.join(OUT_DIR, "predictions.csv")
    if not os.path.exists(path):
        return 0.0, "predictions.csv not found"
    try:
        pred = pd.read_csv(path)
    except Exception as exc:
        return 0.0, "predictions.csv unreadable: %s" % exc
    if list(pred.columns) != PRED_COLUMNS:
        return 0.0, "columns are %s" % list(pred.columns)
    try:
        obs = pd.read_csv(os.path.join(DATA_DIR, "surveillance.csv"))
    except Exception as exc:
        return 0.0, "surveillance.csv unreadable: %s" % exc

    per = {}
    total = 0.0
    for reg in REGIONS:
        sub = pred[pred["region"].astype(str) == reg]
        if len(sub) != WEEKS:
            per[reg] = "expected 52 rows, got %d" % len(sub)
            continue
        try:
            weeks = sorted(sub["week"].astype(int).tolist())
        except Exception:
            per[reg] = "non-integer weeks"
            continue
        if weeks != list(range(WEEKS)):
            per[reg] = "weeks are not 0..51"
            continue
        y = obs[obs["region"].astype(str) == reg]["cases"].to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            per[reg] = "too few observed weeks"
            continue
        p = sub.sort_values("week")["predicted_cases"].to_numpy(dtype=float)
        if not np.isfinite(p).all():
            per[reg] = "non-finite predictions"
            continue
        resid = p[mask] - y[mask]
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mean_obs = float(np.mean(y[mask]))
        nrmse = rmse / (mean_obs + 1e-9)
        cv = float(np.std(y[mask]) / (mean_obs + 1e-9))
        ok = nrmse < 0.8 * cv
        entry = {"nrmse": round(nrmse, 4), "cv": round(cv, 4), "ok": bool(ok)}
        if fits is not None:
            try:
                reported = float(fits.loc[fits["region"].astype(str) == reg, "fit_rmse"].iloc[0])
                entry["reported_rmse"] = reported
                entry["rmse_consistent"] = abs(reported - rmse) <= 0.25 * max(rmse, 1e-9)
            except Exception:
                entry["rmse_consistent"] = False
        total += 1.0 if ok else 0.0
        per[reg] = entry
    return total / float(len(REGIONS)), json.dumps(per)


def check_pooled(fits):
    path = os.path.join(OUT_DIR, "pooled.json")
    if not os.path.exists(path):
        return 0.0, "pooled.json not found"
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except Exception as exc:
        return 0.0, "pooled.json unreadable: %s" % exc
    if not isinstance(obj, dict):
        return 0.0, "not a JSON object"
    if "pooled_R0" not in obj or "pooled_R0_ci" not in obj:
        return 0.0, "missing pooled_R0 / pooled_R0_ci"
    try:
        p = float(obj["pooled_R0"])
        ci = obj["pooled_R0_ci"]
        lo = float(ci[0])
        hi = float(ci[1])
    except Exception as exc:
        return 0.0, "non-numeric values: %s" % exc
    if not (np.isfinite(p) and np.isfinite(lo) and np.isfinite(hi)):
        return 0.0, "non-finite values"

    pts = 0.35
    if 0.6 <= p <= 6.0:
        pts += 0.25
    if lo <= p <= hi and hi > lo:
        pts += 0.20
    if fits is not None:
        try:
            r0s = fits["R0"].to_numpy(dtype=float)
            if float(np.min(r0s)) - 0.20 <= p <= float(np.max(r0s)) + 0.20:
                pts += 0.20
        except Exception:
            pass
    detail = {"pooled_R0": p, "ci": [lo, hi]}
    return pts, json.dumps(detail)


def _parse_r0(stdout):
    for line in reversed(stdout.strip().splitlines()):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and "R0" in obj:
            try:
                return float(obj["R0"])
            except Exception:
                return None
    return None


def check_amortised():
    script = os.path.join(OUT_DIR, "amortized_predict.py")
    if not os.path.exists(script):
        return 0.0, 0.0, "amortized_predict.py not found"
    try:
        feats, truths = gold_gen.generate_holdout()
    except Exception as exc:
        return 0.0, 0.0, "holdout generation failed: %s" % exc

    total_pts = 0.0
    times = []
    detail = []
    for i, (feat, truth) in enumerate(zip(feats, truths)):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(feat, fh)
            fpath = fh.name
        t0 = time.time()
        proc = None
        try:
            proc = subprocess.run(
                [sys.executable, script, "--features", fpath],
                capture_output=True, text=True, timeout=120,
            )
        except Exception:
            proc = None
        elapsed = time.time() - t0
        times.append(elapsed)
        try:
            os.unlink(fpath)
        except Exception:
            pass
        if proc is None or proc.returncode != 0:
            detail.append({"i": i, "status": "exec-error"})
            continue
        est = _parse_r0(proc.stdout)
        if est is None or not np.isfinite(est):
            detail.append({"i": i, "status": "no R0 on stdout"})
            continue
        rel = abs(est - truth) / truth
        if rel <= 0.15:
            pts = 1.0
        elif rel <= 0.25:
            pts = 0.7
        elif rel <= 0.40:
            pts = 0.35
        else:
            pts = 0.0
        total_pts += pts
        detail.append({"i": i, "est": round(est, 3), "true": round(truth, 3),
                       "rel_err": round(rel, 3), "pts": pts})

    n = len(truths)
    gen_score = total_pts / float(n)
    max_t = max(times) if times else 999.0
    runtime_score = 1.0 if max_t < 8.0 else 0.0
    detail_out = {"max_seconds": round(max_t, 3), "per_region": detail}
    return gen_score, runtime_score, json.dumps(detail_out)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    fits = load_fits()

    results = []
    results.append(("fits_format",) + check_fits_format())
    results.append(("parameter_ranges",) + check_parameter_ranges(fits))
    results.append(("r0_recovery",) + check_r0_recovery(fits))
    results.append(("trajectory_fit",) + check_trajectory_fit(fits))
    results.append(("pooled_posterior",) + check_pooled(fits))
    gen_score, runtime_score, am_detail = check_amortised()
    results.append(("amortised_generalisation", gen_score, am_detail))
    results.append(("amortised_runtime", runtime_score, am_detail))

    total = 0.0
    checks = {}
    for name, value, detail in results:
        value = float(min(1.0, max(0.0, value)))
        checks[name] = {"score": round(value, 4), "weight": WEIGHTS[name],
                        "detail": detail}
        total += WEIGHTS[name] * value
    total = float(min(1.0, max(0.0, total)))

    report = {"score": total, "checks": checks, "weights": WEIGHTS}
    with open(os.path.join(LOG_DIR, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(LOG_DIR, "reward.json"), "w") as fh:
        json.dump({"score": total, "reward": total}, fh)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
