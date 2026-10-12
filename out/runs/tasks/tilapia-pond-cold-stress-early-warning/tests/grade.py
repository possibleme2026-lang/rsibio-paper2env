#!/usr/bin/env python3
"""Deep verifier for the cold-stress early-warning task (bioforge/harbor).

The grader re-derives everything it can from the protected data record:
  * it regenerates the synthetic pond record from the protected seed and checks
    that the in-container copy was not tampered with,
  * it recomputes every metric the agent reported, from the agent's own rows,
  * it fits a hidden reference implementation (2022-2023 fit / 2024 threshold /
    2025 test) so MAE and F1 floors are calibrated to this dataset,
  * it recomputes the same-hour-yesterday persistence baseline on the test rows.

Partial credit: score = sum(weight of passing checks) / sum(all weights).
reward.json only ever contains {"score": float, "reward": float}.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

DATA_CSV = "/app/data/pond_temperatures.csv"
PRED_CSV = "/app/output/predictions_2025.csv"
METRICS_JSON = "/app/output/metrics.json"
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_GEN = os.path.join(HERE, "gold_gen.py")
OUTDIR = "/logs/verifier"

LEADS = (6, 12, 18, 24)
COLD_T = 20.0
ZONE_LO, ZONE_HI = 18.0, 22.0
ORIGIN_FIRST = pd.Timestamp("2025-01-01 00:00:00")
ORIGIN_LAST = pd.Timestamp("2025-12-30 18:00:00")

CHECKS: list[dict] = []


def add(name, weight, ok, detail=""):
    ok = bool(ok)
    CHECKS.append({"name": name, "weight": float(weight), "passed": ok, "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} (weight {weight}) :: {detail}", flush=True)


def load_gold():
    spec = importlib.util.spec_from_file_location("gold_gen", GOLD_GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fnum(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def check_integrity(df_gen, df_csv):
    """The scored record must be identical to the protected generator output."""
    try:
        a = df_gen.set_index("timestamp").sort_index()
        b = df_csv.set_index("timestamp").sort_index()
    except Exception as exc:
        return False, f"record not comparable: {exc}"
    if len(a) != len(b) or not a.index.equals(b.index):
        return False, f"timestamp grid differs ({len(b)} rows vs {len(a)})"
    worst, bad = 0.0, 0
    for col in ("air_temp_c", "water_temp_c", "solar_rad_wm2", "humidity_pct", "wind_ms"):
        if col not in b.columns:
            return False, f"column {col} missing from the input record"
        x = a[col].to_numpy(dtype=float)
        y = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
        both = ~(np.isnan(x) & np.isnan(y))
        d = np.abs(x[both] - y[both])
        d = d[~np.isnan(d)]
        if d.size:
            worst = max(worst, float(d.max()))
        bad += int((d > 0.05).sum())
    if bad > 5 or worst > 0.25:
        return False, f"record altered (max |delta|={worst:.3f}, n_bad={bad})"
    return True, f"record matches protected generator (max |delta|={worst:.3f})"


def main():
    gold = load_gold()
    df_gen = gold.build_frame()

    # ---------- the record the agent was scored against --------------------
    df_csv = None
    try:
        df_csv = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    except Exception as exc:
        print("input record unreadable:", exc, file=sys.stderr)
    if df_csv is None:
        integrity, integ_detail = False, "input record unreadable"
    else:
        integrity, integ_detail = check_integrity(df_gen, df_csv)
    add("data_record_intact", 1.0, integrity, integ_detail)
    source = df_csv if integrity else df_gen
    truth = source.set_index("timestamp").sort_index()
    water = truth["water_temp_c"]

    # ---------- hidden reference bars --------------------------------------
    ref = {}
    try:
        ref = gold.reference_run(df_gen)
    except Exception:
        traceback.print_exc()
        print("reference run failed; using paper-scale fallback bars", file=sys.stderr)
    mae_floor = {L: (ref.get(L, {}).get("mae", 1.25) * 1.15 + 0.03) for L in LEADS}
    f1_floor = {L: (ref.get(L, {}).get("f1", 0.60) - 0.10) for L in LEADS}
    rec_floor6 = ref.get(6, {}).get("recall", 0.70) - 0.10
    print("reference bars:", {L: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ref.get(L, {}).items()} for L in LEADS}, flush=True)

    # ---------- agent artefacts --------------------------------------------
    preds = None
    try:
        preds = pd.read_csv(PRED_CSV)
    except Exception as exc:
        print("predictions unreadable:", exc, file=sys.stderr)
    metrics = None
    try:
        with open(METRICS_JSON) as fh:
            metrics = json.load(fh)
    except Exception as exc:
        print("metrics unreadable:", exc, file=sys.stderr)

    need = ["origin_time", "target_time", "lead_hours", "water_temp_pred_c", "alert_pred"]
    schema = preds is not None and all(c in preds.columns for c in need)
    lead_block = (isinstance(metrics, dict) and isinstance(metrics.get("leads"), dict)
                  and all(str(L) in metrics["leads"] for L in LEADS))
    add("outputs_present_and_parsable", 1.0, schema and lead_block,
        f"predictions={'ok' if schema else 'missing/bad columns'}; metrics={'ok' if lead_block else 'missing/bad'}")

    usable = pd.DataFrame(columns=need + ["true_water", "origin_water"])
    if schema:
        p = preds.copy()
        p["origin_time"] = pd.to_datetime(p["origin_time"], errors="coerce")
        p["target_time"] = pd.to_datetime(p["target_time"], errors="coerce")
        p["lead_hours"] = pd.to_numeric(p["lead_hours"], errors="coerce")
        p["water_temp_pred_c"] = pd.to_numeric(p["water_temp_pred_c"], errors="coerce")
        p["alert_pred"] = pd.to_numeric(p["alert_pred"], errors="coerce")
        p = p.dropna(subset=["origin_time", "target_time", "lead_hours", "water_temp_pred_c"])
        p = p[p["lead_hours"].isin(list(LEADS))]
        p["lead_hours"] = p["lead_hours"].astype(int)
        delta = (p["target_time"] - p["origin_time"]).dt.total_seconds() / 3600.0
        p = p[np.isclose(delta, p["lead_hours"].astype(float))]
        p = p[(p["origin_time"] >= ORIGIN_FIRST) & (p["origin_time"] <= ORIGIN_LAST)]
        p = p.assign(true_water=water.reindex(p["target_time"]).to_numpy(),
                     origin_water=water.reindex(p["origin_time"]).to_numpy())
        p = p.dropna(subset=["true_water", "origin_water"])
        v = pd.to_numeric(p["water_temp_pred_c"], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(v) & (v > -5.0) & (v < 45.0)
        usable = p[keep].copy()
        usable["alert_pred"] = pd.to_numeric(usable["alert_pred"], errors="coerce").fillna(0.0)

    # ---------- coverage of the eligible 2025 grid -------------------------
    origins = pd.date_range("2025-01-01 00:00", "2025-12-30 18:00", freq="6h")
    ow = water.reindex(origins).to_numpy(dtype=float)
    ref_pairs = set()
    for L in LEADS:
        tw = water.reindex(origins + pd.Timedelta(hours=L)).to_numpy(dtype=float)
        keep = ~np.isnan(tw) & ~np.isnan(ow)
        for i in np.nonzero(keep)[0]:
            ref_pairs.add((origins[i], L))
    agent_pairs = set(zip(usable["origin_time"], usable["lead_hours"])) if len(usable) else set()
    coverage = (len(ref_pairs & agent_pairs) / max(len(ref_pairs), 1)) if ref_pairs else 0.0
    add("coverage_and_row_validity", 1.5, len(usable) > 0 and coverage >= 0.95,
        f"{len(usable)} usable rows, coverage={coverage:.3f} of {len(ref_pairs)} expected pairs")

    # ---------- recomputed metrics ----------------------------------------
    per = {}
    for L in LEADS:
        sub = usable[usable["lead_hours"] == L] if len(usable) else usable
        if len(sub) < 50:
            per[L] = None
            continue
        t = sub["true_water"].to_numpy(dtype=float)
        pr = sub["water_temp_pred_c"].to_numpy(dtype=float)
        err = pr - t
        alerts = sub["alert_pred"].to_numpy(dtype=float) > 0.5
        cold = t < COLD_T
        tp = float(np.sum(alerts & cold))
        fp = float(np.sum(alerts & ~cold))
        fn = float(np.sum(~alerts & cold))
        prec = tp / max(tp + fp, 1.0)
        rec = tp / max(tp + fn, 1.0)
        zone = (t >= ZONE_LO) & (t <= ZONE_HI)
        per[L] = {"n": int(len(sub)), "mae": float(np.mean(np.abs(err))),
                  "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(np.mean(err)),
                  "boundary_bias": float(np.mean(err[zone])) if zone.any() else float("nan"),
                  "precision": prec, "recall": rec,
                  "f1": 2 * prec * rec / max(prec + rec, 1e-9),
                  "n_alerts": int(np.sum(alerts)), "n_cold": int(np.sum(cold))}
        print(f"  lead {L:>2}h: n={per[L]['n']} mae={per[L]['mae']:.3f} rmse={per[L]['rmse']:.3f} "
              f"f1={per[L]['f1']:.3f} rec={rec:.3f} prec={prec:.3f} "
              f"alerts={per[L]['n_alerts']} cold={per[L]['n_cold']}", flush=True)

    # ---------- alert rule consistency -------------------------------------
    thr_ok, thr_det = False, "no predictions/metrics"
    if lead_block and len(usable):
        ok = True
        parts = []
        for L in LEADS:
            entry = metrics["leads"].get(str(L))
            thr = fnum(entry.get("threshold_c")) if isinstance(entry, dict) else None
            sub = usable[usable["lead_hours"] == L]
            if thr is None or not (17.0 <= thr <= 22.5) or len(sub) == 0:
                ok = False
                parts.append(f"{L}h:missing/bad threshold")
                continue
            expected = (sub["water_temp_pred_c"].to_numpy(dtype=float) < thr).astype(int)
            given = (sub["alert_pred"].to_numpy(dtype=float) > 0.5).astype(int)
            agree = float(np.mean(expected == given))
            parts.append(f"{L}h thr={thr:.2f} agree={agree:.3f}")
            if agree < 0.99:
                ok = False
        thr_ok, thr_det = ok, "; ".join(parts)
    add("alert_rule_matches_reported_threshold", 1.0, thr_ok, thr_det)

    # ---------- honesty of reported metrics --------------------------------
    reg_ok, cls_ok = False, False
    reg_det, cls_det = "no comparison", "no comparison"
    if lead_block and len(usable):
        reg_ok, cls_ok = True, True
        rp, cp = [], []
        for L in LEADS:
            entry = metrics["leads"].get(str(L), {})
            m = per.get(L)
            if not isinstance(entry, dict) or m is None:
                reg_ok = cls_ok = False
                rp.append(f"{L}h:unavailable")
                continue
            for key, tol, got in (("mae_c", 0.05, m["mae"]), ("rmse_c", 0.05, m["rmse"]),
                                  ("bias_c", 0.05, m["bias"])):
                rep = fnum(entry.get(key))
                if rep is None or abs(rep - got) > tol:
                    reg_ok = False
                    rp.append(f"{L}h {key} reported={rep} recomputed={got:.3f}")
            for key, tol, got in (("f1", 0.05, m["f1"]), ("recall", 0.05, m["recall"]),
                                  ("precision", 0.05, m["precision"])):
                rep = fnum(entry.get(key))
                if rep is None or abs(rep - got) > tol:
                    cls_ok = False
                    cp.append(f"{L}h {key} reported={rep} recomputed={got:.3f}")
        reg_det = "; ".join(rp) if rp else "reported regression metrics reproduce"
        cls_det = "; ".join(cp) if cp else "reported classification metrics reproduce"
    add("reported_regression_metrics_are_honest", 1.0, reg_ok, reg_det)
    add("reported_classification_metrics_are_honest", 1.0, cls_ok, cls_det)

    # ---------- persistence baseline ---------------------------------------
    if len(usable):
        sub = usable[usable["lead_hours"] == 6]
        tgt = sub["target_time"]
        pers = water.reindex(tgt - pd.Timedelta(hours=24)).to_numpy(dtype=float)
        pers = np.where(np.isnan(pers), sub["origin_water"].to_numpy(dtype=float), pers)
        base = float(np.mean(np.abs(pers - sub["true_water"].to_numpy(dtype=float))))
        m6 = per.get(6)
        mae6 = m6["mae"] if m6 else float("nan")
        add("beats_persistence_baseline_6h", 1.5, m6 is not None and mae6 <= 0.85 * base,
            f"model MAE {mae6:.3f} vs same-hour-yesterday MAE {base:.3f} on the same rows")
    else:
        add("beats_persistence_baseline_6h", 1.5, False, "no usable predictions")

    # ---------- performance floors -----------------------------------------
    mae_pass = all(per.get(L) is not None and per[L]["mae"] <= mae_floor[L] for L in LEADS)
    add("mae_within_reference_tolerance_all_leads", 1.5, mae_pass,
        "; ".join(f"{L}h {per[L]['mae']:.3f}<={mae_floor[L]:.3f}" if per.get(L) else f"{L}h n/a" for L in LEADS))

    f1_6 = per[6]["f1"] if per.get(6) else float("nan")
    add("f1_floor_6h", 2.0, bool(per.get(6)) and f1_6 >= f1_floor[6],
        f"F1(6h)={f1_6:.3f} floor={f1_floor[6]:.3f}")

    rec_6 = per[6]["recall"] if per.get(6) else float("nan")
    add("cold_sensitivity_floor_6h", 1.5, bool(per.get(6)) and rec_6 >= rec_floor6,
        f"sensitivity(6h)={rec_6:.3f} floor={rec_floor6:.3f}")

    multi = all(per.get(L) is not None and per[L]["f1"] >= f1_floor[L] for L in (12, 18, 24))
    add("f1_floors_12_18_24h", 1.5, multi,
        "; ".join(f"{L}h {per[L]['f1']:.3f}>={f1_floor[L]:.3f}" if per.get(L) else f"{L}h n/a" for L in (12, 18, 24)))

    bb = per[6]["boundary_bias"] if per.get(6) else float("nan")
    add("boundary_zone_bias_bounded_6h", 1.0, bool(per.get(6)) and np.isfinite(bb) and abs(bb) <= 0.8,
        f"mean(pred-true) for observed 18-22C = {bb:.3f} (|.|<=0.8)")

    # ---------- plausibility / anti-leakage --------------------------------
    plausible = all(per.get(L) is not None and per[L]["mae"] >= 0.25 and per[L]["f1"] <= 0.995 for L in LEADS)
    add("no_future_data_leakage", 1.0, plausible, "MAE>=0.25 and F1<=0.995 at every lead")

    if per.get(6) is not None:
        rate = per[6]["n_alerts"] / max(per[6]["n_cold"], 1)
        add("alert_rate_plausible_6h", 1.0, 0.4 <= rate <= 3.0,
            f"{per[6]['n_alerts']} alerts vs {per[6]['n_cold']} cold readings (ratio {rate:.2f})")
    else:
        add("alert_rate_plausible_6h", 1.0, False, "no usable 6h predictions")

    # ---------- reward ------------------------------------------------------
    total = sum(c["weight"] for c in CHECKS)
    got = sum(c["weight"] for c in CHECKS if c["passed"])
    score = round(got / total, 4) if total else 0.0
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "report.json"), "w") as fh:
        json.dump({"score": score, "passed_weight": got, "total_weight": total,
                   "checks": CHECKS,
                   "reference_bars": {str(L): ref.get(L, {}) for L in LEADS}}, fh, indent=2, default=str)
    with open(os.path.join(OUTDIR, "reward.json"), "w") as fh:
        json.dump({"score": score, "reward": score}, fh)
    print(json.dumps({"score": score, "reward": score}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        os.makedirs(OUTDIR, exist_ok=True)
        with open(os.path.join(OUTDIR, "reward.json"), "w") as fh:
            json.dump({"score": 0.0, "reward": 0.0}, fh)
        print(json.dumps({"score": 0.0, "reward": 0.0}))
        sys.exit(0)
