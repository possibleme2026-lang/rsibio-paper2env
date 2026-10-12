#!/usr/bin/env python3
"""Grader for the MAXWELL ProteinMPNN-calibration task.

Seven independently checkable constraints are evaluated with partial credit.
Only plain numbers are written to /logs/verifier/reward.json; the per-constraint
breakdown goes to stdout and to /logs/verifier/report.json.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gold_gen  # noqa: E402

AA = gold_gen.AA
AA_IDX = {a: i for i, a in enumerate(AA)}
PROT_LEN = dict(gold_gen.PROTEINS)

OUT_DIR = "/app/output"
PRED_PATH = os.path.join(OUT_DIR, "predictions.csv")
LAND_DIR = os.path.join(OUT_DIR, "landscape")
REWARD_PATH = "/logs/verifier/reward.json"
REPORT_PATH = "/logs/verifier/report.json"

DDG_MIN, DDG_MAX = -9.0, 9.0
SPEARMAN_FULL = 0.65
SPEARMAN_ZERO = 0.20
RMSE_FULL = 1.60
RMSE_ZERO = 2.40
DIAG_TOL = 0.5
CONSISTENCY_TOL = 1e-3

WEIGHTS = {
    "c1_predictions_complete": 0.10,
    "c2_predictions_range": 0.10,
    "c3_spearman": 0.25,
    "c4_rmse": 0.15,
    "c5_landscape_files": 0.15,
    "c6_landscape_diagonal": 0.10,
    "c7_landscape_consistency": 0.15,
}


def parse_mutation(m):
    m = str(m).strip()
    if len(m) < 3:
        raise ValueError("bad mutation string: %r" % m)
    wt, mut = m[0], m[-1]
    if wt not in AA_IDX or mut not in AA_IDX:
        raise ValueError("non-canonical residue in %r" % m)
    return wt, int(m[1:-1]), mut


def main():
    report = {"constraints": {}, "weights": WEIGHTS}

    gold = gold_gen.gold_test_labels()
    gold_keys = [(p, m) for p, m, _ in gold]
    gold_vals = np.array([d for _, _, d in gold], dtype=float)
    gold_set = set(gold_keys)
    wt_map = {pid: list(s) for pid, s in gold_gen.wildtype_sequences().items()}

    # ---------------- load agent predictions ----------------
    pred_df, load_error = None, None
    if os.path.exists(PRED_PATH):
        try:
            pred_df = pd.read_csv(PRED_PATH)
        except Exception as exc:  # noqa: BLE001
            load_error = "unreadable predictions.csv: %s" % exc
    else:
        load_error = "missing %s" % PRED_PATH

    pred_map = {}
    c1 = c2 = False
    if pred_df is not None and {"protein_id", "mutation", "predicted_ddg"}.issubset(
            {str(c) for c in pred_df.columns}):
        try:
            keys = [(str(p).strip(), str(m).strip())
                    for p, m in zip(pred_df["protein_id"], pred_df["mutation"])]
            vals = pd.to_numeric(pred_df["predicted_ddg"], errors="coerce").to_numpy(dtype=float)
            if len(keys) == len(gold_keys) and len(set(keys)) == len(keys) and set(keys) == gold_set:
                c1 = True
            if np.all(np.isfinite(vals)) and vals.min() >= DDG_MIN and vals.max() <= DDG_MAX:
                c2 = True
            for k, v in zip(keys, vals):
                pred_map.setdefault(k, v)
        except Exception as exc:  # noqa: BLE001
            load_error = load_error or str(exc)

    report["constraints"]["c1_predictions_complete"] = {
        "passed": bool(c1),
        "detail": ("%d unique rows, expected set matched" % len(pred_map)) if c1
                  else (load_error or "rows do not match the %d expected test mutations" % len(gold_keys)),
    }
    report["constraints"]["c2_predictions_range"] = {
        "passed": bool(c2),
        "detail": "all values finite and within [%s, %s]" % (DDG_MIN, DDG_MAX),
    }

    # ---------------- metric constraints ----------------
    rho = rmse = None
    if c1:
        y_pred = np.array([pred_map[k] for k in gold_keys], dtype=float)
        if np.all(np.isfinite(y_pred)):
            rho = float(spearmanr(gold_vals, y_pred).correlation)
            rmse = float(np.sqrt(np.mean((gold_vals - y_pred) ** 2)))

    c3_score = 0.0
    if rho is not None and not math.isnan(rho):
        if rho >= SPEARMAN_FULL:
            c3_score = WEIGHTS["c3_spearman"]
        elif rho > SPEARMAN_ZERO:
            c3_score = WEIGHTS["c3_spearman"] * (rho - SPEARMAN_ZERO) / (SPEARMAN_FULL - SPEARMAN_ZERO)
    report["constraints"]["c3_spearman"] = {
        "passed": bool(rho is not None and not math.isnan(rho) and rho >= SPEARMAN_FULL),
        "spearman": rho, "threshold": SPEARMAN_FULL, "weighted_score": round(c3_score, 4),
    }

    c4_score = 0.0
    if rmse is not None and not math.isnan(rmse):
        if rmse <= RMSE_FULL:
            c4_score = WEIGHTS["c4_rmse"]
        elif rmse < RMSE_ZERO:
            c4_score = WEIGHTS["c4_rmse"] * (RMSE_ZERO - rmse) / (RMSE_ZERO - RMSE_FULL)
    report["constraints"]["c4_rmse"] = {
        "passed": bool(rmse is not None and not math.isnan(rmse) and rmse <= RMSE_FULL),
        "rmse": rmse, "threshold": RMSE_FULL, "weighted_score": round(c4_score, 4),
    }

    # ---------------- landscape constraints ----------------
    land_mat, header_ok, range_ok, problems = {}, {}, {}, []
    for pid in gold_gen.LANDSCAPE_PROTEINS:
        L = PROT_LEN[pid]
        path = os.path.join(LAND_DIR, "%s.csv" % pid)
        land_mat[pid], header_ok[pid], range_ok[pid] = None, False, False
        if not os.path.exists(path):
            problems.append("%s: missing %s" % (pid, path))
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            problems.append("%s: unreadable (%s)" % (pid, exc))
            continue
        cols = [str(c) for c in df.columns]
        if cols[0] != "position" or cols[1:] != AA or len(df) != L:
            problems.append("%s: expected header position+20 AAs and %d rows, got %d rows" % (pid, L, len(df)))
            continue
        header_ok[pid] = True
        mat = df[AA].to_numpy(dtype=float)
        if np.all(np.isfinite(mat)) and mat.min() >= DDG_MIN and mat.max() <= DDG_MAX:
            range_ok[pid] = True
        else:
            problems.append("%s: values out of [%s, %s] or non-finite" % (pid, DDG_MIN, DDG_MAX))
        land_mat[pid] = mat

    c5 = all(header_ok[p] and range_ok[p] for p in gold_gen.LANDSCAPE_PROTEINS)
    report["constraints"]["c5_landscape_files"] = {
        "passed": bool(c5),
        "detail": problems if problems else "all landscapes present with correct shape and range",
    }

    diag_bad, diag_checked = [], 0
    for pid in gold_gen.LANDSCAPE_PROTEINS:
        mat = land_mat.get(pid)
        if mat is None:
            diag_bad.append("%s: landscape unavailable" % pid)
            continue
        for i, w in enumerate(wt_map[pid]):
            diag_checked += 1
            v = float(mat[i, AA_IDX[w]])
            if not np.isfinite(v) or abs(v) > DIAG_TOL:
                diag_bad.append("%s:%s%d -> %.3f" % (pid, w, i + 1, v))
    c6 = len(diag_bad) == 0 and diag_checked > 0
    report["constraints"]["c6_landscape_diagonal"] = {
        "passed": bool(c6),
        "checked": diag_checked,
        "violations": diag_bad[:20],
    }

    incons_bad, n_cmp = [], 0
    land_keys = [(p, m) for (p, m) in gold_keys if p in gold_gen.LANDSCAPE_PROTEINS]
    for (pid, mut) in land_keys:
        mat = land_mat.get(pid)
        if mat is None:
            incons_bad.append("%s: landscape unavailable" % pid)
            continue
        if (pid, mut) not in pred_map:
            incons_bad.append("%s:%s missing from predictions.csv" % (pid, mut))
            continue
        w, pos, m = parse_mutation(mut)
        lv = float(mat[pos - 1, AA_IDX[m]])
        pv = float(pred_map[(pid, mut)])
        n_cmp += 1
        if abs(lv - pv) > CONSISTENCY_TOL:
            incons_bad.append("%s:%s landscape=%.4f prediction=%.4f" % (pid, mut, lv, pv))
    c7 = (len(incons_bad) == 0 and n_cmp == len(land_keys) and len(land_keys) > 0)
    report["constraints"]["c7_landscape_consistency"] = {
        "passed": bool(c7),
        "compared": n_cmp,
        "violations": incons_bad[:20],
    }

    # ---------------- scoring ----------------
    contributions = {
        "c1_predictions_complete": WEIGHTS["c1_predictions_complete"] if c1 else 0.0,
        "c2_predictions_range": WEIGHTS["c2_predictions_range"] if c2 else 0.0,
        "c3_spearman": c3_score,
        "c4_rmse": c4_score,
        "c5_landscape_files": WEIGHTS["c5_landscape_files"] if c5 else 0.0,
        "c6_landscape_diagonal": WEIGHTS["c6_landscape_diagonal"] if c6 else 0.0,
        "c7_landscape_consistency": WEIGHTS["c7_landscape_consistency"] if c7 else 0.0,
    }
    score = max(0.0, min(1.0, float(sum(contributions.values()))))
    report["contributions"] = {k: round(v, 6) for k, v in contributions.items()}
    report["score"] = round(score, 6)
    report["n_test"] = len(gold_keys)

    os.makedirs("/logs/verifier", exist_ok=True)
    with open(REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)
    with open(REWARD_PATH, "w") as fh:
        json.dump({"score": round(score, 6), "reward": round(score, 6)}, fh)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
