#!/usr/bin/env python3
"""Verifier for bioforge/fish-morphology-life-history-phylo-prediction.

Recomputes the reference diagnostics from the protected seed, scores ten
independently checkable constraints with partial credit and writes
/logs/verifier/reward.json with {"score": s, "reward": s}.
"""
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gold_gen  # noqa: E402
from gold_gen import LATERAL, TRAITS, norm  # noqa: E402

PRED_PATH = "/app/output/predictions.csv"
DIAG_PATH = "/app/output/diagnostics.json"
REWARD_PATH = "/logs/verifier/reward.json"

W_FORMAT = 0.08
W_COVERAGE = 0.05
W_RANGE = 0.05
W_SCHEMA = 0.04
W_COUNTS = 0.08
W_VAR = 0.08
W_LOAD = 0.12
W_SLOPE = 0.15
W_GT = 0.20
W_REP = 0.15

GEN_LO, GEN_HI = 0.30, 0.72
REP_LO, REP_HI = 0.30, 0.65

REQUIRED_LOAD_KEYS = set(TRAITS)


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else 0.0


def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, float)
    ranks[order] = np.arange(a.size, dtype=float)
    sorted_a = a[order]
    i = 0
    while i < sorted_a.size:
        j = i
        while j + 1 < sorted_a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def clip01(v):
    return max(0.0, min(1.0, float(v)))


def as_float(obj, key=None):
    try:
        val = obj[key] if key is not None else obj
        val = float(val)
    except (TypeError, ValueError, KeyError):
        return None
    if not math.isfinite(val):
        return None
    return val


def main():
    checks = []

    def add(name, weight, frac, note=""):
        frac = clip01(frac)
        checks.append({"name": name, "weight": weight, "score": weight * frac,
                       "frac": frac, "note": note})

    gold = gold_gen.generate()
    obs = gold_gen.observable(gold)
    ref = gold_gen.reference_pipeline(obs)
    truth = gold_gen.hidden_truth(gold)
    test_keys = sorted(truth.keys())

    # ---------------------------------------------------------------- predictions
    pred = {}
    n_rows = 0
    valid_rows = 0
    header_ok = False
    if os.path.exists(PRED_PATH):
        try:
            with open(PRED_PATH, newline="") as fh:
                reader = csv.DictReader(fh)
                fields = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}
                need = ["species", "generation_turnover", "reproductive_output"]
                header_ok = all(k in fields for k in need)
                if header_ok:
                    rows = list(reader)
                    n_rows = len(rows)
                    for row in rows:
                        key = norm(row.get(fields["species"], ""))
                        gt = as_float(row, fields["generation_turnover"])
                        rp = as_float(row, fields["reproductive_output"])
                        if gt is None or rp is None or not key or key in pred:
                            continue
                        pred[key] = (gt, rp)
                        valid_rows += 1
        except Exception as exc:  # malformed file must not crash the verifier
            add("prediction_file_parse_error", 0.0, 0.0, str(exc))

    frac = 0.0
    if header_ok:
        frac = 0.4 + 0.6 * min(1.0, valid_rows / 50.0)
        if n_rows != 50:
            frac *= 0.7
    add("predictions_format", W_FORMAT, frac,
        "header_ok=%s rows=%d usable=%d" % (header_ok, n_rows, valid_rows))

    common = sorted(set(pred) & set(test_keys))
    add("test_species_coverage", W_COVERAGE, len(common) / 50.0,
        "%d/50 test species present and parseable" % len(common))

    in_range = 0
    for key in common:
        gt, rp = pred[key]
        if 0.1 < gt < 100.0 and 0.005 < rp < 500.0:
            in_range += 1
    add("biological_range_validity", W_RANGE, in_range / 50.0,
        "%d/50 predictions inside the accepted ranges" % in_range)

    # ---------------------------------------------------------------- diagnostics
    diag = None
    if os.path.exists(DIAG_PATH):
        try:
            with open(DIAG_PATH) as fh:
                diag = json.load(fh)
        except Exception:
            diag = None
    if not isinstance(diag, dict):
        diag = {}

    schema_hits = 0
    schema_total = 7
    if isinstance(diag.get("n_species_analysed"), (int, float)):
        schema_hits += 1
    if isinstance(diag.get("n_training_species"), (int, float)):
        schema_hits += 1
    pp = diag.get("phylo_pca_morphology")
    if isinstance(pp, dict) and as_float(pp.get("pc1_variance_explained")) is not None:
        schema_hits += 1
    loadings = pp.get("pc1_loadings") if isinstance(pp, dict) else None
    if isinstance(loadings, dict) and set(loadings) == REQUIRED_LOAD_KEYS:
        if all(as_float(loadings[k]) is not None for k in loadings):
            schema_hits += 1
    slopes = diag.get("pgls_log_generation_turnover")
    if isinstance(slopes, dict) and set(slopes) == set(LATERAL):
        if all(as_float(slopes[k]) is not None for k in slopes):
            schema_hits += 1
    if as_float(diag.get("cv_rmse_log_generation_turnover")) is not None:
        schema_hits += 1
    if as_float(diag.get("cv_rmse_log_reproductive_output")) is not None:
        schema_hits += 1
    add("diagnostics_schema", W_SCHEMA, schema_hits / schema_total,
        "%d/%d schema elements valid" % (schema_hits, schema_total))

    n_hits = 0
    got_n = as_float(diag.get("n_species_analysed"))
    got_t = as_float(diag.get("n_training_species"))
    if got_n is not None and int(round(got_n)) == ref["n_species"]:
        n_hits += 1
    if got_t is not None and int(round(got_t)) == ref["n_train"]:
        n_hits += 1
    add("sample_counts", W_COUNTS, n_hits / 2.0,
        "agent=(%s,%s) reference=(%d,%d)" % (got_n, got_t, ref["n_species"], ref["n_train"]))

    got_var = as_float(pp.get("pc1_variance_explained")) if isinstance(pp, dict) else None
    if got_var is None:
        add("phylo_pc1_variance_explained", W_VAR, 0.0, "missing")
    else:
        delta = abs(got_var - ref["var_expl"])
        frac = 1.0 if delta <= 0.06 else (0.5 if delta <= 0.12 else 0.0)
        add("phylo_pc1_variance_explained", W_VAR, frac,
            "agent=%.4f reference=%.4f delta=%.4f" % (got_var, ref["var_expl"], delta))

    per_load = W_LOAD / len(TRAITS)
    for i, trait in enumerate(TRAITS):
        if not isinstance(loadings, dict):
            add("phylo_pc1_loading_%s" % trait, per_load, 0.0, "missing")
            continue
        got = as_float(loadings.get(trait))
        if got is None:
            add("phylo_pc1_loading_%s" % trait, per_load, 0.0, "missing")
            continue
        delta = abs(got - ref["loadings"][i])
        frac = 1.0 if delta <= 0.20 else (0.5 if delta <= 0.35 else 0.0)
        add("phylo_pc1_loading_%s" % trait, per_load, frac,
            "agent=%.3f reference=%.3f delta=%.3f" % (got, ref["loadings"][i], delta))

    per_slope = W_SLOPE / len(LATERAL)
    for i, trait in enumerate(LATERAL):
        if not isinstance(slopes, dict):
            add("pgls_slope_%s" % trait, per_slope, 0.0, "missing")
            continue
        got = as_float(slopes.get(trait))
        if got is None:
            add("pgls_slope_%s" % trait, per_slope, 0.0, "missing")
            continue
        expected = ref["slopes"][i]
        tol1 = max(0.06, 0.15 * abs(expected))
        delta = abs(got - expected)
        frac = 1.0 if delta <= tol1 else (0.5 if delta <= 2.0 * tol1 else 0.0)
        add("pgls_slope_%s" % trait, per_slope, frac,
            "agent=%.3f reference=%.3f tol=%.3f" % (got, expected, tol1))

    # ---------------------------------------------------------------- accuracy
    def accuracy(key_index, lo, hi):
        true_vals = []
        pred_vals = []
        for key in common:
            true_vals.append(truth[key][key_index])
            pred_vals.append(pred[key][key_index])
        if len(true_vals) < 5:
            return 0.0, 0.0
        tv = np.asarray(true_vals, float)
        pv = np.asarray(pred_vals, float)
        if (pv <= 0).any() or (tv <= 0).any():
            rho = spearman(pv, tv)
            return rho, max(0.0, rho)
        log_r = pearson(np.log10(pv), np.log10(tv))
        rho = spearman(pv, tv)
        r = max(log_r, rho)
        return r, clip01((r - lo) / (hi - lo))

    r_gt, frac_gt = accuracy(0, GEN_LO, GEN_HI)
    add("generation_turnover_accuracy", W_GT, frac_gt,
        "r=%.3f (full credit at r>=%.2f)" % (r_gt, GEN_HI))

    r_rep, frac_rep = accuracy(1, REP_LO, REP_HI)
    add("reproductive_output_accuracy", W_REP, frac_rep,
        "r=%.3f (full credit at r>=%.2f)" % (r_rep, REP_HI))

    # ---------------------------------------------------------------- report
    total = sum(c["score"] for c in checks)
    total = clip01(total)
    print("constraint breakdown")
    for c in checks:
        print("  %-38s weight=%.2f frac=%.2f score=%.3f  %s"
              % (c["name"], c["weight"], c["frac"], c["score"], c["note"]))
    print("TOTAL SCORE: %.4f" % total)

    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(REWARD_PATH, "w") as fh:
        json.dump({"score": total, "reward": total}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
