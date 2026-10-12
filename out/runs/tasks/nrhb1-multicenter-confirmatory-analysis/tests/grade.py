#!/usr/bin/env python3
"""Grader for bioforge/nrhb1-multicenter-confirmatory-analysis.

Regenerates the dataset deterministically from a protected seed, recomputes the
whole pre-specified analysis plan, and compares the submission constraint by
constraint. score = passed / total  (partial credit).
"""
import hashlib
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

DATA = "/app/data"
OUT = "/app/out"
RESULT_JSON = os.path.join(OUT, "results.json")
RESULT_TSV = os.path.join(OUT, "proteomics_results.tsv")
REPORT_DIR = "/logs/verifier"
REFDIR = "/tmp/gold_ref"
TARGETS = ["MBP", "ERBB2", "pERBB2", "ERK1_2", "pERK1_2", "YAP1"]
PATHWAYS = ["Hippo_YAP", "mTORC1", "RTK_Ras_MAPK"]
TSV_COLS = ["protein", "effect", "se", "t", "p", "padj"]
FILES = ["nerve_thickness.csv", "western_blot.csv", "proteomics_panel.tsv",
         "protein_panel_pathways.tsv"]

CHECKS = []


def add(name, ok, detail=""):
    CHECKS.append({"name": name, "passed": bool(ok), "detail": str(detail)[:500]})


def close(a, g, atol, rtol=0.02):
    try:
        a = float(a)
        g = float(g)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(a) and np.isfinite(g)):
        return False
    return abs(a - g) <= atol + rtol * abs(g)


def get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def bh(pvals):
    return multipletests(np.asarray(pvals, dtype=float), method="fdr_bh")[1]


def ols_arm(df, ycol, covars=()):
    d = df.dropna(subset=[ycol]).copy()
    y = d[ycol].astype(float).to_numpy()
    X = pd.DataFrame({"_arm": (d["arm"] == "rhNRGb1").astype(float).to_numpy()})
    for c in covars:
        X[c] = d[c].astype(float).to_numpy()
    for col in ("site", "sex"):
        dm = pd.get_dummies(d[col], prefix=col, drop_first=True).astype(float)
        for c in dm.columns:
            X[c] = dm[c].to_numpy()
    X = sm.add_constant(X, has_constant="add")
    r = sm.OLS(y, X).fit()
    ci = r.conf_int()
    return {
        "n": int(r.nobs),
        "effect": float(r.params["_arm"]),
        "se": float(r.bse["_arm"]),
        "t": float(r.tvalues["_arm"]),
        "p": float(r.pvalues["_arm"]),
        "ci_low": float(ci.loc["_arm", 0]),
        "ci_high": float(ci.loc["_arm", 1]),
    }


def impute_extremes(nerve, mode):
    d = nerve.copy()
    tr = d["arm"] == "rhNRGb1"
    obs_tr = d.loc[tr, "week13_um"].dropna()
    obs_ve = d.loc[~tr, "week13_um"].dropna()
    if mode == "best":
        fill_tr, fill_ve = obs_tr.min(), obs_ve.max()
    else:
        fill_tr, fill_ve = obs_tr.max(), obs_ve.min()
    d.loc[tr & d["week13_um"].isna(), "week13_um"] = fill_tr
    d.loc[(~tr) & d["week13_um"].isna(), "week13_um"] = fill_ve
    return d


def load_data(datadir):
    nerve = pd.read_csv(os.path.join(datadir, "nerve_thickness.csv"), na_values=[""])
    west = pd.read_csv(os.path.join(datadir, "western_blot.csv"), na_values=[""])
    prot = pd.read_csv(os.path.join(datadir, "proteomics_panel.tsv"), sep="\t", na_values=[""])
    paths = pd.read_csv(os.path.join(datadir, "protein_panel_pathways.tsv"), sep="\t")
    return nerve, west, prot, paths


def gold_primary(nerve):
    g = ols_arm(nerve, "week13_um", covars=["baseline_um"])
    tr = nerve["arm"] == "rhNRGb1"
    g["n_missing_treated"] = int((tr & nerve["week13_um"].isna()).sum())
    g["n_missing_vehicle"] = int(((~tr) & nerve["week13_um"].isna()).sum())
    g["p_best_case"] = ols_arm(impute_extremes(nerve, "best"), "week13_um", ["baseline_um"])["p"]
    g["p_worst_case"] = ols_arm(impute_extremes(nerve, "worst"), "week13_um", ["baseline_um"])["p"]
    g["conclusion"] = "efficacy" if (g["p"] < 0.05 and g["effect"] < 0.0) else "no_efficacy"
    return g


def gold_western(west):
    out = {}
    ps = []
    for t in TARGETS:
        w = west.copy()
        w["_y"] = np.log2(w[t].astype(float) / w["actin"].astype(float))
        r = ols_arm(w, "_y")
        out[t] = r
        ps.append(r["p"])
    for t, q in zip(TARGETS, bh(ps)):
        out[t]["p_adj"] = float(q)
    return out


def gold_proteomics(prot, paths):
    n_samples = int(prot["sample_id"].nunique())
    counts = prot.groupby("protein")["intensity"].count()
    miss = 1.0 - counts / float(n_samples)
    retained = sorted(miss.index[miss <= 0.20].tolist())
    sub = prot[prot["protein"].isin(retained)].copy()
    sub["_l2"] = np.log2(sub["intensity"].astype(float))
    sub["_l2"] = sub.groupby(["protein", "site"])["_l2"].transform(lambda s: s.fillna(s.median()))
    sub["_norm"] = sub["_l2"] - sub.groupby("sample_id")["_l2"].transform("median")
    rows = []
    ps = []
    for p in retained:
        r = ols_arm(sub[sub["protein"] == p], "_norm")
        rows.append({"protein": p, "effect": r["effect"], "se": r["se"],
                     "t": r["t"], "p": r["p"]})
        ps.append(r["p"])
    tab = pd.DataFrame(rows)
    tab["padj"] = bh(ps)
    tab = tab.sort_values(["padj", "protein"], kind="mergesort").reset_index(drop=True)
    pw = {}
    for name in PATHWAYS:
        members = set(paths.loc[paths["pathway"] == name, "protein"])
        sub2 = tab[tab["protein"].isin(members)]
        pw[name] = {
            "n_members_retained": int(len(sub2)),
            "mean_effect": float(sub2["effect"].mean()) if len(sub2) else float("nan"),
            "n_significant": int((sub2["padj"] < 0.05).sum()),
        }
    return {
        "n_proteins_input": int(prot["protein"].nunique()),
        "n_proteins_retained": int(len(retained)),
        "n_significant_padj": int((tab["padj"] < 0.05).sum()),
        "n_abs_effect_ge_0.585": int((tab["effect"].abs() >= 0.585).sum()),
        "pathways": pw,
        "table": tab,
    }


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    subprocess.run([sys.executable, "/tests/gold_gen.py", REFDIR], check=True, capture_output=True)

    nerve, west, prot, paths = load_data(REFDIR)
    gp = gold_primary(nerve)
    gw = gold_western(west)
    gpr = gold_proteomics(prot, paths)
    gtab = gpr["table"]

    intact = True
    for f in FILES:
        try:
            if md5(os.path.join(DATA, f)) != md5(os.path.join(REFDIR, f)):
                intact = False
        except OSError:
            intact = False
    add("data_integrity", intact, "the four files in /app/data are unmodified")

    sub = None
    if os.path.isfile(RESULT_JSON):
        try:
            sub = json.load(open(RESULT_JSON))
        except Exception:
            sub = None
    files_ok = os.path.isfile(RESULT_TSV) and isinstance(sub, dict)
    add("submission_files_present", files_ok, "results.json parses and proteomics_results.tsv exists")

    p = get(sub, "primary", default={}) or {}
    add("primary_effect_and_se",
        close(get(p, "effect_um"), gp["effect"], 0.2) and close(get(p, "se_um"), gp["se"], 0.05),
        "effect_um=%r (gold %.4f) se_um=%r (gold %.4f)"
        % (get(p, "effect_um"), gp["effect"], get(p, "se_um"), gp["se"]))
    add("primary_t_and_p",
        close(get(p, "t_stat"), gp["t"], 0.1) and close(get(p, "p_value"), gp["p"], 0.01),
        "t=%r (gold %.4f) p=%r (gold %.6g)"
        % (get(p, "t_stat"), gp["t"], get(p, "p_value"), gp["p"]))
    add("primary_confidence_interval",
        close(get(p, "ci95_low"), gp["ci_low"], 0.5) and close(get(p, "ci95_high"), gp["ci_high"], 0.5),
        "ci=%r / gold %.4f .. %.4f"
        % (get(p, "ci95_low"), gp["ci_low"], gp["ci_high"]))
    add("primary_missing_counts",
        get(p, "n_missing_treated") == gp["n_missing_treated"]
        and get(p, "n_missing_vehicle") == gp["n_missing_vehicle"]
        and get(p, "n_complete") == gp["n"],
        "missing treated/vehicle=%r/%r (gold %d/%d), n_complete=%r (gold %d)"
        % (get(p, "n_missing_treated"), get(p, "n_missing_vehicle"),
           gp["n_missing_treated"], gp["n_missing_vehicle"], get(p, "n_complete"), gp["n"]))
    add("primary_p_best_case", close(get(p, "p_best_case"), gp["p_best_case"], 0.01),
        "p_best_case=%r (gold %.6g)" % (get(p, "p_best_case"), gp["p_best_case"]))
    add("primary_p_worst_case", close(get(p, "p_worst_case"), gp["p_worst_case"], 0.01),
        "p_worst_case=%r (gold %.6g)" % (get(p, "p_worst_case"), gp["p_worst_case"]))

    wsub = get(sub, "western", default={}) or {}
    core_ok = True
    rest_ok = True
    for t in TARGETS:
        ok = (close(get(wsub, t, "effect_log2"), gw[t]["effect"], 0.05)
              and close(get(wsub, t, "p_adj"), gw[t]["p_adj"], 0.01))
        if t in ("MBP", "ERBB2"):
            core_ok = core_ok and ok and close(get(wsub, t, "p_value"), gw[t]["p"], 0.01)
        else:
            rest_ok = rest_ok and ok
    add("western_core_markers", core_ok, "MBP/ERBB2 effect_log2, p_value and p_adj within tolerance")
    add("western_remaining_targets", rest_ok, "pERBB2/ERK1_2/pERK1_2/YAP1 effect_log2 and p_adj within tolerance")

    psub = get(sub, "proteomics", default={}) or {}
    add("proteomics_retained_count",
        get(psub, "n_proteins_retained") == gpr["n_proteins_retained"]
        and get(psub, "n_proteins_input") == gpr["n_proteins_input"],
        "retained=%r (gold %d), input=%r (gold %d)"
        % (get(psub, "n_proteins_retained"), gpr["n_proteins_retained"],
           get(psub, "n_proteins_input"), gpr["n_proteins_input"]))
    add("proteomics_hit_counts",
        get(psub, "n_significant_padj") == gpr["n_significant_padj"]
        and get(psub, "n_abs_effect_ge_0.585") == gpr["n_abs_effect_ge_0.585"],
        "n_sig=%r (gold %d), n_abs>=0.585=%r (gold %d)"
        % (get(psub, "n_significant_padj"), gpr["n_significant_padj"],
           get(psub, "n_abs_effect_ge_0.585"), gpr["n_abs_effect_ge_0.585"]))
    path_ok = True
    path_detail = []
    for name in PATHWAYS:
        g = gpr["pathways"][name]
        ok = (get(psub, "pathways", name, "n_members_retained") == g["n_members_retained"]
              and get(psub, "pathways", name, "n_significant") == g["n_significant"]
              and close(get(psub, "pathways", name, "mean_effect"), g["mean_effect"], 0.02))
        path_ok = path_ok and ok
        path_detail.append("%s:%s (gold %d/%d/%.4f)"
                           % (name, "ok" if ok else "bad", g["n_members_retained"],
                              g["n_significant"], g["mean_effect"]))
    add("proteomics_pathway_summary", path_ok, "; ".join(path_detail))

    tsv = None
    header_ok = False
    if os.path.isfile(RESULT_TSV):
        try:
            with open(RESULT_TSV) as fh:
                first = fh.readline().rstrip("\r\n")
            header_ok = first.split("\t") == TSV_COLS
            tsv = pd.read_csv(RESULT_TSV, sep="\t")
        except Exception:
            tsv = None
    add("proteomics_tsv_format",
        tsv is not None and header_ok and len(tsv) == len(gtab),
        "header_ok=%s rows=%s (gold %d)"
        % (header_ok, None if tsv is None else len(tsv), len(gtab)))

    sort_ok = False
    try:
        s = tsv.sort_values(["padj", "protein"], kind="mergesort").reset_index(drop=True)
        sort_ok = (s["protein"].tolist() == tsv["protein"].tolist()
                   and np.allclose(s["padj"].astype(float).to_numpy(),
                                   tsv["padj"].astype(float).to_numpy(), atol=1e-9))
    except Exception:
        sort_ok = False
    add("proteomics_tsv_sorting", sort_ok, "rows sorted by padj then protein")

    val_ok = False
    detail = ""
    try:
        m = tsv.merge(gtab[["protein", "effect", "p", "padj"]], on="protein",
                      suffixes=("_sub", "_gold"))
        if len(m) == len(gtab):
            e_ok = np.all(np.abs(m["effect_sub"] - m["effect_gold"])
                          <= 0.05 + 0.02 * m["effect_gold"].abs())
            p_ok = np.all(np.abs(m["p_sub"] - m["p_gold"])
                          <= 0.01 + 0.02 * m["p_gold"].abs())
            q_ok = np.all(np.abs(m["padj_sub"] - m["padj_gold"])
                          <= 0.01 + 0.02 * m["padj_gold"].abs())
            val_ok = bool(e_ok and p_ok and q_ok)
            detail = "effect_ok=%s p_ok=%s padj_ok=%s matched_rows=%d" % (e_ok, p_ok, q_ok, len(m))
        else:
            detail = "protein set mismatch: %d matched of %d" % (len(m), len(gtab))
    except Exception as exc:
        detail = "could not evaluate TSV values: %s" % exc
    add("proteomics_tsv_values", val_ok, detail)

    add("primary_conclusion",
        get(sub, "primary_conclusion") == gp["conclusion"],
        "submitted=%r gold=%r" % (get(sub, "primary_conclusion"), gp["conclusion"]))

    passed = sum(1 for c in CHECKS if c["passed"])
    total = len(CHECKS)
    score = round(passed / float(total), 6)

    report = {
        "score": score,
        "passed": passed,
        "total": total,
        "checks": CHECKS,
        "gold": {
            "primary": gp,
            "western": {t: gw[t] for t in TARGETS},
            "proteomics": {k: v for k, v in gpr.items() if k != "table"},
        },
    }
    with open(os.path.join(REPORT_DIR, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(os.path.join(REPORT_DIR, "reward.json"), "w") as fh:
        json.dump({"score": float(score), "reward": float(score)}, fh)

    print("passed %d/%d -> score %.4f" % (passed, total, score))
    for c in CHECKS:
        print("  [%s] %s :: %s" % ("PASS" if c["passed"] else "FAIL", c["name"], c["detail"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        try:
            os.makedirs(REPORT_DIR, exist_ok=True)
            with open(os.path.join(REPORT_DIR, "report.json"), "w") as fh:
                json.dump({"score": 0.0, "error": tb}, fh, indent=2)
            with open(os.path.join(REPORT_DIR, "reward.json"), "w") as fh:
                json.dump({"score": 0.0, "reward": 0.0}, fh)
        except Exception:
            pass
        print(tb)
        sys.exit(0)
