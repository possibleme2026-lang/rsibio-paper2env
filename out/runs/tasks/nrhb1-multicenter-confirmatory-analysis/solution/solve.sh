#!/bin/bash
set -euo pipefail
mkdir -p /app/out
cat > /app/solve.py <<'PY'
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

DATA = "/app/data"
OUT = "/app/out"
TARGETS = ["MBP", "ERBB2", "pERBB2", "ERK1_2", "pERK1_2", "YAP1"]
PATHWAYS = ["Hippo_YAP", "mTORC1", "RTK_Ras_MAPK"]


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
    return {"n": int(r.nobs), "effect": float(r.params["_arm"]), "se": float(r.bse["_arm"]),
            "t": float(r.tvalues["_arm"]), "p": float(r.pvalues["_arm"]),
            "ci_low": float(ci.loc["_arm", 0]), "ci_high": float(ci.loc["_arm", 1])}


def impute_extremes(nerve, mode):
    d = nerve.copy()
    tr = d["arm"] == "rhNRGb1"
    obs_tr = d.loc[tr, "week13_um"].dropna()
    obs_ve = d.loc[~tr, "week13_um"].dropna()
    if mode == "best":
        ft, fv = obs_tr.min(), obs_ve.max()
    else:
        ft, fv = obs_tr.max(), obs_ve.min()
    d.loc[tr & d["week13_um"].isna(), "week13_um"] = ft
    d.loc[(~tr) & d["week13_um"].isna(), "week13_um"] = fv
    return d


nerve = pd.read_csv(os.path.join(DATA, "nerve_thickness.csv"), na_values=[""])
west = pd.read_csv(os.path.join(DATA, "western_blot.csv"), na_values=[""])
prot = pd.read_csv(os.path.join(DATA, "proteomics_panel.tsv"), sep="\t", na_values=[""])
paths = pd.read_csv(os.path.join(DATA, "protein_panel_pathways.tsv"), sep="\t")

# ---- A: primary outcome ----
prim = ols_arm(nerve, "week13_um", covars=["baseline_um"])
tr = nerve["arm"] == "rhNRGb1"
res_primary = {
    "n_complete": prim["n"],
    "n_missing_treated": int((tr & nerve["week13_um"].isna()).sum()),
    "n_missing_vehicle": int(((~tr) & nerve["week13_um"].isna()).sum()),
    "effect_um": prim["effect"],
    "se_um": prim["se"],
    "t_stat": prim["t"],
    "p_value": prim["p"],
    "ci95_low": prim["ci_low"],
    "ci95_high": prim["ci_high"],
    "p_best_case": ols_arm(impute_extremes(nerve, "best"), "week13_um", ["baseline_um"])["p"],
    "p_worst_case": ols_arm(impute_extremes(nerve, "worst"), "week13_um", ["baseline_um"])["p"],
}

# ---- B: western blot ----
west_res = {}
ps = []
for t in TARGETS:
    w = west.copy()
    w["_y"] = np.log2(w[t].astype(float) / w["actin"].astype(float))
    r = ols_arm(w, "_y")
    west_res[t] = {"effect_log2": r["effect"], "p_value": r["p"]}
    ps.append(r["p"])
padj = multipletests(np.asarray(ps), method="fdr_bh")[1]
for t, q in zip(TARGETS, padj):
    west_res[t]["p_adj"] = float(q)

# ---- C: proteomics ----
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
    rows.append({"protein": p, "effect": r["effect"], "se": r["se"], "t": r["t"], "p": r["p"]})
    ps.append(r["p"])
tab = pd.DataFrame(rows)
tab["padj"] = multipletests(np.asarray(ps), method="fdr_bh")[1]
tab = tab.sort_values(["padj", "protein"], kind="mergesort").reset_index(drop=True)

tab.to_csv(os.path.join(OUT, "proteomics_results.tsv"), sep="\t", index=False, float_format="%.6g")

pw = {}
for name in PATHWAYS:
    members = set(paths.loc[paths["pathway"] == name, "protein"])
    s2 = tab[tab["protein"].isin(members)]
    pw[name] = {"n_members_retained": int(len(s2)),
                "mean_effect": float(s2["effect"].mean()) if len(s2) else float("nan"),
                "n_significant": int((s2["padj"] < 0.05).sum())}

results = {
    "primary": res_primary,
    "western": west_res,
    "proteomics": {
        "n_proteins_input": int(prot["protein"].nunique()),
        "n_proteins_retained": int(len(retained)),
        "n_significant_padj": int((tab["padj"] < 0.05).sum()),
        "n_abs_effect_ge_0.585": int((tab["effect"].abs() >= 0.585).sum()),
        "pathways": pw,
    },
    "primary_conclusion": "efficacy" if (prim["p"] < 0.05 and prim["effect"] < 0.0) else "no_efficacy",
}
with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(results, fh, indent=2)
print(json.dumps(results, indent=2)[:1200])
PY
python /app/solve.py
ls -l /app/out
