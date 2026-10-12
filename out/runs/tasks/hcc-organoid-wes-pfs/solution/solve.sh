#!/bin/bash
set -euo pipefail

mkdir -p /app/output

python3 - <<'PY'
import json

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats

DATA = "/app/data"
OUT = "/app/output"
TAB = chr(9)
CHROMS = [str(i) for i in range(1, 23)] + ["X"]
CHROM_INDEX = {c: i for i, c in enumerate(CHROMS)}
PAIRS = ["HCC-01", "HCC-04", "HCC-08", "HCC-12", "HCC-17", "HCC-23"]

clin = pd.read_csv(DATA + "/clinical.tsv", sep=TAB)
n_patients = int(len(clin))
est = clin[clin["organoid_established"] == 1].copy()
n_established = int(len(est))
rate = round(100.0 * n_established / n_patients, 1)
morph = {"cystic": int((est["organoid_morphology"] == "cystic").sum()),
         "solid": int((est["organoid_morphology"] == "solid").sum())}

def fisher(mask):
    a = int(((clin["organoid_established"] == 1) & mask).sum())
    b = int(((clin["organoid_established"] == 1) & (~mask)).sum())
    c = int(((clin["organoid_established"] == 0) & mask).sum())
    d = int(((clin["organoid_established"] == 0) & (~mask)).sum())
    return float(stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")[1])

p_age = fisher(clin["age"] > 70)
p_hbv = fisher(clin["etiology"] == "HBV")
p_bcat = fisher(clin["bcat_positive"] == 1)

snps = pd.read_csv(DATA + "/common_snps.tsv", sep=TAB)
snp_keys = set(zip(snps["chrom"], snps["pos"].astype(int), snps["ref"], snps["alt"]))

def load_sample(pid, kind):
    df = pd.read_csv(DATA + "/variants/" + pid + "_" + kind + ".tsv", sep=TAB)
    df = df[(df["depth"] >= 20) & (df["vaf"] >= 0.10)].copy()
    keys = list(zip(df["chrom"], df["pos"].astype(int), df["ref"], df["alt"]))
    df = df[[k not in snp_keys for k in keys]]
    return df

counts = {}
pair_rows = []
shared_rows = []

for pid in sorted(PAIRS):
    t = load_sample(pid, "tumor")
    o = load_sample(pid, "organoid")
    counts[pid + "_tumor"] = int(len(t))
    counts[pid + "_organoid"] = int(len(o))

    t_map = {(r.chrom, int(r.pos), r.ref, r.alt): (float(r.vaf), r.gene) for r in t.itertuples()}
    o_map = {(r.chrom, int(r.pos), r.ref, r.alt): float(r.vaf) for r in o.itertuples()}

    shared = sorted(set(t_map) & set(o_map), key=lambda k: (CHROM_INDEX[k[0]], k[1]))
    n_shared = len(shared)
    n_t_only = len(t_map) - n_shared
    n_o_only = len(o_map) - n_shared
    union = len(t_map) + len(o_map) - n_shared
    jaccard = n_shared / float(union) if union else 0.0

    tv = [t_map[k][0] for k in shared]
    ov = [o_map[k] for k in shared]
    rho = float(stats.spearmanr(tv, ov)[0]) if len(shared) >= 3 else 0.0
    mad = float(np.median(np.abs(np.array(tv) - np.array(ov)))) if shared else 0.0
    fidelity = bool(jaccard >= 0.60 and rho >= 0.70)

    pair_rows.append([pid, n_shared, n_t_only, n_o_only, "%.4f" % jaccard,
                      "%.4f" % rho, "%.4f" % mad, "true" if fidelity else "false"])
    for key in shared:
        shared_rows.append([pid, key[0], str(key[1]), key[2], key[3], t_map[key][1],
                            "%.4f" % t_map[key][0], "%.4f" % o_map[key]])

times = est["pfs_months"].astype(float)
events = est["pfs_event"].astype(int)
is_solid = (est["organoid_morphology"] == "solid").astype(int)
surv = pd.DataFrame({"T": times, "E": events, "solid": is_solid})
cph = CoxPHFitter()
cph.fit(surv, duration_col="T", event_col="E")
hr = float(cph.hazard_ratios_["solid"])

solid_df = surv[surv["solid"] == 1]
cystic_df = surv[surv["solid"] == 0]
p_logrank = float(logrank_test(solid_df["T"], cystic_df["T"],
                               solid_df["E"], cystic_df["E"]).p_value)
med_solid = float(KaplanMeierFitter().fit(solid_df["T"], solid_df["E"]).median_survival_time_)
med_cystic = float(KaplanMeierFitter().fit(cystic_df["T"], cystic_df["E"]).median_survival_time_)

summary = {
    "n_patients": n_patients,
    "n_organoid_established": n_established,
    "establishment_rate_pct": rate,
    "morphology_counts": morph,
    "filtered_variant_counts": counts,
    "n_fidelity_pass": sum(1 for r in pair_rows if r[7] == "true"),
    "cox_hr_solid_vs_cystic": round(hr, 4),
    "logrank_p": p_logrank,
    "median_pfs_months": {"cystic": med_cystic, "solid": med_solid},
    "establishment_p_values": {"age_gt70": p_age, "hbv": p_hbv, "bcat_positive": p_bcat},
}
with open(OUT + "/summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)

with open(OUT + "/pair_concordance.tsv", "w", encoding="utf-8") as fh:
    fh.write(TAB.join(["patient_id", "shared", "tumor_only", "organoid_only", "jaccard",
                       "vaf_spearman", "median_abs_vaf_diff", "fidelity_pass"]) + chr(10))
    for row in pair_rows:
        fh.write(TAB.join(str(x) for x in row) + chr(10))

with open(OUT + "/shared_variants.tsv", "w", encoding="utf-8") as fh:
    fh.write(TAB.join(["patient_id", "chrom", "pos", "ref", "alt", "gene",
                       "tumor_vaf", "organoid_vaf"]) + chr(10))
    for row in shared_rows:
        fh.write(TAB.join(str(x) for x in row) + chr(10))

print(json.dumps(summary, indent=2))
PY
