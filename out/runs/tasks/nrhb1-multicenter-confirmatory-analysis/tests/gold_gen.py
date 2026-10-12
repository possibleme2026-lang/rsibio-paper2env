#!/usr/bin/env python3
"""Deterministic generator for the rhNRGbeta1 multicenter confirmatory dataset.

Usage: python gold_gen.py [outdir]        # default: /app/data

Writes nerve_thickness.csv, western_blot.csv, proteomics_panel.tsv and
protein_panel_pathways.tsv. Contains no analysis results: the gold standard
of the task is recomputed from these raw files by the grader.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

SEED = 20260731
SITES = ["S1", "S2", "S3"]
ARMS = ["Vehicle", "rhNRGb1"]
SEXES = ["male", "female"]
N_PER_BLOCK = 18
WESTERN_TARGETS = ["MBP", "ERBB2", "pERBB2", "ERK1_2", "pERK1_2", "YAP1"]

PANEL = [
    ("YAP1", "Hippo_YAP"), ("WWTR1", "Hippo_YAP"), ("LATS1", "Hippo_YAP"),
    ("LATS2", "Hippo_YAP"), ("STK3", "Hippo_YAP"), ("STK4", "Hippo_YAP"),
    ("TEAD1", "Hippo_YAP"), ("TEAD2", "Hippo_YAP"), ("AMOTL2", "Hippo_YAP"),
    ("NF2", "Hippo_YAP"), ("SAV1", "Hippo_YAP"), ("MOB1A", "Hippo_YAP"),
    ("CCN1", "Hippo_YAP"), ("CCN2", "Hippo_YAP"), ("AXL", "Hippo_YAP"),
    ("MTOR", "mTORC1"), ("RPTOR", "mTORC1"), ("RICTOR", "mTORC1"),
    ("RPS6KB1", "mTORC1"), ("RPS6", "mTORC1"), ("EIF4EBP1", "mTORC1"),
    ("AKT1", "mTORC1"), ("TSC1", "mTORC1"), ("TSC2", "mTORC1"),
    ("RHEB", "mTORC1"), ("DEPTOR", "mTORC1"), ("MLST8", "mTORC1"),
    ("LAMTOR1", "mTORC1"), ("RRAGD", "mTORC1"), ("SLC7A5", "mTORC1"),
    ("ERBB2", "RTK_Ras_MAPK"), ("ERBB3", "RTK_Ras_MAPK"), ("NRG1", "RTK_Ras_MAPK"),
    ("HRAS", "RTK_Ras_MAPK"), ("KRAS", "RTK_Ras_MAPK"), ("BRAF", "RTK_Ras_MAPK"),
    ("MAP2K1", "RTK_Ras_MAPK"), ("MAP2K2", "RTK_Ras_MAPK"), ("MAPK1", "RTK_Ras_MAPK"),
    ("MAPK3", "RTK_Ras_MAPK"), ("RAF1", "RTK_Ras_MAPK"), ("SOS1", "RTK_Ras_MAPK"),
    ("GRB2", "RTK_Ras_MAPK"), ("PTPN11", "RTK_Ras_MAPK"), ("SHC1", "RTK_Ras_MAPK"),
    ("MBP", "Differentiation"), ("MPZ", "Differentiation"), ("PMP22", "Differentiation"),
    ("S100B", "Differentiation"), ("SOX10", "Differentiation"),
    ("GAPDH", "Metabolic"), ("PKM", "Metabolic"), ("LDHA", "Metabolic"),
    ("HK2", "Metabolic"), ("PFKP", "Metabolic"), ("ACLY", "Metabolic"),
    ("FASN", "Metabolic"), ("SLC2A1", "Metabolic"), ("PDK1", "Metabolic"),
    ("CPT1A", "Metabolic"),
]


def main(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ------------------------- animal-level design -------------------------
    rows = []
    mid = 0
    for s in SITES:
        for arm in ARMS:
            for sex in SEXES:
                for _ in range(N_PER_BLOCK):
                    mid += 1
                    rows.append(("MB-%03d" % mid, s, sex, arm))
    mice = pd.DataFrame(rows, columns=["mouse_id", "site", "sex", "arm"])
    n = len(mice)
    site = mice["site"].to_numpy()
    sex = mice["sex"].to_numpy()
    arm = mice["arm"].to_numpy()

    # --------------------------- primary outcome ---------------------------
    baseline = rng.normal(520.0, 55.0, n)
    site_shift = pd.Series(site).map({"S1": 0.0, "S2": -18.0, "S3": 21.0}).to_numpy()
    sex_shift = np.where(sex == "female", 12.0, -9.0)
    wk13 = baseline + site_shift + sex_shift + rng.normal(0.0, 58.0, n)

    p_miss = np.full(n, 0.05)
    p_miss += np.where(site == "S1", 0.05, 0.0)
    p_miss += np.where(site == "S3", 0.15, 0.0)
    p_miss += np.where((site == "S3") & (arm == "Vehicle"), 0.12, 0.0)
    missing = rng.random(n) < p_miss
    wk13_obs = wk13.copy()
    wk13_obs[missing] = np.nan

    nerve = mice.copy()
    nerve["baseline_um"] = np.round(baseline, 3)
    nerve["week13_um"] = np.round(wk13_obs, 3)
    nerve.to_csv(os.path.join(outdir, "nerve_thickness.csv"), index=False, na_rep="")

    # ------------------------------ western blot ---------------------------
    sel = np.sort(rng.permutation(n)[:144])
    w = mice.iloc[sel].reset_index(drop=True)
    m = len(w)
    actin = np.exp(rng.normal(12.0, 0.30, m))
    wb = {
        "mouse_id": w["mouse_id"],
        "site": w["site"],
        "sex": w["sex"],
        "arm": w["arm"],
        "actin": np.round(actin, 2),
    }
    base_ab = {"MBP": 0.05, "ERBB2": -0.02, "pERBB2": -0.70,
               "ERK1_2": 0.30, "pERK1_2": -1.20, "YAP1": 0.10}
    for target in WESTERN_TARGETS:
        site_effect = pd.Series(w["site"]).map(
            dict(zip(SITES, rng.normal(0.0, 0.12, len(SITES))))
        ).to_numpy()
        wb[target] = np.round(actin * np.exp(base_ab[target] + site_effect + rng.normal(0.0, 0.38, m)), 2)
    pd.DataFrame(wb).to_csv(os.path.join(outdir, "western_blot.csv"), index=False)

    # ------------------------------- proteomics ----------------------------
    names = [p for p, _ in PANEL]
    n_prot = len(names)
    base = rng.normal(22.0, 1.2, n_prot)
    loading = rng.normal(0.0, 0.22, n)
    site_off = rng.normal(0.0, 0.13, size=(n_prot, len(SITES)))
    sex_off = rng.normal(0.0, 0.10, size=(n_prot, len(SEXES)))
    eff = np.zeros(n_prot)
    chosen = rng.choice(n_prot, size=11, replace=False)
    eff[chosen] = rng.choice([-1.0, 1.0], size=11) * rng.uniform(0.12, 0.80, size=11)
    miss_p = rng.uniform(0.01, 0.08, n_prot)
    poor = rng.choice(n_prot, size=5, replace=False)
    miss_p[poor] = rng.uniform(0.22, 0.45, size=5)

    site_idx = pd.Series(site).map({s: i for i, s in enumerate(SITES)}).to_numpy()
    sex_idx = (sex == "female").astype(int)
    arm_num = (arm == "rhNRGb1").astype(float)

    recs = []
    for j in range(n):
        for p in range(n_prot):
            if rng.random() < miss_p[p]:
                val = np.nan
            else:
                l2 = (base[p] + loading[j] + site_off[p, site_idx[j]]
                      + sex_off[p, sex_idx[j]] + eff[p] * arm_num[j]
                      + rng.normal(0.0, 0.40))
                val = 2.0 ** l2
            recs.append((mice["mouse_id"].iloc[j], site[j], sex[j], arm[j], names[p], val))
    prot = pd.DataFrame(recs, columns=["sample_id", "site", "sex", "arm", "protein", "intensity"])
    prot["intensity"] = prot["intensity"].round(2)
    prot.to_csv(os.path.join(outdir, "proteomics_panel.tsv"), sep="\t", index=False, na_rep="")

    pd.DataFrame(PANEL, columns=["protein", "pathway"]).to_csv(
        os.path.join(outdir, "protein_panel_pathways.tsv"), sep="\t", index=False
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/app/data")
