#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

SEED = 20260802
CELL_LINES = ["H69", "H146", "H526", "DMS79"]
L_DOSES = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
P_DOSES = [0.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
CONDITIONS = ["vehicle", "L_1nM", "L_10nM", "P_30nM", "P_300nM", "L_10nM_P_300nM"]
PROTEINS = ["beta-actin", "pChk1_S296", "pChk1_S317", "pChk1_S345", "pChk2_Y68", "Chk1", "Chk2", "Wee1", "gammaH2AX", "cleaved_PARP"]

PARAMS = {
    "H69": {"lurbinectedin": [0.05, 1.00, 1.5, 1.2], "prexasertib": [0.10, 1.00, 20.0, 1.0]},
    "H146": {"lurbinectedin": [0.08, 1.00, 2.5, 1.4], "prexasertib": [0.12, 1.00, 35.0, 0.9]},
    "H526": {"lurbinectedin": [0.10, 1.00, 4.0, 1.1], "prexasertib": [0.15, 1.00, 50.0, 1.1]},
    "DMS79": {"lurbinectedin": [0.15, 1.00, 10.0, 0.9], "prexasertib": [0.20, 1.00, 80.0, 0.8]},
}

SYNERGY = {
    "H69": {"amp": 0.30, "cL": 3.0, "cP": 30.0, "sL": 0.6, "sP": 0.8},
    "H146": {"amp": 0.25, "cL": 10.0, "cP": 100.0, "sL": 0.7, "sP": 0.7},
    "H526": {"amp": 0.05, "cL": 10.0, "cP": 100.0, "sL": 1.0, "sP": 1.0},
    "DMS79": {"amp": -0.10, "cL": 30.0, "cP": 300.0, "sL": 0.8, "sP": 0.8},
}

BASE_INT = {
    "beta-actin": 1000.0,
    "pChk1_S296": 120.0,
    "pChk1_S317": 90.0,
    "pChk1_S345": 80.0,
    "pChk2_Y68": 150.0,
    "Chk1": 200.0,
    "Chk2": 180.0,
    "Wee1": 100.0,
    "gammaH2AX": 60.0,
    "cleaved_PARP": 40.0,
}

COND_FOLD = {
    "vehicle": {p: 1.0 for p in PROTEINS},
    "L_1nM": {"pChk1_S296": 1.8, "pChk1_S317": 2.0, "pChk1_S345": 2.2, "pChk2_Y68": 2.5, "Wee1": 1.4, "gammaH2AX": 1.2, "cleaved_PARP": 1.1, "Chk1": 1.0, "Chk2": 1.0, "beta-actin": 1.0},
    "L_10nM": {"pChk1_S296": 3.0, "pChk1_S317": 3.5, "pChk1_S345": 3.8, "pChk2_Y68": 4.0, "Wee1": 2.0, "gammaH2AX": 1.5, "cleaved_PARP": 1.3, "Chk1": 1.0, "Chk2": 1.0, "beta-actin": 1.0},
    "P_30nM": {"pChk1_S296": 0.8, "pChk1_S317": 0.7, "pChk1_S345": 0.6, "pChk2_Y68": 0.9, "Wee1": 0.9, "gammaH2AX": 1.1, "cleaved_PARP": 1.0, "Chk1": 1.0, "Chk2": 1.0, "beta-actin": 1.0},
    "P_300nM": {"pChk1_S296": 0.5, "pChk1_S317": 0.4, "pChk1_S345": 0.3, "pChk2_Y68": 0.7, "Wee1": 0.6, "gammaH2AX": 1.3, "cleaved_PARP": 1.2, "Chk1": 1.0, "Chk2": 1.0, "beta-actin": 1.0},
    "L_10nM_P_300nM": {"pChk1_S296": 1.2, "pChk1_S317": 1.0, "pChk1_S345": 0.9, "pChk2_Y68": 1.5, "Wee1": 0.8, "gammaH2AX": 5.0, "cleaved_PARP": 3.0, "Chk1": 1.0, "Chk2": 1.0, "beta-actin": 1.0},
}

CELL_MOD = {
    "H69": {"L_resp": 1.2, "combo_suppress": 0.5, "damage": 1.0},
    "H146": {"L_resp": 1.1, "combo_suppress": 0.5, "damage": 0.9},
    "H526": {"L_resp": 0.8, "combo_suppress": 0.8, "damage": 0.6},
    "DMS79": {"L_resp": 0.5, "combo_suppress": 1.8, "damage": 0.3},
}

def four_pl(dose, bottom, top, ic50, hill):
    dose = np.asarray(dose, dtype=float)
    return bottom + (top - bottom) / (1.0 + (dose / ic50) ** hill)

def synergy_effect(dL, dP, sp):
    if dL <= 0 or dP <= 0:
        return 0.0
    return sp["amp"] * np.exp(-0.5 * ((np.log10(dL) - np.log10(sp["cL"])) / sp["sL"]) ** 2) * np.exp(-0.5 * ((np.log10(dP) - np.log10(sp["cP"])) / sp["sP"]) ** 2)

def true_fold(cl, cond, prot):
    if prot == "beta-actin":
        return 1.0
    base = COND_FOLD[cond].get(prot, 1.0)
    mod = CELL_MOD[cl]
    if cond in ("L_1nM", "L_10nM"):
        if prot in ("pChk1_S296", "pChk1_S317", "pChk1_S345", "pChk2_Y68", "Wee1"):
            return 1.0 + (base - 1.0) * mod["L_resp"]
        else:
            return base
    elif cond == "L_10nM_P_300nM":
        if prot in ("pChk1_S296", "pChk1_S317", "pChk1_S345", "pChk2_Y68", "Wee1"):
            return max(0.1, base * mod["combo_suppress"])
        elif prot in ("gammaH2AX", "cleaved_PARP"):
            return 1.0 + (base - 1.0) * mod["damage"]
        else:
            return base
    else:
        return base

def generate_viability():
    rng = np.random.default_rng(SEED)
    rows = []
    for cl in CELL_LINES:
        pL = PARAMS[cl]["lurbinectedin"]
        pP = PARAMS[cl]["prexasertib"]
        sp = SYNERGY[cl]
        for dL in L_DOSES:
            vL = four_pl(dL, *pL)
            for dP in P_DOSES:
                vP = four_pl(dP, *pP)
                expected = min(vL, vP)
                syn = synergy_effect(dL, dP, sp)
                true_v = np.clip(expected - syn, 0.0, 1.2)
                for rep in range(3):
                    obs = np.clip(true_v + rng.normal(0, 0.025), 0.0, 1.2)
                    rows.append({"cell_line": cl, "lurbinectedin_nM": dL, "prexasertib_nM": dP, "replicate": rep + 1, "viability": round(float(obs), 4)})
    return pd.DataFrame(rows)

def generate_western():
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for cl in CELL_LINES:
        for cond in CONDITIONS:
            for rep in range(3):
                for prot in PROTEINS:
                    base = BASE_INT[prot]
                    fold = true_fold(cl, cond, prot)
                    noise = np.exp(rng.normal(0, 0.08))
                    intensity = base * fold * noise
                    rows.append({"cell_line": cl, "condition": cond, "replicate": rep + 1, "protein": prot, "intensity": round(float(intensity), 2)})
    return pd.DataFrame(rows)

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    viab = generate_viability()
    wb = generate_western()
    viab.to_csv(os.path.join(out_dir, "viability_matrix.csv"), index=False)
    wb.to_csv(os.path.join(out_dir, "western_blot.csv"), index=False)
    print("Generated data files.")

if __name__ == "__main__":
    main()
