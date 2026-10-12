import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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

def get_reference():
    viab = generate_viability()
    wb = generate_western()
    mean_viab = viab.groupby(["cell_line", "lurbinectedin_nM", "prexasertib_nM"])["viability"].mean().reset_index()
    hsa_rows = []
    for cl in CELL_LINES:
        sub = mean_viab[mean_viab["cell_line"] == cl]
        L_map = dict(zip(sub[sub["prexasertib_nM"] == 0]["lurbinectedin_nM"], sub[sub["prexasertib_nM"] == 0]["viability"]))
        P_map = dict(zip(sub[sub["lurbinectedin_nM"] == 0]["prexasertib_nM"], sub[sub["lurbinectedin_nM"] == 0]["viability"]))
        for _, row in sub.iterrows():
            dL = row["lurbinectedin_nM"]
            dP = row["prexasertib_nM"]
            mean_v = row["viability"]
            expected = min(L_map[dL], P_map[dP])
            score = expected - mean_v
            if score > 0.05:
                call = "synergistic"
            elif score < -0.05:
                call = "antagonistic"
            else:
                call = "additive"
            hsa_rows.append({"cell_line": cl, "lurbinectedin_nM": dL, "prexasertib_nM": dP, "mean_viability": mean_v, "expected_hsa": expected, "hsa_score": score, "call": call})
    hsa = pd.DataFrame(hsa_rows)
    ic50_rows = []
    for cl in CELL_LINES:
        for drug in ["lurbinectedin", "prexasertib"]:
            if drug == "lurbinectedin":
                sub = mean_viab[(mean_viab["cell_line"] == cl) & (mean_viab["prexasertib_nM"] == 0)]
                doses = sub["lurbinectedin_nM"].values
                viabs = sub["viability"].values
            else:
                sub = mean_viab[(mean_viab["cell_line"] == cl) & (mean_viab["lurbinectedin_nM"] == 0)]
                doses = sub["prexasertib_nM"].values
                viabs = sub["viability"].values
            mask = doses > 0
            x = np.log10(doses[mask])
            y = viabs[mask]
            bottom0 = max(0.0, min(y))
            top0 = min(1.2, max(y))
            ic500 = np.median(doses[mask])
            hill0 = 1.0
            p0 = [bottom0, top0, ic500, hill0]
            bounds = ([0.0, 0.5, 0.01, 0.1], [0.5, 1.2, 1000.0, 5.0])
            def model(x, bottom, top, ic50, hill):
                return bottom + (top - bottom) / (1.0 + (10.0 ** (x - np.log10(ic50))) ** hill)
            try:
                popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=10000)
            except Exception:
                popt = [bottom0, top0, ic500, hill0]
            y_pred = model(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ic50_rows.append({"cell_line": cl, "drug": drug, "ic50_nM": float(popt[2]), "hill": float(popt[3]), "bottom": float(popt[0]), "top": float(popt[1]), "r2": float(r2)})
    ic50 = pd.DataFrame(ic50_rows)
    wb_wide = wb.pivot_table(index=["cell_line", "condition", "replicate"], columns="protein", values="intensity").reset_index()
    for prot in PROTEINS:
        if prot == "beta-actin":
            continue
        wb_wide[f"norm_{prot}"] = wb_wide[prot] / wb_wide["beta-actin"]
    mech_rows = []
    for cl in CELL_LINES:
        for prot in PROTEINS:
            if prot == "beta-actin":
                continue
            for cond in CONDITIONS:
                sub = wb_wide[(wb_wide["cell_line"] == cl) & (wb_wide["condition"] == cond)]
                median_norm = sub[f"norm_{prot}"].median()
                veh_sub = wb_wide[(wb_wide["cell_line"] == cl) & (wb_wide["condition"] == "vehicle")]
                veh_median = veh_sub[f"norm_{prot}"].median()
                fold = median_norm / veh_median if veh_median > 0 else 0.0
                mech_rows.append({"cell_line": cl, "protein": prot, "condition": cond, "median_fold_change": fold})
    mech = pd.DataFrame(mech_rows)
    mechanism_report = {}
    for cl in CELL_LINES:
        l10 = mech[(mech["cell_line"] == cl) & (mech["condition"] == "L_10nM")]
        pchk1_sites = ["pChk1_S296", "pChk1_S317", "pChk1_S345"]
        pchk1_up = sum(l10[l10["protein"] == s]["median_fold_change"].values[0] > 1.5 for s in pchk1_sites)
        pchk2_up = l10[l10["protein"] == "pChk2_Y68"]["median_fold_change"].values[0] > 1.5
        pchk_up = (pchk1_up >= 2) or pchk2_up
        combo = mech[(mech["cell_line"] == cl) & (mech["condition"] == "L_10nM_P_300nM")]
        l10 = mech[(mech["cell_line"] == cl) & (mech["condition"] == "L_10nM")]
        suppressed = 0
        for prot in ["pChk1_S296", "pChk1_S317", "pChk1_S345", "pChk2_Y68"]:
            combo_val = combo[combo["protein"] == prot]["median_fold_change"].values[0]
            l10_val = l10[l10["protein"] == prot]["median_fold_change"].values[0]
            if combo_val < 0.7 * l10_val:
                suppressed += 1
        combo_supp = suppressed >= 2
        gh2ax = combo[combo["protein"] == "gammaH2AX"]["median_fold_change"].values[0]
        parp = combo[combo["protein"] == "cleaved_PARP"]["median_fold_change"].values[0]
        damage = (gh2ax > 2.0) and (parp > 1.5)
        mechanism_report[cl] = {"pchk_upregulated_by_lurbinectedin": bool(pchk_up), "combination_suppresses_pchk": bool(combo_supp), "dna_damage_apoptosis": bool(damage)}
    synergy_counts = hsa[hsa["call"] == "synergistic"].groupby("cell_line").size()
    hsa_synergy_cell_lines = sorted(list(synergy_counts[synergy_counts >= 3].index))
    max_hsa_score = hsa.groupby("cell_line")["hsa_score"].max().to_dict()
    ic50_nM = {}
    for cl in CELL_LINES:
        ic50_nM[cl] = {}
        for drug in ["lurbinectedin", "prexasertib"]:
            val = ic50[(ic50["cell_line"] == cl) & (ic50["drug"] == drug)]["ic50_nM"].values[0]
            ic50_nM[cl][drug] = float(val)
    report = {"hsa_synergy_cell_lines": hsa_synergy_cell_lines, "max_hsa_score": {k: float(v) for k, v in max_hsa_score.items()}, "ic50_nM": ic50_nM, "mechanism": mechanism_report}
    return viab, wb, hsa, ic50, mech, report
