#!/usr/bin/env python3
import sys, os, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from tlr_sim import simulate, SPECIES
import gold_gen

DATA_DIR = "/app/data"
OUT_DIR = "/app/output"
REPORT_PATH = "/logs/verifier/report.json"
REWARD_PATH = "/logs/verifier/reward.json"
TIME_POINTS = np.array([0,5,15,30,60,120,240,360,480,720], dtype=float)
CONDITIONS = ["WT", "IKKepsilon_KO", "TANK_KO"]
GOLD_KFB = 2.5
GOLD_KFB_K = 0.3

def load_agent():
    with open(os.path.join(OUT_DIR, "result.json")) as f:
        result = json.load(f)
    with open(os.path.join(OUT_DIR, "model_fitted.json")) as f:
        model = json.load(f)
    return result, model

def check_constraints(result, model, gt_df, wt_max):
    checks = []
    required = ["missing_regulator", "scaffold", "affected_node", "feedback_parameters", "model_file"]
    ok = all(k in result for k in required)
    checks.append(("result_keys", ok, "result.json contains required keys"))
    regulator = str(result.get("missing_regulator", "")).lower()
    ok = "ikk" in regulator and ("epsilon" in regulator or "ε" in regulator)
    checks.append(("regulator", ok, f"missing_regulator={regulator}"))
    scaffold = str(result.get("scaffold", "")).upper()
    ok = "TANK" in scaffold
    checks.append(("scaffold", ok, f"scaffold={scaffold}"))
    node = str(result.get("affected_node", "")).upper()
    ok = "IRAK1" in node and "TRAF6" in node
    checks.append(("affected_node", ok, f"affected_node={node}"))
    ok = model.get("feedback_enabled", False) is True
    checks.append(("feedback_enabled", ok, "model feedback_enabled true"))
    ok = model.get("feedback_enzyme") == "E_act"
    checks.append(("feedback_enzyme", ok, "feedback_enzyme E_act"))
    targets = model.get("feedback_targets", [])
    ok = "I_ub" in targets and "T_ub" in targets
    checks.append(("feedback_targets", ok, f"targets={targets}"))
    params = result.get("feedback_parameters", {})
    k_fb = params.get("k_fb", None)
    ok = k_fb is not None and 0.1 <= k_fb <= 20.0 and abs(k_fb - GOLD_KFB) <= GOLD_KFB
    checks.append(("k_fb", ok, f"k_fb={k_fb} expected ~{GOLD_KFB}"))
    K_fb = params.get("K_fb", None)
    ok = K_fb is not None and 0.01 <= K_fb <= 2.0 and abs(K_fb - GOLD_KFB_K) <= GOLD_KFB_K
    checks.append(("K_fb", ok, f"K_fb={K_fb} expected ~{GOLD_KFB_K}"))
    try:
        sims = {}
        for cond in CONDITIONS:
            y = simulate(model, cond, TIME_POINTS)
            sims[cond] = y
        rmse_vals = []
        for cond in CONDITIONS:
            y_norm = sims[cond] / wt_max
            for i, t in enumerate(TIME_POINTS):
                for j, sp in enumerate(SPECIES):
                    gt = gt_df.loc[t, (cond, sp)]
                    pred = y_norm[i, j]
                    rmse_vals.append((pred - gt)**2)
        rmse = np.sqrt(np.mean(rmse_vals))
        ok = rmse < 0.15
        checks.append(("rmse", ok, f"RMSE={rmse:.4f} threshold 0.15"))
    except Exception as e:
        checks.append(("rmse", False, f"simulation failed: {e}"))
    try:
        wt_norm = sims["WT"] / wt_max
        ko_norm = sims["IKKepsilon_KO"] / wt_max
        idx = {sp: j for j, sp in enumerate(SPECIES)}
        t240 = list(TIME_POINTS).index(240)
        for sp in ["NFkB", "MAPK", "Cyt"]:
            ratio = ko_norm[t240, idx[sp]] / (wt_norm[t240, idx[sp]] + 1e-9)
            ok = ratio > 1.5
            checks.append((f"ikk_ko_{sp}", ok, f"{sp} ratio={ratio:.2f}"))
    except Exception as e:
        checks.append(("ikk_ko_phenotype", False, str(e)))
    try:
        tank_norm = sims["TANK_KO"] / wt_max
        t60 = list(TIME_POINTS).index(60)
        e_act = tank_norm[t60, idx["E_act"]]
        ok = e_act < 0.1
        checks.append(("tank_ko_E_act", ok, f"E_act at t=60 = {e_act:.4f}"))
    except Exception as e:
        checks.append(("tank_ko_E_act", False, str(e)))
    try:
        t120 = list(TIME_POINTS).index(120)
        for cond in ["IKKepsilon_KO", "TANK_KO"]:
            ko_norm = sims[cond] / wt_max
            for sp in ["I_ub", "T_ub"]:
                ratio = ko_norm[t120, idx[sp]] / (wt_norm[t120, idx[sp]] + 1e-9)
                ok = ratio > 1.2
                checks.append((f"{cond}_{sp}_increase", ok, f"{sp} ratio={ratio:.2f}"))
    except Exception as e:
        checks.append(("ko_ubiquitin_increase", False, str(e)))
    try:
        t120 = list(TIME_POINTS).index(120)
        for cond in ["IKKepsilon_KO", "TANK_KO"]:
            ko_norm = sims[cond] / wt_max
            sp = "M_ub"
            ratio = ko_norm[t120, idx[sp]] / (wt_norm[t120, idx[sp]] + 1e-9)
            ok = ratio < 1.1
            checks.append((f"{cond}_M_ub", ok, f"M_ub ratio={ratio:.2f}"))
    except Exception as e:
        checks.append(("M_ub_no_increase", False, str(e)))
    ok = os.path.exists(os.path.join(OUT_DIR, "model_fitted.json")) and os.path.exists(os.path.join(OUT_DIR, "result.json"))
    checks.append(("output_files", ok, "model_fitted.json and result.json exist"))
    return checks

def main():
    os.makedirs("/logs/verifier", exist_ok=True)
    try:
        gt_df, wt_max = gold_gen.generate_ground_truth()
        result, model = load_agent()
    except Exception as e:
        print(f"Failed to load inputs: {e}")
        with open(REWARD_PATH, "w") as f:
            json.dump({"score": 0.0, "reward": 0.0}, f)
        return
    checks = check_constraints(result, model, gt_df, wt_max)
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    score = passed / total
    for name, ok, msg in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}")
    print(f"Score: {score:.4f} ({passed}/{total})")
    with open(REPORT_PATH, "w") as f:
        json.dump({"checks": [{"name": n, "pass": o, "message": m} for n, o, m in checks], "score": score}, f, indent=2)
    with open(REWARD_PATH, "w") as f:
        json.dump({"score": score, "reward": score}, f)

if __name__ == "__main__":
    main()
