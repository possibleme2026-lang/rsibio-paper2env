#!/usr/bin/env python3
import os, sys, json, math
import numpy as np
import pandas as pd
sys.path.insert(0, "/tests")
import gold_gen

OUT_DIR = "/app/output"
LOG_DIR = "/logs/verifier"
os.makedirs(LOG_DIR, exist_ok=True)

def load_agent_csv(name):
    path = os.path.join(OUT_DIR, name)
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def load_agent_json(name):
    path = os.path.join(OUT_DIR, name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def main():
    ref_viab, ref_wb, ref_hsa, ref_ic50, ref_mech, ref_report = gold_gen.get_reference()
    constraints = {}
    details = {}

    required_files = ["hsa_scores.csv", "ic50_summary.csv", "mechanism_summary.csv", "report.json"]
    missing = [f for f in required_files if not os.path.isfile(os.path.join(OUT_DIR, f))]
    constraints["outputs_exist"] = len(missing) == 0
    details["outputs_exist"] = {"missing": missing}

    agent_hsa = load_agent_csv("hsa_scores.csv")
    agent_ic50 = load_agent_csv("ic50_summary.csv")
    agent_mech = load_agent_csv("mechanism_summary.csv")
    agent_report = load_agent_json("report.json")

    ok = True
    msg = []
    if agent_hsa is None:
        ok = False
        msg.append("hsa_scores.csv missing or unreadable")
    else:
        expected_cols = ["cell_line", "lurbinectedin_nM", "prexasertib_nM", "mean_viability", "expected_hsa", "hsa_score", "call"]
        if list(agent_hsa.columns) != expected_cols:
            ok = False
            msg.append(f"columns mismatch: {list(agent_hsa.columns)} != {expected_cols}")
        elif len(agent_hsa) != 256:
            ok = False
            msg.append(f"row count {len(agent_hsa)} != 256")
        else:
            ref_sorted = ref_hsa.sort_values(["cell_line", "lurbinectedin_nM", "prexasertib_nM"]).reset_index(drop=True)
            agent_sorted = agent_hsa.sort_values(["cell_line", "lurbinectedin_nM", "prexasertib_nM"]).reset_index(drop=True)
            if not (agent_sorted["cell_line"] == ref_sorted["cell_line"]).all():
                ok = False
                msg.append("cell_line mismatch")
            elif not np.allclose(agent_sorted["lurbinectedin_nM"], ref_sorted["lurbinectedin_nM"], atol=1e-6):
                ok = False
                msg.append("lurbinectedin_nM mismatch")
            elif not np.allclose(agent_sorted["prexasertib_nM"], ref_sorted["prexasertib_nM"], atol=1e-6):
                ok = False
                msg.append("prexasertib_nM mismatch")
            else:
                for col in ["mean_viability", "expected_hsa", "hsa_score"]:
                    if not np.allclose(agent_sorted[col], ref_sorted[col], atol=0.01):
                        ok = False
                        msg.append(f"{col} mismatch beyond 0.01")
                        break
                if ok and not (agent_sorted["call"] == ref_sorted["call"]).all():
                    ok = False
                    msg.append("call mismatch")
    constraints["hsa_scores_match"] = ok
    details["hsa_scores_match"] = msg

    ok = True
    msg = []
    if agent_ic50 is None:
        ok = False
        msg.append("ic50_summary.csv missing or unreadable")
    else:
        expected_cols = ["cell_line", "drug", "ic50_nM", "hill", "bottom", "top", "r2"]
        if list(agent_ic50.columns) != expected_cols:
            ok = False
            msg.append(f"columns mismatch: {list(agent_ic50.columns)} != {expected_cols}")
        elif len(agent_ic50) != 8:
            ok = False
            msg.append(f"row count {len(agent_ic50)} != 8")
        else:
            ref_sorted = ref_ic50.sort_values(["cell_line", "drug"]).reset_index(drop=True)
            agent_sorted = agent_ic50.sort_values(["cell_line", "drug"]).reset_index(drop=True)
            if not (agent_sorted["cell_line"] == ref_sorted["cell_line"]).all() or not (agent_sorted["drug"] == ref_sorted["drug"]).all():
                ok = False
                msg.append("cell_line/drug keys mismatch")
            else:
                rel_err = np.abs(agent_sorted["ic50_nM"] - ref_sorted["ic50_nM"]) / ref_sorted["ic50_nM"].replace(0, np.nan)
                if not (rel_err.fillna(0) <= 0.15).all():
                    ok = False
                    msg.append("ic50_nM relative error > 0.15")
                elif not np.allclose(agent_sorted["hill"], ref_sorted["hill"], atol=0.2):
                    ok = False
                    msg.append("hill absolute error > 0.2")
                else:
                    for col in ["bottom", "top"]:
                        if not np.allclose(agent_sorted[col], ref_sorted[col], atol=0.1):
                            ok = False
                            msg.append(f"{col} absolute error > 0.1")
                            break
                    if ok and not np.allclose(agent_sorted["r2"], ref_sorted["r2"], atol=0.05):
                        ok = False
                        msg.append("r2 absolute error > 0.05")
    constraints["ic50_fits_match"] = ok
    details["ic50_fits_match"] = msg

    ok = True
    msg = []
    if agent_mech is None:
        ok = False
        msg.append("mechanism_summary.csv missing or unreadable")
    else:
        expected_cols = ["cell_line", "protein", "condition", "median_fold_change"]
        if list(agent_mech.columns) != expected_cols:
            ok = False
            msg.append(f"columns mismatch: {list(agent_mech.columns)} != {expected_cols}")
        elif len(agent_mech) != 216:
            ok = False
            msg.append(f"row count {len(agent_mech)} != 216")
        else:
            ref_sorted = ref_mech.sort_values(["cell_line", "protein", "condition"]).reset_index(drop=True)
            agent_sorted = agent_mech.sort_values(["cell_line", "protein", "condition"]).reset_index(drop=True)
            for col in ["cell_line", "protein", "condition"]:
                if not (agent_sorted[col] == ref_sorted[col]).all():
                    ok = False
                    msg.append(f"{col} mismatch")
                    break
            if ok:
                rel_err = np.abs(agent_sorted["median_fold_change"] - ref_sorted["median_fold_change"]) / ref_sorted["median_fold_change"].replace(0, np.nan)
                if not (rel_err.fillna(0) <= 0.05).all():
                    ok = False
                    msg.append("median_fold_change relative error > 0.05")
    constraints["mechanism_match"] = ok
    details["mechanism_match"] = msg

    ok = True
    msg = []
    if agent_report is None or not isinstance(agent_report, dict):
        ok = False
        msg.append("report.json missing or invalid")
    else:
        required_keys = ["hsa_synergy_cell_lines", "max_hsa_score", "ic50_nM", "mechanism"]
        missing_keys = [k for k in required_keys if k not in agent_report]
        if missing_keys:
            ok = False
            msg.append(f"missing keys: {missing_keys}")
        else:
            if set(agent_report["hsa_synergy_cell_lines"]) != set(ref_report["hsa_synergy_cell_lines"]):
                ok = False
                msg.append("hsa_synergy_cell_lines mismatch")
            for cl in ref_report["max_hsa_score"]:
                if cl not in agent_report["max_hsa_score"]:
                    ok = False
                    msg.append(f"max_hsa_score missing {cl}")
                    break
                if abs(agent_report["max_hsa_score"][cl] - ref_report["max_hsa_score"][cl]) > 0.01:
                    ok = False
                    msg.append(f"max_hsa_score mismatch for {cl}")
                    break
            if ok:
                for cl in ref_report["ic50_nM"]:
                    if cl not in agent_report["ic50_nM"]:
                        ok = False
                        msg.append(f"ic50_nM missing {cl}")
                        break
                    for drug in ref_report["ic50_nM"][cl]:
                        if drug not in agent_report["ic50_nM"][cl]:
                            ok = False
                            msg.append(f"ic50_nM missing {cl}/{drug}")
                            break
                        ref_val = ref_report["ic50_nM"][cl][drug]
                        agent_val = agent_report["ic50_nM"][cl][drug]
                        if abs(agent_val - ref_val) / ref_val > 0.15:
                            ok = False
                            msg.append(f"ic50_nM mismatch {cl}/{drug}")
                            break
                    if not ok:
                        break
            if ok:
                for cl in ref_report["mechanism"]:
                    if cl not in agent_report["mechanism"]:
                        ok = False
                        msg.append(f"mechanism missing {cl}")
                        break
                    for flag in ref_report["mechanism"][cl]:
                        if agent_report["mechanism"][cl].get(flag) != ref_report["mechanism"][cl][flag]:
                            ok = False
                            msg.append(f"mechanism flag mismatch {cl}/{flag}")
                            break
                    if not ok:
                        break
    constraints["report_json_match"] = ok
    details["report_json_match"] = msg

    ok = False
    msg = []
    if agent_hsa is not None and "call" in agent_hsa.columns and "cell_line" in agent_hsa.columns:
        synergy_counts = agent_hsa[agent_hsa["call"] == "synergistic"].groupby("cell_line").size()
        cell_lines_with_synergy = set(synergy_counts[synergy_counts >= 3].index)
        if len(cell_lines_with_synergy) >= 1 and len(cell_lines_with_synergy) < 4:
            ok = True
        else:
            msg.append(f"cell lines with >=3 synergy: {sorted(cell_lines_with_synergy)}")
    else:
        msg.append("hsa_scores.csv missing or malformed")
    constraints["cell_line_specificity"] = ok
    details["cell_line_specificity"] = msg

    ok = True
    msg = []
    if agent_hsa is None or agent_report is None:
        ok = False
        msg.append("missing hsa_scores.csv or report.json")
    else:
        max_scores = agent_hsa.groupby("cell_line")["hsa_score"].max().to_dict()
        for cl, val in max_scores.items():
            if cl not in agent_report.get("max_hsa_score", {}):
                ok = False
                msg.append(f"report missing max_hsa_score for {cl}")
                break
            if abs(agent_report["max_hsa_score"][cl] - val) > 1e-6:
                ok = False
                msg.append(f"max_hsa_score inconsistency for {cl}")
                break
    constraints["internal_consistency"] = ok
    details["internal_consistency"] = msg

    passed = sum(1 for v in constraints.values() if v)
    total = len(constraints)
    score = passed / total if total > 0 else 0.0

    reward = {"score": float(score), "reward": float(score)}
    with open(os.path.join(LOG_DIR, "reward.json"), "w") as f:
        json.dump(reward, f)
    report_out = {
        "score": float(score),
        "constraints": {k: bool(v) for k, v in constraints.items()},
        "details": details,
        "passed": passed,
        "total": total
    }
    with open(os.path.join(LOG_DIR, "report.json"), "w") as f:
        json.dump(report_out, f, indent=2)
    print(json.dumps(report_out, indent=2))

if __name__ == "__main__":
    main()
