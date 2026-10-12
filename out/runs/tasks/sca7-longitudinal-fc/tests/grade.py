import json
import sys
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/tests")
import gold_gen

OUT = Path("/app/output")
GOLD_SUB, GOLD_VIS, GOLD_FC, GOLD_EDGES = gold_gen.generate_data()
GOLD_INTERACTION = gold_gen.run_interaction_analysis(GOLD_FC, GOLD_VIS)
GOLD_CLINICAL = gold_gen.run_clinical_associations(GOLD_FC, GOLD_VIS)
GOLD_METRICS, GOLD_PRED, GOLD_TOP = gold_gen.run_classification(GOLD_FC, GOLD_VIS)

constraints = []
def check(name, condition, weight=1.0):
    constraints.append({"name": name, "passed": bool(condition), "weight": weight})

required_files = ["interaction_results.csv", "clinical_associations.csv", "classification_metrics.json", "predictions.csv", "top_features.csv"]
check("files_exist", all((OUT / f).exists() for f in required_files))

inter = None
try:
    inter = pd.read_csv(OUT / "interaction_results.csv")
    expected_cols = ["edge_id", "network_i", "network_j", "beta_group_visit", "se", "t", "p_value", "p_fdr", "significant"]
    check("interaction_schema", all(c in inter.columns for c in expected_cols) and len(inter) == 28)
    check("interaction_sorted", inter["p_value"].is_monotonic_increasing)
    check("interaction_edge_set", set(inter["edge_id"]) == set(GOLD_INTERACTION["edge_id"]))
except Exception:
    check("interaction_schema", False)
    check("interaction_sorted", False)
    check("interaction_edge_set", False)

try:
    vis_dm = inter[inter["edge_id"] == "Visual-DefaultMode"].iloc[0]
    som_cb = inter[inter["edge_id"] == "Somatomotor-Cerebellar"].iloc[0]
    check("interaction_vis_dm_signif", vis_dm["p_fdr"] < 0.05 and vis_dm["beta_group_visit"] < 0)
    check("interaction_som_cb_signif", som_cb["p_fdr"] < 0.05 and som_cb["beta_group_visit"] < 0)
except Exception:
    check("interaction_vis_dm_signif", False)
    check("interaction_som_cb_signif", False)

try:
    merged = inter.merge(GOLD_INTERACTION[["edge_id", "beta_group_visit"]], on="edge_id", suffixes=("_agent", "_gold"))
    r, _ = pearsonr(merged["beta_group_visit_agent"], merged["beta_group_visit_gold"])
    check("interaction_beta_corr", r > 0.8)
except Exception:
    check("interaction_beta_corr", False)

clin = None
try:
    clin = pd.read_csv(OUT / "clinical_associations.csv")
    check("clinical_schema", all(c in clin.columns for c in ["score", "edge_id", "r", "p_value", "n"]) and len(clin) == 3)
    mmse = clin[clin["score"] == "MMSE"].iloc[0]
    moca = clin[clin["score"] == "MoCA"].iloc[0]
    sara = clin[clin["score"] == "SARA"].iloc[0]
    check("clinical_mmse_sign", mmse["r"] < 0 and mmse["p_value"] < 0.05)
    check("clinical_moca_sign", moca["r"] < 0 and moca["p_value"] < 0.05)
    check("clinical_sara_sign", sara["r"] > 0 and sara["p_value"] < 0.05)
    check("clinical_n", all(clin["n"] == 32))
except Exception:
    check("clinical_schema", False)
    check("clinical_mmse_sign", False)
    check("clinical_moca_sign", False)
    check("clinical_sara_sign", False)
    check("clinical_n", False)

try:
    merged = clin.merge(GOLD_CLINICAL[["score", "r"]], on="score", suffixes=("_agent", "_gold"))
    max_diff = (merged["r_agent"] - merged["r_gold"]).abs().max()
    check("clinical_r_tolerance", max_diff < 0.15)
except Exception:
    check("clinical_r_tolerance", False)

pred = None
try:
    pred = pd.read_csv(OUT / "predictions.csv")
    check("predictions_schema", all(c in pred.columns for c in ["subject_id", "true_group", "predicted_group", "probability_SCA7"]) and len(pred) == 32)
    check("predictions_prob_range", pred["probability_SCA7"].between(0, 1).all())
    gold_true = GOLD_PRED.set_index("subject_id")["true_group"]
    merged_true = pred.set_index("subject_id")["true_group"]
    check("predictions_true_labels", (merged_true == gold_true).all())
except Exception:
    check("predictions_schema", False)
    check("predictions_prob_range", False)
    check("predictions_true_labels", False)

try:
    y_true = pred["true_group"]
    y_pred = pred["predicted_group"]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    check("classification_accuracy", acc >= 0.80)
    check("classification_f1", f1 >= 0.80)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):
        recall0 = cm[0,0] / cm[0,:].sum() if cm[0,:].sum() > 0 else 0
        recall1 = cm[1,1] / cm[1,:].sum() if cm[1,:].sum() > 0 else 0
        check("classification_balanced_recall", recall0 >= 0.75 and recall1 >= 0.75)
    else:
        check("classification_balanced_recall", False)
except Exception:
    check("classification_accuracy", False)
    check("classification_f1", False)
    check("classification_balanced_recall", False)

try:
    metrics = json.loads((OUT / "classification_metrics.json").read_text())
    reported_acc = float(metrics.get("accuracy", -1))
    reported_f1 = float(metrics.get("f1", -1))
    check("metrics_match_predictions", abs(reported_acc - acc) < 0.02 and abs(reported_f1 - f1) < 0.02)
except Exception:
    check("metrics_match_predictions", False)

try:
    top = pd.read_csv(OUT / "top_features.csv")
    check("top_features_schema", all(c in top.columns for c in ["edge_id", "coefficient", "abs_coefficient", "rank"]) and len(top) == 28)
    top5 = set(top.nsmallest(5, "rank")["edge_id"])
    check("top_features_key_edges", "Visual-DefaultMode" in top5 and "Somatomotor-Cerebellar" in top5)
    gold_top10 = set(GOLD_TOP.nsmallest(10, "rank")["edge_id"])
    agent_top10 = set(top.nsmallest(10, "rank")["edge_id"])
    overlap = len(gold_top10 & agent_top10)
    check("top_features_overlap", overlap >= 6)
except Exception:
    check("top_features_schema", False)
    check("top_features_key_edges", False)
    check("top_features_overlap", False)

repro_passed = False
try:
    script = OUT / "analysis.py"
    if script.exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = ["python", str(script), "--input_dir", "/app/data", "--output_dir", tmpdir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                repro_inter = pd.read_csv(Path(tmpdir) / "interaction_results.csv")
                merged = repro_inter.merge(GOLD_INTERACTION[["edge_id", "beta_group_visit"]], on="edge_id", suffixes=("_agent", "_gold"))
                r, _ = pearsonr(merged["beta_group_visit_agent"], merged["beta_group_visit_gold"])
                repro_passed = r > 0.8 and len(repro_inter) == 28
    check("reproducibility", repro_passed)
except Exception:
    check("reproducibility", False)

try:
    vis_dm_gold = GOLD_INTERACTION[GOLD_INTERACTION["edge_id"] == "Visual-DefaultMode"].iloc[0]
    vis_dm_agent = inter[inter["edge_id"] == "Visual-DefaultMode"].iloc[0]
    check("interaction_p_tolerance", abs(vis_dm_agent["p_value"] - vis_dm_gold["p_value"]) < 0.02)
except Exception:
    check("interaction_p_tolerance", False)

total = sum(c["weight"] for c in constraints)
passed = sum(c["weight"] for c in constraints if c["passed"])
score = passed / total if total > 0 else 0.0
reward = {"score": score, "reward": score}
print(json.dumps(reward))
Path("/logs/verifier/reward.json").parent.mkdir(parents=True, exist_ok=True)
Path("/logs/verifier/reward.json").write_text(json.dumps(reward))