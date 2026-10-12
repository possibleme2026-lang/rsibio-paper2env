#!/bin/bash
set -euo pipefail
mkdir -p /app/output
cat > /app/output/analysis.py <<'PY'
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/app/data")
    parser.add_argument("--output_dir", default="/app/output")
    args = parser.parse_args()
    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    subjects = pd.read_csv(indir / "subjects.csv")
    visits = pd.read_csv(indir / "visits.csv")
    fc = pd.read_csv(indir / "fc_longitudinal.csv")
    edges = pd.read_csv(indir / "edges.csv")
    
    df = fc.merge(visits[["subject_id", "visit", "group", "group_code"]], on=["subject_id", "visit"], how="left")
    results = []
    for edge in sorted(df["edge_id"].unique()):
        sub = df[df["edge_id"] == edge].copy()
        try:
            model = smf.mixedlm("fc_z ~ group_code * visit", sub, groups=sub["subject_id"])
            fit = model.fit(reml=False, method='lbfgs', maxiter=200)
            beta = fit.params.get("group_code:visit", np.nan)
            se = fit.bse.get("group_code:visit", np.nan)
            t = fit.tvalues.get("group_code:visit", np.nan)
            p = fit.pvalues.get("group_code:visit", np.nan)
        except Exception:
            beta = se = t = p = np.nan
        results.append({"edge_id": edge, "beta_group_visit": beta, "se": se, "t": t, "p_value": p})
    res_df = pd.DataFrame(results)
    res_df["p_fdr"] = multipletests(res_df["p_value"].fillna(1.0), method="fdr_bh")[1]
    res_df["significant"] = res_df["p_fdr"] < 0.05
    res_df = res_df.sort_values("p_value").reset_index(drop=True)
    res_df["network_i"] = res_df["edge_id"].apply(lambda x: x.split("-")[0])
    res_df["network_j"] = res_df["edge_id"].apply(lambda x: x.split("-")[1])
    res_df = res_df[["edge_id", "network_i", "network_j", "beta_group_visit", "se", "t", "p_value", "p_fdr", "significant"]]
    res_df.to_csv(outdir / "interaction_results.csv", index=False)
    
    baseline = visits[visits["visit"] == 0].copy()
    vis_dm = fc[(fc["edge_id"] == "Visual-DefaultMode") & (fc["visit"] == 0)][["subject_id", "fc_z"]].rename(columns={"fc_z": "vis_dm_fc"})
    baseline = baseline.merge(vis_dm, on="subject_id", how="left")
    baseline["dysconnectivity"] = -baseline["vis_dm_fc"]
    clin_results = []
    for score in ["MMSE", "SARA", "MoCA"]:
        r, p = pearsonr(baseline["dysconnectivity"], baseline[score])
        clin_results.append({"score": score, "edge_id": "Visual-DefaultMode", "r": r, "p_value": p, "n": len(baseline)})
    clin_df = pd.DataFrame(clin_results)
    clin_df.to_csv(outdir / "clinical_associations.csv", index=False)
    
    baseline_ids = visits[visits["visit"] == 0]["subject_id"].tolist()
    fc_base = fc[fc["visit"] == 0].pivot(index="subject_id", columns="edge_id", values="fc_z")
    subjects_base = visits[visits["visit"] == 0][["subject_id", "group_code"]].set_index("subject_id")
    X = fc_base.loc[subjects_base.index].values
    y = subjects_base["group_code"].values
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty='l2', C=0.1, solver='liblinear', random_state=0, max_iter=1000)
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    preds = np.zeros_like(y)
    probs = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in cv.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(X[test_idx])
        probs[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds).tolist()
    metrics = {"accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f1), "confusion_matrix": cm, "cv_folds": 5}
    with open(outdir / "classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    predictions = pd.DataFrame({
        "subject_id": subjects_base.index,
        "true_group": y,
        "predicted_group": preds,
        "probability_SCA7": probs
    })
    predictions.to_csv(outdir / "predictions.csv", index=False)
    
    pipe.fit(X, y)
    coefs = pipe.named_steps['logisticregression'].coef_[0]
    edge_names = fc_base.columns.tolist()
    top_features = pd.DataFrame({
        "edge_id": edge_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    top_features["rank"] = top_features.index + 1
    top_features.to_csv(outdir / "top_features.csv", index=False)

if __name__ == "__main__":
    main()
PY
python /app/output/analysis.py --input_dir /app/data --output_dir /app/output