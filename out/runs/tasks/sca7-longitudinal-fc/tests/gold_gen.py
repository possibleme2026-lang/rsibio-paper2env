import numpy as np
import pandas as pd
from pathlib import Path
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

SEED = 20260215
NETWORKS = ["Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Limbic", "Frontoparietal", "DefaultMode", "Cerebellar"]

def generate_data(seed=SEED):
    rng = np.random.default_rng(seed)
    n_sca = 16
    n_ctrl = 16
    subjects = []
    for i in range(n_sca):
        subjects.append({
            "subject_id": f"SCA7_{i+1:02d}",
            "group": "SCA7",
            "age": float(rng.normal(45, 8)),
            "sex": str(rng.choice(["M", "F"]))
        })
    for i in range(n_ctrl):
        subjects.append({
            "subject_id": f"CTRL_{i+1:02d}",
            "group": "Control",
            "age": float(rng.normal(45, 8)),
            "sex": str(rng.choice(["M", "F"]))
        })
    subjects_df = pd.DataFrame(subjects)
    subjects_df["group_code"] = (subjects_df["group"] == "SCA7").astype(int)
    
    visits = []
    for _, sub in subjects_df.iterrows():
        for v in [0, 1, 2]:
            visits.append({
                "subject_id": sub["subject_id"],
                "visit": v,
                "months": v * 12,
                "age_at_visit": sub["age"] + v,
                "group": sub["group"],
                "group_code": sub["group_code"]
            })
    visits_df = pd.DataFrame(visits)
    
    edge_names = []
    for i in range(len(NETWORKS)):
        for j in range(i+1, len(NETWORKS)):
            edge_names.append(f"{NETWORKS[i]}-{NETWORKS[j]}")
    
    base_mean = rng.normal(0.30, 0.05, size=len(edge_names))
    base_mean[edge_names.index("Visual-DefaultMode")] = 0.38
    base_mean[edge_names.index("Somatomotor-Cerebellar")] = 0.42
    base_mean[edge_names.index("Visual-Somatomotor")] = 0.35
    base_mean[edge_names.index("Frontoparietal-DefaultMode")] = 0.33
    base_mean[edge_names.index("DefaultMode-Cerebellar")] = 0.30
    base_mean[edge_names.index("Visual-Frontoparietal")] = 0.28
    
    group_diff = np.zeros(len(edge_names))
    group_diff[edge_names.index("Visual-DefaultMode")] = -0.50
    group_diff[edge_names.index("Somatomotor-Cerebellar")] = -0.45
    group_diff[edge_names.index("Visual-Somatomotor")] = -0.35
    group_diff[edge_names.index("Frontoparietal-DefaultMode")] = -0.30
    group_diff[edge_names.index("DefaultMode-Cerebellar")] = -0.25
    group_diff[edge_names.index("Visual-Frontoparietal")] = -0.20
    
    visit_effect = rng.normal(0.0, 0.01, size=len(edge_names))
    interaction = np.zeros(len(edge_names))
    interaction[edge_names.index("Visual-DefaultMode")] = -0.06
    interaction[edge_names.index("Somatomotor-Cerebellar")] = -0.05
    interaction[edge_names.index("Visual-Somatomotor")] = -0.04
    interaction[edge_names.index("Frontoparietal-DefaultMode")] = -0.03
    
    subj_global = {sid: rng.normal(0, 0.10) for sid in subjects_df["subject_id"]}
    subj_edge = {}
    for sid in subjects_df["subject_id"]:
        subj_edge[sid] = rng.normal(0, 0.10, size=len(edge_names))
    
    fc_rows = []
    for _, sub in subjects_df.iterrows():
        sid = sub["subject_id"]
        g = sub["group_code"]
        for v in [0, 1, 2]:
            u_global = subj_global[sid]
            u_edge = subj_edge[sid]
            for e_idx, ename in enumerate(edge_names):
                mean = base_mean[e_idx] + group_diff[e_idx] * g + visit_effect[e_idx] * v + interaction[e_idx] * g * v
                val = mean + u_global + u_edge[e_idx] + rng.normal(0, 0.08)
                val = float(np.clip(val, -0.95, 0.95))
                fc_rows.append({
                    "subject_id": sid,
                    "visit": v,
                    "edge_id": ename,
                    "network_i": ename.split("-")[0],
                    "network_j": ename.split("-")[1],
                    "fc_z": val
                })
    fc_df = pd.DataFrame(fc_rows)
    
    vis_dm = fc_df[fc_df["edge_id"] == "Visual-DefaultMode"][["subject_id", "visit", "fc_z"]].rename(columns={"fc_z": "vis_dm_fc"})
    visits_df = visits_df.merge(vis_dm, on=["subject_id", "visit"], how="left")
    visits_df["dysconnectivity"] = -visits_df["vis_dm_fc"]
    
    mmse = []
    moca = []
    sara = []
    for _, row in visits_df.iterrows():
        g = row["group_code"]
        v = row["visit"]
        dys = row["dysconnectivity"]
        mmse_val = 29.0 - 3.0*g - 0.1*g*v - 4.0*dys + rng.normal(0, 1.0)
        mmse_val = float(np.clip(mmse_val, 15, 30))
        moca_val = 28.0 - 3.0*g - 0.05*g*v - 4.0*dys + rng.normal(0, 1.0)
        moca_val = float(np.clip(moca_val, 10, 30))
        sara_val = 0.5 + 7.5*g + 0.6*g*v + 6.0*dys + rng.normal(0, 1.5)
        sara_val = float(np.clip(sara_val, 0, 40))
        mmse.append(mmse_val)
        moca.append(moca_val)
        sara.append(sara_val)
    visits_df["MMSE"] = mmse
    visits_df["MoCA"] = moca
    visits_df["SARA"] = sara
    
    return subjects_df, visits_df, fc_df, edge_names

def run_interaction_analysis(fc_df, visits_df):
    df = fc_df.merge(visits_df[["subject_id", "visit", "group", "group_code"]], on=["subject_id", "visit"], how="left")
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
        results.append({
            "edge_id": edge,
            "beta_group_visit": beta,
            "se": se,
            "t": t,
            "p_value": p
        })
    res_df = pd.DataFrame(results)
    res_df["p_fdr"] = multipletests(res_df["p_value"].fillna(1.0), method="fdr_bh")[1]
    res_df["significant"] = res_df["p_fdr"] < 0.05
    res_df = res_df.sort_values("p_value").reset_index(drop=True)
    res_df["network_i"] = res_df["edge_id"].apply(lambda x: x.split("-")[0])
    res_df["network_j"] = res_df["edge_id"].apply(lambda x: x.split("-")[1])
    cols = ["edge_id", "network_i", "network_j", "beta_group_visit", "se", "t", "p_value", "p_fdr", "significant"]
    return res_df[cols]

def run_clinical_associations(fc_df, visits_df):
    baseline = visits_df[visits_df["visit"] == 0].copy()
    vis_dm = fc_df[(fc_df["edge_id"] == "Visual-DefaultMode") & (fc_df["visit"] == 0)][["subject_id", "fc_z"]].rename(columns={"fc_z": "vis_dm_fc"})
    baseline = baseline.merge(vis_dm, on="subject_id", how="left")
    baseline["dysconnectivity"] = -baseline["vis_dm_fc"]
    results = []
    for score in ["MMSE", "SARA", "MoCA"]:
        r, p = pearsonr(baseline["dysconnectivity"], baseline[score])
        results.append({
            "score": score,
            "edge_id": "Visual-DefaultMode",
            "r": r,
            "p_value": p,
            "n": len(baseline)
        })
    return pd.DataFrame(results)

def run_classification(fc_df, visits_df):
    baseline_ids = visits_df[visits_df["visit"] == 0]["subject_id"].tolist()
    fc_base = fc_df[fc_df["visit"] == 0].pivot(index="subject_id", columns="edge_id", values="fc_z")
    subjects = visits_df[visits_df["visit"] == 0][["subject_id", "group_code"]].set_index("subject_id")
    X = fc_base.loc[subjects.index].values
    y = subjects["group_code"].values
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
    pipe.fit(X, y)
    coefs = pipe.named_steps['logisticregression'].coef_[0]
    edge_names = fc_base.columns.tolist()
    top_features = pd.DataFrame({
        "edge_id": edge_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    top_features["rank"] = top_features.index + 1
    predictions = pd.DataFrame({
        "subject_id": subjects.index,
        "true_group": y,
        "predicted_group": preds,
        "probability_SCA7": probs
    })
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "cv_folds": 5
    }
    return metrics, predictions, top_features