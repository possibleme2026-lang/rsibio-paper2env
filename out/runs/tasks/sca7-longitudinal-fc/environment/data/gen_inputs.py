import numpy as np
import pandas as pd
from pathlib import Path

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
    
    out_dir = Path(__file__).parent
    subjects_df.to_csv(out_dir / "subjects.csv", index=False)
    visits_df.to_csv(out_dir / "visits.csv", index=False)
    fc_df.to_csv(out_dir / "fc_longitudinal.csv", index=False)
    pd.DataFrame({"edge_id": edge_names}).to_csv(out_dir / "edges.csv", index=False)
    with open(out_dir / "README.txt", "w") as f:
        f.write("Synthetic longitudinal resting-state fMRI network-to-network functional connectivity dataset.\n")
        f.write("Networks: " + ", ".join(NETWORKS) + "\n")
        f.write("Edges: " + ", ".join(edge_names) + "\n")
        f.write("FC values are Fisher z-transformed. Clinical scores are MMSE, MoCA, SARA.\n")

if __name__ == "__main__":
    generate_data()