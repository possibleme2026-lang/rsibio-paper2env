import os, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from Bio import SeqIO
from Bio.Align import PairwiseAligner

DATA_DIR = "/app/data"
REPORT_PATH = "/app/results/report.json"
TOL = 0.01

def load_data():
    assays = {}
    for f in os.listdir(os.path.join(DATA_DIR, "assays")):
        if f.endswith(".csv"):
            assay_id = f.replace(".csv", "")
            assays[assay_id] = pd.read_csv(os.path.join(DATA_DIR, "assays", f))
    train_seqs = {}
    for rec in SeqIO.parse(os.path.join(DATA_DIR, "train_proteins.fasta"), "fasta"):
        train_seqs[rec.id] = str(rec.seq)
    sel_df = pd.read_csv(os.path.join(DATA_DIR, "selectivity.csv"))
    return assays, train_seqs, sel_df

def compute_identity(seq1, seq2):
    aligner = PairwiseAligner()
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -1
    alignments = aligner.align(seq1, seq2)
    best = alignments[0]
    matches = 0
    total = 0
    for (a, b) in zip(best.aligned[0], best.aligned[1]):
        for i in range(int(a[0]), int(a[1])):
            total += 1
            if seq1[i] == seq2[i]:
                matches += 1
    return matches / total if total > 0 else 0.0

def max_identity_to_train(seq, train_seqs):
    return max(compute_identity(seq, tseq) for tseq in train_seqs.values())

def compute_metrics(df):
    df_nesso = df.dropna(subset=["measured_pKd", "nesso_pred"])
    df_boltz = df.dropna(subset=["measured_pKd", "boltz_pred"])
    nesso_spearman = spearmanr(df_nesso["measured_pKd"], df_nesso["nesso_pred"])[0] if len(df_nesso) > 1 else 0.0
    boltz_spearman = spearmanr(df_boltz["measured_pKd"], df_boltz["boltz_pred"])[0] if len(df_boltz) > 1 else 0.0
    nesso_mae = mean_absolute_error(df_nesso["measured_pKd"], df_nesso["nesso_pred"]) if len(df_nesso) > 0 else 0.0
    boltz_mae = mean_absolute_error(df_boltz["measured_pKd"], df_boltz["boltz_pred"]) if len(df_boltz) > 0 else 0.0
    return nesso_spearman, boltz_spearman, nesso_mae, boltz_mae

def main():
    assays, train_seqs, sel_df = load_data()
    ref_per_assay = {}
    for assay_id, df in assays.items():
        nesso_s, boltz_s, nesso_mae, boltz_mae = compute_metrics(df)
        seq = df["protein_sequence"].iloc[0]
        max_id = max_identity_to_train(seq, train_seqs)
        is_ood = max_id < 0.40
        ref_per_assay[assay_id] = {
            "nesso_spearman": round(nesso_s, 4),
            "boltz_spearman": round(boltz_s, 4),
            "nesso_mae": round(nesso_mae, 4),
            "boltz_mae": round(boltz_mae, 4),
            "max_identity": round(max_id, 4),
            "is_ood": is_ood
        }
    id_assays = [a for a, v in ref_per_assay.items() if not v["is_ood"]]
    ood_assays = [a for a, v in ref_per_assay.items() if v["is_ood"]]
    mean_id_nesso = np.mean([ref_per_assay[a]["nesso_spearman"] for a in id_assays]) if id_assays else 0.0
    mean_id_boltz = np.mean([ref_per_assay[a]["boltz_spearman"] for a in id_assays]) if id_assays else 0.0
    mean_ood_nesso = np.mean([ref_per_assay[a]["nesso_spearman"] for a in ood_assays]) if ood_assays else 0.0
    mean_ood_boltz = np.mean([ref_per_assay[a]["boltz_spearman"] for a in ood_assays]) if ood_assays else 0.0
    nesso_outperforms = [a for a in ref_per_assay if ref_per_assay[a]["nesso_spearman"] - ref_per_assay[a]["boltz_spearman"] >= 0.1]
    nesso_failures = [a for a in ref_per_assay if ref_per_assay[a]["nesso_spearman"] < 0.3]
    ref_sel = {}
    for assay_id, group in sel_df.groupby("assay_id"):
        g = group.dropna(subset=["on_measured_pKd", "off_measured_pKd", "on_nesso_pred", "off_nesso_pred", "on_boltz_pred", "off_boltz_pred"])
        if len(g) < 2:
            ref_sel[assay_id] = {"nesso_spearman": 0.0, "boltz_spearman": 0.0}
            continue
        true_delta = g["on_measured_pKd"] - g["off_measured_pKd"]
        nesso_delta = g["on_nesso_pred"] - g["off_nesso_pred"]
        boltz_delta = g["on_boltz_pred"] - g["off_boltz_pred"]
        nesso_s = spearmanr(true_delta, nesso_delta)[0]
        boltz_s = spearmanr(true_delta, boltz_delta)[0]
        ref_sel[assay_id] = {"nesso_spearman": round(nesso_s, 4), "boltz_spearman": round(boltz_s, 4)}
    if not os.path.exists(REPORT_PATH):
        print(json.dumps({"score": 0.0, "reward": 0.0}))
        return
    with open(REPORT_PATH) as f:
        agent = json.load(f)
    checks = []
    if "per_assay" not in agent:
        checks.append(("per_assay_present", False))
    else:
        checks.append(("per_assay_present", True))
        for assay_id, ref in ref_per_assay.items():
            if assay_id not in agent["per_assay"]:
                checks.append((f"per_assay_{assay_id}_present", False))
                continue
            ag = agent["per_assay"][assay_id]
            for key in ["nesso_spearman", "boltz_spearman", "nesso_mae", "boltz_mae", "max_identity"]:
                if key not in ag:
                    checks.append((f"per_assay_{assay_id}_{key}_present", False))
                else:
                    checks.append((f"per_assay_{assay_id}_{key}", abs(ag[key] - ref[key]) <= TOL))
            if "is_ood" not in ag:
                checks.append((f"per_assay_{assay_id}_is_ood_present", False))
            else:
                checks.append((f"per_assay_{assay_id}_is_ood", ag["is_ood"] == ref["is_ood"]))
    if "aggregates" not in agent:
        checks.append(("aggregates_present", False))
    else:
        agg_checks = [
            ("mean_spearman_id_nesso", mean_id_nesso),
            ("mean_spearman_id_boltz", mean_id_boltz),
            ("mean_spearman_ood_nesso", mean_ood_nesso),
            ("mean_spearman_ood_boltz", mean_ood_boltz),
        ]
        for key, ref_val in agg_checks:
            if key not in agent["aggregates"]:
                checks.append((f"aggregates_{key}_present", False))
            else:
                checks.append((f"aggregates_{key}", abs(agent["aggregates"][key] - ref_val) <= TOL))
    if "nesso_outperforms" not in agent:
        checks.append(("nesso_outperforms_present", False))
    else:
        checks.append(("nesso_outperforms", set(agent["nesso_outperforms"]) == set(nesso_outperforms)))
    if "nesso_failures" not in agent:
        checks.append(("nesso_failures_present", False))
    else:
        checks.append(("nesso_failures", set(agent["nesso_failures"]) == set(nesso_failures)))
    if "selectivity" not in agent:
        checks.append(("selectivity_present", False))
    else:
        for assay_id, ref in ref_sel.items():
            if assay_id not in agent["selectivity"]:
                checks.append((f"selectivity_{assay_id}_present", False))
                continue
            ag = agent["selectivity"][assay_id]
            for key in ["nesso_spearman", "boltz_spearman"]:
                if key not in ag:
                    checks.append((f"selectivity_{assay_id}_{key}_present", False))
                else:
                    checks.append((f"selectivity_{assay_id}_{key}", abs(ag[key] - ref[key]) <= TOL))
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    score = passed / total if total > 0 else 0.0
    print(json.dumps({"score": score, "reward": score}))

if __name__ == "__main__":
    main()
