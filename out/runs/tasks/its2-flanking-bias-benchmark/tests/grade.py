#!/usr/bin/env python3
import json, os, sys, tempfile
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/tests")
import gold_gen

OUT = "/app/output"
REPORT = "/logs/verifier/report.json"
REWARD = "/logs/verifier/reward.json"

def write_reward(score):
    os.makedirs("/logs/verifier", exist_ok=True)
    with open(REWARD, "w") as f:
        json.dump({"score": float(score), "reward": float(score)}, f)

def load_agent_metrics():
    path = f"{OUT}/metrics.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_predictions():
    path = f"{OUT}/predictions.tsv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t")

def load_graft():
    path = f"{OUT}/graft_predictions.tsv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t")

def check_schema(pred_df):
    if pred_df is None:
        return False
    required_cols = {"query_id", "method", "locus", "predicted_family", "confidence"}
    if not required_cols.issubset(pred_df.columns):
        return False
    if len(pred_df) != 1200 * 5 * 2:
        return False
    if pred_df["predicted_family"].isna().any():
        return False
    if pred_df["confidence"].isna().any():
        return False
    return True

def recompute_metrics(pred_df, graft_df, meta):
    metrics = {}
    seen = meta[meta.stratum == "seen"]
    for method in ["besthit", "sintax", "hitac", "cnn", "transformer"]:
        for locus in ["full", "its2"]:
            sub = pred_df[(pred_df.method == method) & (pred_df.locus == locus)]
            if sub.empty:
                return None
            merged = sub.merge(seen, on="query_id")
            acc = (merged.predicted_family == merged.family).mean()
            metrics[f"{method}_{locus}_seen_family_acc"] = float(acc)
    metrics["cnn_full_minus_its2_seen_family_acc"] = metrics["cnn_full_seen_family_acc"] - metrics["cnn_its2_seen_family_acc"]
    metrics["transformer_full_minus_its2_seen_family_acc"] = metrics["transformer_full_seen_family_acc"] - metrics["transformer_its2_seen_family_acc"]
    for model in ["cnn", "transformer"]:
        sub = graft_df[graft_df.model == model]
        if sub.empty:
            return None
        donor_rate = (sub.predicted_family == sub.donor_family).mean()
        query_rate = (sub.predicted_family == sub.query_family).mean()
        metrics[f"{model}_graft_donor_family_rate"] = float(donor_rate)
        metrics[f"{model}_graft_query_family_rate"] = float(query_rate)
    def auroc(method, locus):
        sub = pred_df[(pred_df.method == method) & (pred_df.locus == locus)]
        merged = sub.merge(meta, on="query_id")
        y = (merged.stratum == "novel").astype(int)
        if y.nunique() < 2:
            return 0.5
        return float(roc_auc_score(y, merged.confidence))
    metrics["cnn_its2_novel_vs_seen_auroc"] = auroc("cnn", "its2")
    metrics["transformer_its2_novel_vs_seen_auroc"] = auroc("transformer", "its2")
    metrics["besthit_its2_novel_vs_seen_auroc"] = auroc("besthit", "its2")
    return metrics

def check_its2_extraction(pristine_dir):
    path = f"{OUT}/query_its2.fasta"
    if not os.path.exists(path):
        return False
    primers = {}
    with open(f"{pristine_dir}/primers.tsv") as f:
        next(f)
        for line in f:
            name, seq = line.strip().split("\t")
            primers[name] = seq
    ITS3 = primers["ITS3"]
    ITS4 = primers["ITS4"]
    ITS4_rc = str(Seq(ITS4).reverse_complement())
    expected = {}
    for rec in SeqIO.parse(f"{pristine_dir}/queries.fasta", "fasta"):
        seq = str(rec.seq)
        start = seq.find(ITS3)
        if start == -1:
            return False
        end = seq.find(ITS4_rc, start)
        if end == -1:
            return False
        expected[rec.id] = seq[start+len(ITS3):end]
    got = {}
    for rec in SeqIO.parse(path, "fasta"):
        got[rec.id] = str(rec.seq)
    if set(expected.keys()) != set(got.keys()):
        return False
    for k in expected:
        if expected[k] != got[k]:
            return False
        if not (150 <= len(got[k]) <= 350):
            return False
    return True

def main():
    os.makedirs("/logs/verifier", exist_ok=True)
    pristine = tempfile.mkdtemp()
    gold_gen.main(pristine)
    meta = pd.read_csv(f"{pristine}/queries_meta.tsv", sep="\t")

    agent_metrics = load_agent_metrics()
    pred_df = load_predictions()
    graft_df = load_graft()

    if agent_metrics is None or pred_df is None or graft_df is None:
        write_reward(0.0)
        print("Missing required output files.")
        return

    constraints = []
    c1 = check_its2_extraction(pristine)
    constraints.append(("its2_extraction", c1))
    c2 = check_schema(pred_df)
    constraints.append(("prediction_schema", c2))

    recomputed = recompute_metrics(pred_df, graft_df, meta)
    if recomputed is None:
        write_reward(0.0)
        print("Failed to recompute metrics from predictions.")
        return

    consistency = True
    for k, v in recomputed.items():
        if k not in agent_metrics:
            consistency = False
            break
        if not isinstance(agent_metrics[k], (int, float)):
            consistency = False
            break
        if abs(agent_metrics[k] - v) > 1e-6:
            consistency = False
            break
    constraints.append(("metrics_consistency", consistency))

    c4 = recomputed.get("besthit_full_seen_family_acc", 0) >= 0.70
    constraints.append(("besthit_full_acc", c4))
    cnn_its2 = recomputed.get("cnn_its2_seen_family_acc", 1)
    cnn_full = recomputed.get("cnn_full_seen_family_acc", 0)
    c5 = cnn_its2 <= 0.40 and (cnn_full - cnn_its2) >= 0.30
    constraints.append(("cnn_its2_collapse", c5))
    trans_its2 = recomputed.get("transformer_its2_seen_family_acc", 1)
    trans_full = recomputed.get("transformer_full_seen_family_acc", 0)
    c6 = trans_its2 <= 0.40 and (trans_full - trans_its2) >= 0.30
    constraints.append(("transformer_its2_collapse", c6))
    c7 = recomputed.get("hitac_its2_seen_family_acc", 0) >= 0.70
    constraints.append(("hitac_its2_acc", c7))
    cnn_donor = recomputed.get("cnn_graft_donor_family_rate", 0)
    cnn_query = recomputed.get("cnn_graft_query_family_rate", 1)
    trans_donor = recomputed.get("transformer_graft_donor_family_rate", 0)
    trans_query = recomputed.get("transformer_graft_query_family_rate", 1)
    c8 = (cnn_donor >= 0.20 and (cnn_donor - cnn_query) >= 0.10 and
          trans_donor >= 0.20 and (trans_donor - trans_query) >= 0.10)
    constraints.append(("graft_flanking_effect", c8))
    c9 = recomputed.get("cnn_its2_novel_vs_seen_auroc", 1) <= 0.65
    constraints.append(("cnn_its2_auroc_chance", c9))
    c10 = recomputed.get("besthit_its2_novel_vs_seen_auroc", 0) >= 0.75
    constraints.append(("besthit_its2_auroc", c10))

    passed = sum(1 for _, ok in constraints if ok)
    total = len(constraints)
    score = passed / total
    write_reward(score)
    report = {name: bool(ok) for name, ok in constraints}
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()