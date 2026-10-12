#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python3 - <<'PY'
import os, subprocess, json, math, itertools
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

DATA = "/app/data"
OUT = "/app/output"

primers = {}
with open(f"{DATA}/primers.tsv") as f:
    next(f)
    for line in f:
        name, seq = line.strip().split("\t")
        primers[name] = seq
ITS3 = primers["ITS3"]
ITS4 = primers["ITS4"]
ITS4_rc = str(Seq(ITS4).reverse_complement())

def extract_its2(seq):
    start = seq.find(ITS3)
    if start == -1:
        return None
    end = seq.find(ITS4_rc, start)
    if end == -1:
        return None
    return seq[start+len(ITS3):end]

query_recs = list(SeqIO.parse(f"{DATA}/queries.fasta", "fasta"))
ref_recs = list(SeqIO.parse(f"{DATA}/reference.fasta", "fasta"))
ref_dict = {r.id: r for r in ref_recs}

query_its2 = []
for rec in query_recs:
    its2 = extract_its2(str(rec.seq))
    if its2 is None:
        raise ValueError(f"No ITS2 for {rec.id}")
    query_its2.append(SeqRecord(Seq(its2), id=rec.id, description=""))
ref_its2 = []
for rec in ref_recs:
    its2 = extract_its2(str(rec.seq))
    if its2 is None:
        raise ValueError(f"No ITS2 for {rec.id}")
    ref_its2.append(SeqRecord(Seq(its2), id=rec.id, description=""))

SeqIO.write(query_its2, f"{OUT}/query_its2.fasta", "fasta")
SeqIO.write(ref_its2, f"{OUT}/reference_its2.fasta", "fasta")

query_meta = pd.read_csv(f"{DATA}/queries_meta.tsv", sep="\t")
ref_tax = pd.read_csv(f"{DATA}/reference_taxonomy.tsv", sep="\t")
ref_tax_map = dict(zip(ref_tax.id, ref_tax.family))
genus_to_family = dict(zip(ref_tax.genus, ref_tax.family))

def run_vsearch_global(query_fa, db_fa, out_tsv):
    subprocess.run([
        "vsearch", "--usearch_global", query_fa,
        "--db", db_fa,
        "--id", "0.8",
        "--maxaccepts", "1",
        "--blast6out", out_tsv,
        "--threads", "2"
    ], check=True)
    hits = {}
    with open(out_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            hits[parts[0]] = (parts[1], float(parts[2]))
    return hits

besthit_full = run_vsearch_global(f"{DATA}/queries.fasta", f"{DATA}/reference.fasta", f"{OUT}/besthit_full.tsv")
besthit_its2 = run_vsearch_global(f"{OUT}/query_its2.fasta", f"{OUT}/reference_its2.fasta", f"{OUT}/besthit_its2.tsv")

def run_sintax(query_fa, db_fa, out_tsv):
    subprocess.run([
        "vsearch", "--sintax", query_fa,
        "--db", db_fa,
        "--tabbedout", out_tsv,
        "--sintax_cutoff", "0.8",
        "--threads", "2"
    ], check=True)
    preds = {}
    with open(out_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: continue
            fam = "unassigned"
            for token in parts[1].split(","):
                if token.startswith("f:"):
                    fam = token[2:]
            preds[parts[0]] = fam
    return preds

sintax_full = run_sintax(f"{DATA}/queries.fasta", f"{DATA}/reference.fasta", f"{OUT}/sintax_full.tsv")
sintax_its2 = run_sintax(f"{OUT}/query_its2.fasta", f"{OUT}/reference_its2.fasta", f"{OUT}/sintax_its2.tsv")

def kmer_vec(seq, k=4):
    kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
    kmap = {kmer: i for i, kmer in enumerate(kmers)}
    vec = np.zeros(len(kmers))
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        if kmer in kmap:
            vec[kmap[kmer]] += 1
    return vec

def train_hitac(ref_fasta, ref_tax_df):
    recs = list(SeqIO.parse(ref_fasta, "fasta"))
    X = np.array([kmer_vec(str(r.seq)) for r in recs])
    id_to_genus = dict(zip(ref_tax_df.id, ref_tax_df.genus))
    y = np.array([id_to_genus[r.id] for r in recs])
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, n_jobs=1)
    clf.fit(X, y_enc)
    return clf, le

clf_full, le_full = train_hitac(f"{DATA}/reference.fasta", ref_tax)
clf_its2, le_its2 = train_hitac(f"{OUT}/reference_its2.fasta", ref_tax)

def predict_hitac(clf, le, fasta_path):
    preds = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        vec = kmer_vec(str(rec.seq)).reshape(1, -1)
        proba = clf.predict_proba(vec)[0]
        idx = int(np.argmax(proba))
        genus = le.inverse_transform([idx])[0]
        fam = genus_to_family.get(genus, "unassigned")
        conf = float(proba[idx])
        preds[rec.id] = (fam, conf)
    return preds

hitac_full_pred = predict_hitac(clf_full, le_full, f"{DATA}/queries.fasta")
hitac_its2_pred = predict_hitac(clf_its2, le_its2, f"{OUT}/query_its2.fasta")

def run_mycoai(model, fasta_in, out_tsv):
    subprocess.run([
        "python", f"{DATA}/models/predict_mycoai.py",
        "--model", model,
        "--fasta", fasta_in,
        "--out", out_tsv
    ], check=True)
    preds = {}
    with open(out_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            preds[parts[0]] = (parts[1], float(parts[2]))
    return preds

cnn_full = run_mycoai("cnn", f"{DATA}/queries.fasta", f"{OUT}/cnn_full.tsv")
cnn_its2 = run_mycoai("cnn", f"{OUT}/query_its2.fasta", f"{OUT}/cnn_its2.tsv")
trans_full = run_mycoai("transformer", f"{DATA}/queries.fasta", f"{OUT}/transformer_full.tsv")
trans_its2 = run_mycoai("transformer", f"{OUT}/query_its2.fasta", f"{OUT}/transformer_its2.tsv")

query_phylum = dict(zip(query_meta.query_id, query_meta.phylum))
d onor_map = {}
for qid, qphy in query_phylum.items():
    for _, rrow in ref_tax.iterrows():
        if rrow.phylum != qphy:
            donor_map[qid] = rrow.id
            break

grafted = []
for qrec in query_recs:
    qid = qrec.id
    its2 = extract_its2(str(qrec.seq))
    donor_id = donor_map[qid]
    donor_seq = str(ref_dict[donor_id].seq)
    start = donor_seq.find(ITS3) + len(ITS3)
    end = donor_seq.find(ITS4_rc, start)
    grafted_seq = donor_seq[:start] + its2 + donor_seq[end:]
    grafted.append(SeqRecord(Seq(grafted_seq), id=qid, description=""))
SeqIO.write(grafted, f"{OUT}/grafted.fasta", "fasta")

cnn_graft = run_mycoai("cnn", f"{OUT}/grafted.fasta", f"{OUT}/cnn_graft.tsv")
trans_graft = run_mycoai("transformer", f"{OUT}/grafted.fasta", f"{OUT}/transformer_graft.tsv")

predictions = []
for qid in query_meta.query_id:
    if qid in besthit_full:
        target, ident = besthit_full[qid]
        fam = ref_tax_map.get(target, "unassigned")
    else:
        fam, ident = "unassigned", 0.0
    predictions.append({"query_id": qid, "method": "besthit", "locus": "full", "predicted_family": fam, "confidence": ident})
    if qid in besthit_its2:
        target, ident = besthit_its2[qid]
        fam = ref_tax_map.get(target, "unassigned")
    else:
        fam, ident = "unassigned", 0.0
    predictions.append({"query_id": qid, "method": "besthit", "locus": "its2", "predicted_family": fam, "confidence": ident})
    fam = sintax_full.get(qid, "unassigned")
    predictions.append({"query_id": qid, "method": "sintax", "locus": "full", "predicted_family": fam, "confidence": 1.0 if fam != "unassigned" else 0.0})
    fam = sintax_its2.get(qid, "unassigned")
    predictions.append({"query_id": qid, "method": "sintax", "locus": "its2", "predicted_family": fam, "confidence": 1.0 if fam != "unassigned" else 0.0})
    fam, conf = hitac_full_pred.get(qid, ("unassigned", 0.0))
    predictions.append({"query_id": qid, "method": "hitac", "locus": "full", "predicted_family": fam, "confidence": conf})
    fam, conf = hitac_its2_pred.get(qid, ("unassigned", 0.0))
    predictions.append({"query_id": qid, "method": "hitac", "locus": "its2", "predicted_family": fam, "confidence": conf})
    fam, conf = cnn_full.get(qid, ("unassigned", 0.0))
    predictions.append({"query_id": qid, "method": "cnn", "locus": "full", "predicted_family": fam, "confidence": conf})
    fam, conf = cnn_its2.get(qid, ("unassigned", 0.0))
    predictions.append({"query_id": qid, "method": "cnn", "locus": "its2", "predicted_family": fam, "confidence": conf})
    fam, conf = trans_full.get(qid, ("unassigned", 0.0))
    predictions.append({"query_id": qid, "method": "transformer", "locus": "full", "predicted_family": fam, "confidence": conf})
    fam, conf = trans_its2.get(qid, ("unassigned", 0.0))
    predictions.append({"query_id": qid, "method": "transformer", "locus": "its2", "predicted_family": fam, "confidence": conf})

pred_df = pd.DataFrame(predictions)
pred_df.to_csv(f"{OUT}/predictions.tsv", sep="\t", index=False)

graft_rows = []
for qid in query_meta.query_id:
    donor_id = donor_map[qid]
    donor_fam = ref_tax_map[donor_id]
    query_fam = query_meta.set_index("query_id").loc[qid, "family"]
    for model, preds in [("cnn", cnn_graft), ("transformer", trans_graft)]:
        fam, conf = preds.get(qid, ("unassigned", 0.0))
        graft_rows.append({"query_id": qid, "model": model, "predicted_family": fam, "donor_family": donor_fam, "query_family": query_fam, "confidence": conf})
graft_df = pd.DataFrame(graft_rows)
graft_df.to_csv(f"{OUT}/graft_predictions.tsv", sep="\t", index=False)

metrics = {}
seen_meta = query_meta[query_meta.stratum == "seen"]
for method in ["besthit", "sintax", "hitac", "cnn", "transformer"]:
    for locus in ["full", "its2"]:
        sub = pred_df[(pred_df.method == method) & (pred_df.locus == locus)]
        merged = sub.merge(seen_meta, on="query_id")
        acc = (merged.predicted_family == merged.family).mean()
        metrics[f"{method}_{locus}_seen_family_acc"] = float(acc)

metrics["cnn_full_minus_its2_seen_family_acc"] = metrics["cnn_full_seen_family_acc"] - metrics["cnn_its2_seen_family_acc"]
metrics["transformer_full_minus_its2_seen_family_acc"] = metrics["transformer_full_seen_family_acc"] - metrics["transformer_its2_seen_family_acc"]

for model in ["cnn", "transformer"]:
    sub = graft_df[graft_df.model == model]
    donor_rate = (sub.predicted_family == sub.donor_family).mean()
    query_rate = (sub.predicted_family == sub.query_family).mean()
    metrics[f"{model}_graft_donor_family_rate"] = float(donor_rate)
    metrics[f"{model}_graft_query_family_rate"] = float(query_rate)

def auroc(method, locus):
    sub = pred_df[(pred_df.method == method) & (pred_df.locus == locus)]
    merged = sub.merge(query_meta, on="query_id")
    y = (merged.stratum == "novel").astype(int)
    if y.nunique() < 2:
        return 0.5
    return float(roc_auc_score(y, merged.confidence))

metrics["cnn_its2_novel_vs_seen_auroc"] = auroc("cnn", "its2")
metrics["transformer_its2_novel_vs_seen_auroc"] = auroc("transformer", "its2")
metrics["besthit_its2_novel_vs_seen_auroc"] = auroc("besthit", "its2")

with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Done")
PY