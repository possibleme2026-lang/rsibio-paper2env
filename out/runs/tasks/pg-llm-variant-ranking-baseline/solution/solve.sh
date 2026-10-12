#!/bin/bash
set -euo pipefail

mkdir -p /app/output

cat > /tmp/solve_pgllm.py <<'PY'
import csv
import json
import math
import os
from collections import defaultdict

from Bio.Align import substitution_matrices
from scipy import stats

DATA = "/app/data"
OUT = "/app/output"
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

_mat = substitution_matrices.load("BLOSUM62")


def blosum(a, b):
    try:
        return int(_mat[a, b])
    except Exception:
        alphabet = _mat.alphabet
        return int(_mat[alphabet.index(a)][alphabet.index(b)])


def read_fasta(path):
    records, name, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(buf)))
                name, buf = line[1:], []
            else:
                buf.append(line)
    if name is not None:
        records.append((name, "".join(buf)))
    return records


manifest = list(csv.DictReader(open(os.path.join(DATA, "assay_manifest.csv"))))
rows_out = []
per_assay = {}

for entry in manifest:
    assay = entry["assay_id"]
    adir = os.path.join(DATA, "assays", assay)
    wt = read_fasta(os.path.join(adir, "wt.fasta"))[0][1]
    msa = [seq for _, seq in read_fasta(os.path.join(adir, "msa.fasta"))]
    length = len(wt)

    counts = [defaultdict(int) for _ in range(length)]
    for seq in msa:
        for p, ch in enumerate(seq):
            if ch in AA_SET:
                counts[p][ch] += 1

    freq = []
    for p in range(length):
        total = sum(counts[p].values())
        freq.append({a: (counts[p].get(a, 0) + 1.0) / (total + 20.0) for a in AA})

    variants = list(csv.DictReader(open(os.path.join(adir, "variants.csv"))))
    scores = {}
    for variant in variants:
        vid = variant["variant_id"]
        muts = variant["mutations"]
        if muts == "WT":
            scores[vid] = 0.0
        else:
            total = 0.0
            for single in muts.split(":"):
                w = single[0]
                p = int(single[1:-1]) - 1
                a = single[-1]
                total += math.log2(freq[p][a] / freq[p][w])
                total += 0.10 * (blosum(w, a) - blosum(w, w))
            scores[vid] = total
        rows_out.append((assay, vid, scores[vid]))

    pred_vec = [scores[v["variant_id"]] for v in variants]
    meas_vec = [float(v["dms_score"]) for v in variants]
    rho = float(stats.spearmanr(pred_vec, meas_vec)[0])
    if not math.isfinite(rho):
        rho = 0.0
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    per_assay[assay] = {
        "n_variants": len(variants),
        "spearman_rho": rho,
        "top5": [k for k, _ in order[:5]],
    }

with open(os.path.join(OUT, "predictions.csv"), "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["assay_id", "variant_id", "predicted_score"])
    for assay, vid, value in rows_out:
        writer.writerow([assay, vid, "%.10g" % value])

split_of = {entry["assay_id"]: entry["split"] for entry in manifest}
split_mean = {}
counts_split = {}
for split in ("proteinGym", "heldout"):
    members = [a for a in per_assay if split_of[a] == split]
    counts_split[split] = len(members)
    split_mean[split] = sum(per_assay[a]["spearman_rho"] for a in members) / len(members)

report = {
    "predictor": "MSA-CONS-BLOSUM",
    "per_assay": per_assay,
    "split_mean_rho": split_mean,
    "n_assays": counts_split,
}
with open(os.path.join(OUT, "report.json"), "w") as fh:
    json.dump(report, fh, indent=2)

print("wrote %d predictions across %d assays" % (len(rows_out), len(per_assay)))
PY

python /tmp/solve_pgllm.py
head -3 /app/output/predictions.csv
python -c "import json;print(json.load(open('/app/output/report.json'))['split_mean_rho'])"
