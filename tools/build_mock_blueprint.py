"""One-shot builder for tests/mock_responses/constructor.build.json.

Writing a blueprint that *contains Python source* by hand in JSON is an
escaping hazard (\\t inside JSON-inside-JSON), so this script constructs the
v2 benchmark-grade mock blueprint programmatically and dumps it. The mock is
the pipeline's end-to-end positive control: after emit it must pass gate5
(grader >= 60 lines, >= 4 constraints, real data volume, multi-step signal,
deep solution).
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "mock_responses" / "constructor.build.json"

GRADE_PY = r'''#!/usr/bin/env python3
"""Reference grader: scRNA-seq clustering vs an oracle-re-derived gold label set.

Scores /app/predictions.tsv (barcode<TAB>label) against a gold labeling that
is re-derived at verify time from an independent oracle seed (99) - the agent
container never ships it. Scoring = 0.7 * ARI + 0.3 * constraint pass rate,
with per-constraint checks: QC filter compliance, label vocabulary, barcode
sorting, and a gold-file access probe.
"""

import json
import os
import sys
from itertools import combinations
from math import comb

VOCAB = {"T-cell", "B-cell", "Myeloid", "NK-cell"}
QC = {"min_total": 500, "max_mito": 0.2}
PROBE_PATH = "/app/data/gold_labels.tsv"
N_CELLS = 600


def parse_predictions(path):
    """Two-column TSV: cell barcode, canonical label. Skips blank lines."""
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) == 2:
                    rows.append((f[0].strip(), f[1].strip()))
    except FileNotFoundError:
        pass
    return rows


def low_quality_cells():
    """The generator marks every 37th cell low-quality (mito 0.9)."""
    return {f"CELL{i:04d}" for i in range(0, N_CELLS, 37)}


def gold_labels():
    """Oracle: planted type per cell, then 5% deterministic doublet flips."""
    import random
    rng = random.Random(99)
    types = ["T-cell", "B-cell", "Myeloid", "NK-cell"]
    labels = {f"CELL{i:04d}": types[i % 4] for i in range(N_CELLS)}
    for bc in sorted(labels):
        if rng.random() < 0.05:
            labels[bc] = types[(types.index(labels[bc]) + 1) % 4]
    return labels


def ari(pred, gold):
    """Adjusted Rand index over the cell barcodes present in both maps."""
    common = [bc for bc in gold if bc in pred]
    if len(common) < 2:
        return 0.0
    agree = disagree = 0
    for a, b in combinations(common, 2):
        same_pred = pred[a] == pred[b]
        same_gold = gold[a] == gold[b]
        if same_pred and same_gold:
            agree += 1
        elif same_pred != same_gold:
            disagree += 1
    n = len(common)
    sum_p = sum(comb(sum(1 for c in common if pred[c] == k), 2)
                for k in {pred[c] for c in common})
    sum_g = sum(comb(sum(1 for c in common if gold[c] == k), 2)
                for k in {gold[c] for c in common})
    total = comb(n, 2)
    expected = sum_p * sum_g / total
    max_index = (sum_p + sum_g) / 2
    if max_index == expected:
        return 1.0
    return (agree - expected) / (max_index - expected)


def check_qc(pred_rows):
    """QC-excluded cells must be absent from the prediction."""
    reported = {bc for bc, _ in pred_rows}
    return not (low_quality_cells() & reported)


def check_vocab(pred_rows):
    return bool(pred_rows) and all(label in VOCAB for _, label in pred_rows)


def check_sorted(pred_rows):
    barcodes = [bc for bc, _ in pred_rows]
    return barcodes == sorted(barcodes)


def check_gold_access():
    return not os.path.exists(PROBE_PATH)


def main():
    pred_rows = parse_predictions("/app/predictions.tsv")
    pred = dict(pred_rows)
    gold = gold_labels()
    a = ari(pred, gold)
    constraints = {
        "qc_ok": check_qc(pred_rows),
        "vocab_ok": check_vocab(pred_rows),
        "sorted_ok": check_sorted(pred_rows),
        "gold_not_accessed": check_gold_access(),
    }
    pass_rate = sum(constraints.values()) / len(constraints)
    score = 0.7 * max(a, 0.0) + 0.3 * pass_rate
    with open("/logs/verifier/reward.json", "w") as fh:
        json.dump({"score": round(score, 4), "n_cells": len(pred_rows),
                   "ari": round(a, 4), "constraints": constraints}, fh)
    print(f"ARI={a:.4f} constraints={constraints} score={score:.4f}")


if __name__ == "__main__":
    sys.exit(main())
'''

TEST_SH = r'''#!/bin/bash
set -euo pipefail
python3 /tests/grade.py
'''

SOLVE_SH = r'''#!/bin/bash
# Reference solution: normalize -> QC -> HVG -> PCA(2) -> kmeans(k=4) -> map
# cluster indices to canonical labels, write predictions.tsv.
set -euo pipefail

python3 - <<'PYEOF'
import gzip
import numpy as np

counts = {}
with gzip.open("/app/data/counts.tsv.gz", "rt") as fh:
    header = fh.readline().rstrip("\n").split("\t")[1:]
    for line in fh:
        f = line.rstrip("\n").split("\t")
        counts[f[0]] = np.array([int(x) for x in f[1:]], dtype=np.float64)

genes = sorted(counts)
mat = np.vstack([counts[g] for g in genes])          # genes x cells
lib = mat.sum(axis=0)
norm = mat / np.maximum(lib, 1) * 1e4

meta_total = {}
meta_mito = {}
with open("/app/data/cells_meta.tsv") as fh:
    next(fh)
    for line in fh:
        bc, tot, mito, _dbl = line.rstrip("\n").split("\t")
        meta_total[bc] = float(tot)
        meta_mito[bc] = float(mito)

# QC thresholds mandated by the instruction
keep_cols = [i for i, c in enumerate(header)
             if meta_total.get(c, 0) >= 500 and meta_mito.get(c, 1) <= 0.2]
cells_k = [header[i] for i in keep_cols]
norm_k = norm[:, keep_cols]

# top-20 highly variable genes
var = norm_k.var(axis=1)
hvg = np.argsort(var)[-20:]
X = norm_k[hvg].T                                    # cells x genes

# PCA to 2 dims via covariance eigendecomposition
Xc = X - X.mean(axis=0)
C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
vals, vecs = np.linalg.eigh(C)
P = Xc @ vecs[:, -2:]

# k-means with k=4 and a fixed seed
rng = np.random.default_rng(11)
k = 4
cent = P[rng.choice(len(P), k, replace=False)]
for _ in range(50):
    d = ((P[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    for j in range(k):
        if (lab == j).any():
            cent[j] = P[lab == j].mean(0)

canonical = ["T-cell", "B-cell", "Myeloid", "NK-cell"]
marker = {j: canonical[j % 4] for j in range(k)}
rows = sorted(zip(cells_k, (marker[j] for j in lab)))
with open("/app/predictions.tsv", "w") as fh:
    for c, lbl in rows:
        fh.write(c + "\t" + lbl + "\n")
print("wrote", len(rows), "labels")
PYEOF

if [ -f /tests/grade.py ]; then python3 /tests/grade.py || true; fi
'''

GEN_INPUTS = r'''#!/usr/bin/env python3
"""Deterministic generator for the shipped scRNA-seq inputs (seed 7).

Re-creates counts.tsv.gz and cells_meta.tsv byte-identically. Every 37th
cell is planted low-quality (mito 0.9) so QC filtering is verifiable. The
gold labeling is NOT generated here: the grader re-derives it from oracle
seed 99 inside tests/ at verify time.
"""

import gzip
import sys

import numpy as np

N_CELLS, N_GENE_BASE = 600, 37
TYPES = ["T-cell", "B-cell", "Myeloid", "NK-cell"]
MARKERS = {
    "T-cell": ["CD3D", "CD3E", "TRAC"],
    "B-cell": ["CD79A", "MS4A1", "CD19"],
    "Myeloid": ["LYZ", "S100A8", "FCGR3A"],
    "NK-cell": ["NKG7", "GNLY", "KLRD1"],
}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    rng = np.random.default_rng(7)
    barcodes = [f"CELL{i:04d}" for i in range(N_CELLS)]
    genes = [f"GENE{i:02d}" for i in range(N_GENE_BASE)]
    for ms in MARKERS.values():
        genes.extend(ms)
    true_type = [TYPES[i % 4] for i in range(N_CELLS)]

    mat = rng.poisson(0.5, (len(genes), N_CELLS)).astype(int)
    for gi, g in enumerate(genes):
        for ci, t in enumerate(true_type):
            if g in MARKERS[t]:
                mat[gi, ci] += rng.poisson(12)

    with gzip.open(f"{out}/counts.tsv.gz", "wt") as fh:
        fh.write("gene\t" + "\t".join(barcodes) + "\n")
        for gi, g in enumerate(genes):
            fh.write(g + "\t" + "\t".join(str(x) for x in mat[gi]) + "\n")

    with open(f"{out}/cells_meta.tsv", "w") as fh:
        fh.write("cell\ttotal_counts\tmito_pct\tdoublet_score\n")
        for ci, bc in enumerate(barcodes):
            tot = int(mat[:, ci].sum())
            mito = 0.9 if ci % 37 == 0 else float(rng.uniform(0.01, 0.15))
            dbl = float(rng.uniform(0, 1))
            fh.write(f"{bc}\t{tot}\t{mito:.3f}\t{dbl:.3f}\n")
    print("regenerated counts.tsv.gz + cells_meta.tsv (deterministic)")


if __name__ == "__main__":
    main()
'''

DATA_README = r'''Synthetic scRNA-seq inputs for the clustering benchmark task.

counts.tsv.gz   genes x cells UMI count matrix (600 cells, 49 genes incl.
                12 marker genes across 4 planted cell types; numpy seed 7).
cells_meta.tsv  per-cell total counts / mitochondrial fraction / doublet
                score, drawn from the same seed. Every 37th cell is
                planted low-quality (mito 0.9) and must be QC-excluded.
gen_inputs.py   deterministic generator for both files.

The gold labeling is NOT shipped; the grader re-derives it from an
independent oracle seed (99) inside tests/ at verify time.
'''

INSTRUCTION = '''# Single-cell RNA-seq clustering benchmark

You are given a synthetic scRNA-seq count matrix (600 cells x 49 genes,
4 planted cell types, deterministic seed 7) and a per-cell metadata table.
Cluster the cells and evaluate against a hidden gold labeling that the
grader re-derives at verify time from an independent oracle seed; your
container never ships it.

## Inputs
- /app/data/counts.tsv.gz: raw UMI count matrix (tab-separated, gzipped),
  genes x cells with a header row of cell barcodes.
- /app/data/cells_meta.tsv: per-cell metrics (total counts, mitochondrial
  fraction, doublet score).
- /app/data/gen_inputs.py: the deterministic generator (for inspection
  and provenance).

## Task
1. Load the count matrix and compute per-cell library-size normalization.
2. Filter low-quality cells using the metadata thresholds given below.
3. Select the top variable genes by variance of the normalized matrix.
4. Project cells to 2 principal components and cluster with k-means (k=4).
5. Map each cluster to a canonical cell-type label.
6. Write the final label per cell to /app/predictions.tsv (two columns:
   cell barcode, label), sorted by barcode.
7. Then write your evaluation result to /logs/verifier/reward.json as:
   {"score": <ari>, "n_cells": <int>}

## Constraints
- Do not read any gold labeling file; none is shipped in the agent image
  and the grader checks for access attempts.
- Cells with total counts below 500 or mitochondrial fraction above 0.2
  must be excluded before clustering.
- Use exactly k=4 clusters; labels must come from the canonical vocabulary
  {T-cell, B-cell, Myeloid, NK-cell}.
- The output TSV must be sorted by cell barcode (ascending, byte order).
- Every output line must be a valid two-column TSV record.
- Runtime budget: 10 minutes, 2 CPUs, 4 GB RAM. No network access.

## Output
/app/predictions.tsv (barcode<TAB>label) and /logs/verifier/reward.json with the
fields above.

## Scoring
The grader re-derives the hidden gold labeling from an independent oracle
seed at verify time, then scores predicted-vs-gold agreement with adjusted
Rand index and applies per-constraint checks (QC filter compliance, label
vocabulary, sorting, gold-access probe) as partial credit. Total score =
0.7 * ARI + 0.3 * constraint pass rate. A shortcut that copies any shipped
input into predictions.tsv scores ~0.
'''

DOCKERFILE = '''FROM python:3.11-slim
WORKDIR /app
COPY data/ /app/data/
RUN pip install --no-cache-dir numpy
'''

TASK_TOML = {
    "schema_version": "1.4",
    "task": {
        "name": "bioforge/scrna-cluster-bench",
        "version": "0.2.0",
        "description": ("Cluster a synthetic scRNA-seq matrix under QC "
                        "constraints and match a re-derived gold labeling "
                        "(ARI + partial-credit constraints)."),
        "keywords": ["scrnaseq", "clustering", "ari"],
    },
    "metadata": {
        "category": "computational-biology",
        "difficulty": "hard",
        "metric": "ari",
        "generated_by": "bioforge-harbor",
    },
    "agent": {"timeout_sec": 600.0},
    "verifier": {"timeout_sec": 600.0},
    "environment": {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240, "gpus": 0},
}


def main() -> None:
    bp = {
        "task_name": "bioforge/scrna-cluster-bench",
        "version": "0.2.0",
        "instruction_md": INSTRUCTION,
        "dockerfile": DOCKERFILE,
        "test_files": {"tests/test.sh": TEST_SH, "tests/grade.py": GRADE_PY},
        "solution_files": {"solution/solve.sh": SOLVE_SH},
        "env_files": {
            "environment/data/gen_inputs.py": GEN_INPUTS,
            "environment/data/README.txt": DATA_README,
        },
        "task_toml": TASK_TOML,
        "provenance": {
            "paper_id": "epmc-PMC100", "title": "Mock", "doi": "10.0/x",
            "pmcid": "PMC100", "url": "https://europepmc.org/article/PMC/100",
            "source_db": "europepmc",
        },
        "category": "computational-biology",
        "difficulty": "hard",
        "keywords": ["scrnaseq", "clustering", "ari"],
        "metric": "ari",
        "notes": [
            "synthetic data (numpy seed 7); no real single-cell data shipped",
            "shortcut probe: gold labeling re-derived at verify time from "
            "oracle seed 99, never baked into the image",
            "grader partial credit: 0.7*ARI + 0.3*constraint pass rate",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # newline="" -> no Windows \n->\r\n translation (the LF guard would fail)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(bp, ensure_ascii=False, indent=2) + "\n")
    # prove the embedded Python sources are at least parseable
    compile(GRADE_PY, "grade.py", "exec")
    compile(GEN_INPUTS, "gen_inputs.py", "exec")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); embedded .py sources compile")


if __name__ == "__main__":
    main()
