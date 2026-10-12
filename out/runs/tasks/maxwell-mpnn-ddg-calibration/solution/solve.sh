#!/bin/bash
set -euo pipefail
mkdir -p /app/output/landscape
python3 - <<'PY'
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

AA = list("ACDEFGHIKLMNPQRSTVWY")
AIDX = {a: i for i, a in enumerate(AA)}
DATA, OUT = "/app/data", "/app/output"

mats = {}
for fn in sorted(os.listdir(os.path.join(DATA, "mpnn"))):
    if fn.endswith(".csv"):
        mats[fn[:-4]] = pd.read_csv(os.path.join(DATA, "mpnn", fn))


def parse_mut(s):
    s = str(s).strip()
    return s[0], int(s[1:-1]), s[-1]


def features(pid, pos, wt_aa, mut_aa):
    # matrix-wise signal: logit of the mutant relative to the logit of the wild type
    logits = mats[pid][AA].to_numpy(dtype=float)[pos - 1]
    w, m = AIDX[wt_aa], AIDX[mut_aa]
    f = np.zeros(41)
    f[0] = logits[m] - logits[w]
    f[1 + w] = 1.0   # wild-type identity (needed so the WT->WT diagonal collapses to 0)
    f[21 + m] = 1.0  # mutant identity (residue propensity term)
    return f


train = pd.read_csv(os.path.join(DATA, "train_ddg.csv"))
X = np.zeros((len(train), 41))
for k, (pid, mut) in enumerate(zip(train["protein_id"], train["mutation"])):
    w, p, m = parse_mut(mut)
    X[k] = features(pid, p, w, m)
model = Ridge(alpha=1.0).fit(X, train["ddg"].to_numpy(dtype=float))


def raw(pid, pos, wt_aa, mut_aa):
    return float(model.predict(features(pid, pos, wt_aa, mut_aa)[None, :])[0])


landscape = {}
for pid in open(os.path.join(DATA, "landscape_proteins.txt")).read().split():
    wts = mats[pid]["wt"].tolist()
    L = len(wts)
    M = np.zeros((L, 20))
    for i in range(L):
        for j, a in enumerate(AA):
            M[i, j] = raw(pid, i + 1, wts[i], a)
        M[i, :] -= raw(pid, i + 1, wts[i], wts[i])   # anchor the wild-type diagonal to 0
    landscape[pid] = M
    out = pd.DataFrame(M, columns=AA)
    out.insert(0, "position", np.arange(1, L + 1))
    out.to_csv(os.path.join(OUT, "landscape", pid + ".csv"), index=False)

test = pd.read_csv(os.path.join(DATA, "test_mutations.csv"))
rows = []
for pid, mut in zip(test["protein_id"], test["mutation"]):
    w, p, m = parse_mut(mut)
    v = float(landscape[pid][p - 1, AIDX[m]]) if pid in landscape else raw(pid, p, w, m)
    rows.append((pid, mut, v))
pd.DataFrame(rows, columns=["protein_id", "mutation", "predicted_ddg"]).to_csv(
    os.path.join(OUT, "predictions.csv"), index=False)
print("wrote", len(rows), "test predictions and", len(landscape), "landscapes")
PY
