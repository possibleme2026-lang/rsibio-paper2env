#!/bin/bash
set -euo pipefail
mkdir -p /app/output
cat > /app/solve_pipeline.py <<'PYEOF'
import json
import numpy as np
import pandas as pd
from Bio import Phylo
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

DATA = "/app/data"
OUT = "/app/output"
TRAITS = ["body_elongation", "relative_eye_size", "oral_gape_position",
          "caudal_fin_ratio", "pectoral_fin_ratio", "mouth_width_ratio",
          "head_length_ratio", "body_depth_ratio"]
LAT = ["body_elongation", "relative_eye_size", "oral_gape_position"]
CLADES = ["Perciformes", "Cypriniformes", "Salmoniformes",
          "Gadiformes", "Clupeiformes", "Pleuronectiformes"]


def key(s):
    return str(s).strip().replace(" ", "_")


morph = pd.read_csv(DATA + "/morphology.csv")
lh = pd.read_csv(DATA + "/life_history_train.csv")
test_disp = [l.strip() for l in open(DATA + "/test_species.txt") if l.strip()]
tree = Phylo.read(DATA + "/phylogeny.nwk", "newick")
tree_names = [t.name for t in tree.get_terminals()]

morph["key"] = morph["species"].map(key)
lh["key"] = lh["species"].map(key)
treek = set(tree_names)
morph = morph[morph["key"].isin(treek)].reset_index(drop=True)

for clade, grp in morph.groupby("clade"):
    idx = list(grp.index)
    for t in TRAITS:
        med = morph.loc[idx, t].median()
        morph.loc[idx, t] = morph.loc[idx, t].fillna(med)

# --- Brownian-motion covariance of the phylogeny -------------------------
def path_depths(tr):
    out = {}
    for term in tr.get_terminals():
        d = {}
        dep = 0.0
        for cl in tr.get_path(term):
            dep += (cl.branch_length or 0.0)
            d[id(cl)] = dep
        out[term.name] = d
    return out


paths = path_depths(tree)
n = len(tree_names)
height = max(max(p.values()) for p in paths.values())
C = np.zeros((n, n))
for i in range(n):
    pa = paths[tree_names[i]]
    C[i, i] = height
    for j in range(i + 1, n):
        pb = paths[tree_names[j]]
        h = 0.0
        for k, v in pa.items():
            if v > h and k in pb:
                h = v
        C[i, j] = C[j, i] = h

M = morph.set_index("key").loc[tree_names, TRAITS].to_numpy(float)

# --- phylogenetic PCA ----------------------------------------------------
X = (M - M.mean(0)) / M.std(0, ddof=1)
w, V = np.linalg.eigh(C)
Cinv_half = (V * (1.0 / np.sqrt(w))) @ V.T
Xw = Cinv_half @ X
Xw = Xw - Xw.mean(0)
_, s, Vt = np.linalg.svd(Xw, full_matrices=False)
ev = s ** 2 / (n - 1)
var_expl = float(ev[0] / ev.sum())
load = Vt[0].copy()
if load[TRAITS.index("body_elongation")] < 0:
    load = -load

# --- PGLS of log generation turnover on the lateral-size traits ----------
train_key = set(lh["key"])
tri = np.array([i for i, k in enumerate(tree_names) if k in train_key], int)
Mtr = M[tri]
Ztr = (Mtr - Mtr.mean(0)) / Mtr.std(0, ddof=1)
lh_idx = lh.set_index("key")
y_gt = np.log(lh_idx.loc[[tree_names[_k] for _k in tri], "generation_turnover_years"].to_numpy(float))
y_rp = np.log(lh_idx.loc[[tree_names[_k] for _k in tri], "reproductive_output_index"].to_numpy(float))
Ctr = C[np.ix_(tri, tri)]
Ci = np.linalg.inv(Ctr)
cols = [TRAITS.index(t) for t in LAT]
Xd = np.column_stack([np.ones(len(tri))] + [Ztr[:, c] for c in cols])
beta = np.linalg.solve(Xd.T @ Ci @ Xd, Xd.T @ Ci @ y_gt)
slopes = {t: float(b) for t, b in zip(LAT, beta[1:])}

# --- predictive models with water-column / clade interactions ------------
def design(Z, wc, cl):
    parts = [Z]
    for lvl in ["pelagic", "demersal", "benthic"]:
        parts.append(Z[:, cols] * (wc == lvl).astype(float)[:, None])
    for lvl in CLADES[1:]:
        parts.append(Z[:, cols] * (cl == lvl).astype(float)[:, None])
    return np.hstack(parts)


md = morph.set_index("key").loc[[tree_names[_k] for _k in tri]]
Ftr = design(Ztr, md["water_column"].to_numpy(), md["clade"].to_numpy())
kf = KFold(n_splits=5, shuffle=True, random_state=0)


def cv_rmse(F, yv):
    p = cross_val_predict(Ridge(alpha=1.0), F, yv, cv=kf)
    return float(np.sqrt(np.mean((p - yv) ** 2)))


rmse_gt = cv_rmse(Ftr, y_gt)
rmse_rp = cv_rmse(Ftr, y_rp)

te = [k for k in tree_names if k not in train_key]
mte = morph.set_index("key").loc[te]
Zte = (mte[TRAITS].to_numpy(float) - Mtr.mean(0)) / Mtr.std(0, ddof=1)
Fte = design(Zte, mte["water_column"].to_numpy(), mte["clade"].to_numpy())

pred_gt = np.exp(Ridge(alpha=1.0).fit(Ftr, y_gt).predict(Fte))
pred_rp = np.exp(Ridge(alpha=1.0).fit(Ftr, y_rp).predict(Fte))

disp = dict(zip(morph["key"], morph["species"]))
with open(OUT + "/predictions.csv", "w") as fh:
    print("species,generation_turnover,reproductive_output", file=fh)
    for i, k in enumerate(te):
        print("%s,%.4f,%.4f" % (disp[k], pred_gt[i], pred_rp[i]), file=fh)

out = {
    "n_species_analysed": int(n),
    "n_training_species": int(len(tri)),
    "phylo_pca_morphology": {
        "pc1_variance_explained": var_expl,
        "pc1_loadings": {t: float(v) for t, v in zip(TRAITS, load)},
    },
    "pgls_log_generation_turnover": slopes,
    "cv_rmse_log_generation_turnover": rmse_gt,
    "cv_rmse_log_reproductive_output": rmse_rp,
}
with open(OUT + "/diagnostics.json", "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
PYEOF
python /app/solve_pipeline.py
