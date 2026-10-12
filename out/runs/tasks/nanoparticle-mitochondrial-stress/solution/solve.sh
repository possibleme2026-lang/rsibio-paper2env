#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import os, json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

expr = pd.read_csv("/app/data/expression_matrix.csv").set_index("sample_id")
meta = pd.read_csv("/app/data/sample_metadata.csv").set_index("sample_id")
expr = expr.loc[meta.index]
genes = list(expr.columns)

z = pd.DataFrame(StandardScaler().fit_transform(expr), index=expr.index, columns=genes)

pca = PCA(n_components=5, svd_solver="full", random_state=0)
pcs = pca.fit_transform(z)
evr = [float(x) for x in pca.explained_variance_ratio_[:3]]

def top_genes(comp, k=5):
    load = np.abs(pca.components_[comp])
    order = sorted(range(len(genes)), key=lambda i: (-load[i], genes[i]))
    return [genes[i] for i in order[:k]]

pc1 = top_genes(0)
pc2 = top_genes(1)

dist = 1 - np.corrcoef(z.values)
dist = np.clip(dist, 0, 2)
Z = linkage(dist, method="average")
clusters = fcluster(Z, t=3, criterion="maxclust")
cluster_means = {c: pcs[clusters == c, 0].mean() for c in np.unique(clusters)}
sorted_clusters = sorted(cluster_means, key=lambda c: cluster_means[c])
mapping = {old: new for new, old in enumerate(sorted_clusters)}
labels = np.array([mapping[c] for c in clusters])
hclust = {sid: int(lab) for sid, lab in zip(z.index, labels)}

corr = np.corrcoef(z.values.T)
adj = np.abs(corr) >= 0.7
np.fill_diagonal(adj, False)
degrees = adj.sum(axis=1)
hub_order = sorted(range(len(genes)), key=lambda i: (-degrees[i], genes[i]))
hubs = [genes[i] for i in hub_order[:5]]

particles = meta["particle"].unique()
edge_sets = []
for p in particles:
    sub = expr[meta["particle"] == p]
    c = np.corrcoef(sub.values.T)
    a = np.abs(c) >= 0.7
    np.fill_diagonal(a, False)
    edges = set()
    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            if a[i,j]:
                e = (genes[i], genes[j]) if genes[i] < genes[j] else (genes[j], genes[i])
                edges.add(e)
    edge_sets.append(edges)
conserved = edge_sets[0].intersection(*edge_sets[1:])
conserved_count = len(conserved)
r_means = {}
for e in conserved:
    rs = []
    for p in particles:
        sub = expr[meta["particle"] == p]
        i = genes.index(e[0]); j = genes.index(e[1])
        rs.append(abs(np.corrcoef(sub.values[:,i], sub.values[:,j])[0,1]))
    r_means[e] = np.mean(rs)
top_edges = sorted(conserved, key=lambda e: (-r_means[e], e[0], e[1]))[:5]
top_edges_str = [f"{e[0]}-{e[1]}" for e in top_edges]

i = genes.index("OMA1"); j = genes.index("DELE1")
oma1_dele1 = float(np.corrcoef(expr.values[:,i], expr.values[:,j])[0,1])

oxi = ["NFE2L2","HMOX1","NQO1","GCLM","TXN","SOD1","CAT","GPX1"]
adapt = ["PPARGC1A","TFAM","NRF1","POLG","TWNK"]
isr = ["OMA1","DELE1","ATF4","DDIT3","CHAC1"]
inflam = ["NFKB1","TNF","IL6","NLRP3","IL1B","CXCL8"]
complexg = ["MT-ND1","NDUFA1","SDHA","UQCRC1","COX5A","ATP5F1A"]
scores = pd.DataFrame(index=z.index)
scores["oxidative"] = z[oxi].mean(axis=1)
scores["adaptation"] = z[adapt].mean(axis=1)
scores["isr"] = z[isr].mean(axis=1)
scores["inflammation"] = z[inflam].mean(axis=1)
scores["complex"] = z[complexg].mean(axis=1)
scores = scores.join(meta[["particle","time_h"]])
grouped = scores.groupby(["particle","time_h"]).mean()

particle_response = {}
for p in particles:
    sub = grouped.loc[p]
    O = sub["oxidative"]; A = sub["adaptation"]; I = sub["isr"]; F = sub["inflammation"]; C = sub["complex"]
    peak = O.idxmax()
    if peak == 6 and O[72] < 0.5 * O[6]:
        kinetics = "acute"
    elif peak == 72 and O[72] >= 0.8 * max(O[6], O[24]):
        kinetics = "chronic"
    else:
        kinetics = "mixed"
    if A[72] - A[6] >= 0.3 and I[72] < 0.0 and C[72] > 0.0:
        mito_state = "adaptation"
    elif I[72] > 0.5 or C[72] < -0.5:
        mito_state = "dysfunction"
    elif A[72] - A[6] >= 0.5 and I[72] > 0.0:
        mito_state = "biogenesis"
    else:
        mito_state = "other"
    isr_axis = "active" if I.mean() > 0.0 else "weak"
    inflammation = "high" if F[72] > 0.0 else "low"
    particle_response[p] = {
        "kinetics": kinetics,
        "mito_state": mito_state,
        "isr_axis": isr_axis,
        "inflammation": inflammation
    }

conserved_genes = set()
for e in conserved:
    conserved_genes.add(e[0]); conserved_genes.add(e[1])
biomarkers = sorted([g for g in hubs if g in conserved_genes])

out = {
    "pca_explained_variance": evr,
    "pc1_top_genes": pc1,
    "pc2_top_genes": pc2,
    "hierarchical_clusters": hclust,
    "hub_genes": hubs,
    "oma1_dele1_correlation": round(oma1_dele1, 4),
    "conserved_network_edge_count": conserved_count,
    "conserved_network_top_edges": top_edges_str,
    "particle_response": particle_response,
    "biomarkers": biomarkers
}
os.makedirs("/app/output", exist_ok=True)
with open("/app/output/results.json", "w") as f:
    json.dump(out, f, indent=2)
PY
