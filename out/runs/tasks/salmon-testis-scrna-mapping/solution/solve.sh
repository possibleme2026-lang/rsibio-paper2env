#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python <<'PY'
import os, json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.stats import ttest_ind

DATA = '/app/data'
OUT = '/app/output'
ref = mmread(os.path.join(DATA, 'reference_counts.mtx')).tocsr()
cult = mmread(os.path.join(DATA, 'cultured_counts.mtx')).tocsr()
genes = pd.read_csv(os.path.join(DATA, 'genes.txt'), header=None)[0].astype(str).tolist()
ref_cells = pd.read_csv(os.path.join(DATA, 'reference_cells.csv'))
cult_cells = pd.read_csv(os.path.join(DATA, 'cultured_cells.csv'))

ref_totals = np.asarray(ref.sum(axis=1)).ravel()
keep = ref_totals >= 10
ref = ref[keep, :]
cult = cult[keep, :]
genes = [g for g, k in zip(genes, keep) if k]
n_genes_filtered = len(genes)

def norm_log2(mat):
    totals = np.asarray(mat.sum(axis=0)).ravel().astype(float)
    totals[totals == 0] = 1.0
    scale = 1e4 / totals
    mat = mat.tocsr() @ sparse.diags(scale)
    mat = mat.tocsr()
    mat.data = np.log2(1 + mat.data)
    return mat

ref_log = norm_log2(ref)
cult_log = norm_log2(cult)

ref_types = ref_cells['cell_type'].values
unique_types = sorted(ref_cells['cell_type'].unique())
centroids = []
for ct in unique_types:
    idx = np.where(ref_types == ct)[0]
    centroids.append(np.asarray(ref_log[:, idx].mean(axis=1)).ravel())
cent_mat = np.vstack(centroids)

cult_dense = cult_log.toarray().T
A = cult_dense - cult_dense.mean(axis=1, keepdims=True)
B = cent_mat - cent_mat.mean(axis=1, keepdims=True)
A_norm = np.sqrt((A ** 2).sum(axis=1, keepdims=True))
A_norm[A_norm == 0] = 1
B_norm = np.sqrt((B ** 2).sum(axis=1, keepdims=True))
B_norm[B_norm == 0] = 1
corr = (A @ B.T) / (A_norm * B_norm.T)
best = np.argmax(corr, axis=1)
assigned = [unique_types[i] for i in best]
assign_df = pd.DataFrame({'cell_id': cult_cells['cell_id'], 'condition': cult_cells['condition'], 'assigned_type': assigned})
assign_df.to_csv(os.path.join(OUT, 'assignments.csv'), index=False)

comp_rows = []
for cond in ['Basal', 'Differentiation', 'EGF_IGF']:
    sub = assign_df[assign_df['condition'] == cond]
    total = len(sub)
    for ct in unique_types:
        n = int((sub['assigned_type'] == ct).sum())
        comp_rows.append({'condition': cond, 'cell_type': ct, 'n_cells': n, 'proportion': n / total})
comp = pd.DataFrame(comp_rows)
comp.to_csv(os.path.join(OUT, 'composition.csv'), index=False)

props = comp.pivot(index='condition', columns='cell_type', values='proportion').fillna(0)
enrich_rows = []
for cond in ['Basal', 'Differentiation', 'EGF_IGF']:
    others = [c for c in props.index if c != cond]
    mean_other = props.loc[others].mean(axis=0)
    log2fc = np.log2((props.loc[cond] + 1e-6) / (mean_other + 1e-6))
    enriched_type = log2fc.idxmax()
    enrich_rows.append({'condition': cond, 'enriched_type': enriched_type, 'log2FC': float(log2fc[enriched_type])})
pd.DataFrame(enrich_rows).to_csv(os.path.join(OUT, 'enrichment.csv'), index=False)

sertoli = assign_df[assign_df['assigned_type'] == 'Sertoli']
egf_ids = sertoli[sertoli['condition'] == 'EGF_IGF']['cell_id'].tolist()
basal_ids = sertoli[sertoli['condition'] == 'Basal']['cell_id'].tolist()
cid2idx = {cid: i for i, cid in enumerate(cult_cells['cell_id'])}
egf_idx = [cid2idx[c] for c in egf_ids]
basal_idx = [cid2idx[c] for c in basal_ids]
egf_expr = cult_log[:, egf_idx].toarray()
basal_expr = cult_log[:, basal_idx].toarray()
mean_egf = egf_expr.mean(axis=1)
mean_basal = basal_expr.mean(axis=1)
log2fc = mean_egf - mean_basal
pvals = []
for i in range(egf_expr.shape[0]):
    if np.all(egf_expr[i] == egf_expr[i, 0]) and np.all(basal_expr[i] == basal_expr[i, 0]):
        pvals.append(1.0)
    else:
        _, p = ttest_ind(egf_expr[i], basal_expr[i], equal_var=False)
        pvals.append(p if not np.isnan(p) else 1.0)
pvals = np.array(pvals)
n = len(pvals)
order = np.argsort(pvals)
ranked = pvals[order]
adj = ranked * n / (np.arange(1, n + 1))
adj = np.minimum.accumulate(adj[::-1])[::-1]
adj = np.clip(adj, 0, 1)
adj_full = np.empty(n)
adj_full[order] = adj
de = pd.DataFrame({'gene': genes, 'log2FC': log2fc, 'p_value': pvals, 'adj_p_value': adj_full})
de = de[de['log2FC'] > 0].sort_values(by=['adj_p_value', 'log2FC', 'gene'], ascending=[True, False, True]).head(10)
de.to_csv(os.path.join(OUT, 'sertoli_de.csv'), index=False)

qc = {
    'n_genes_filtered': int(n_genes_filtered),
    'n_reference_cells': int(len(ref_cells)),
    'n_cultured_cells': int(len(cult_cells)),
    'n_sertoli_EGF_IGF': int(len(egf_ids)),
    'n_sertoli_Basal': int(len(basal_ids)),
}
with open(os.path.join(OUT, 'qc_summary.json'), 'w') as f:
    json.dump(qc, f, indent=2)
PY
