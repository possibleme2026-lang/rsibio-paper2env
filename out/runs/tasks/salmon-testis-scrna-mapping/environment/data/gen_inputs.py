#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

np.random.seed(42)
OUT = '/app/data'
os.makedirs(OUT, exist_ok=True)

N_GENES = 6000
N_REF = 2000
N_CULT = 1200
CELL_TYPES = ['Endothelial', 'Leydig', 'Macrophage', 'Peritubular', 'Sertoli', 'Spermatids', 'Spermatocytes', 'Spermatogonia']
CONDITIONS = ['Basal', 'Differentiation', 'EGF_IGF']

markers = {
    'Spermatogonia': ['DMRT1', 'PLZF', 'UTF1', 'SALL4', 'GFRA1', 'RET', 'ZBTB16', 'NANOS2', 'PIWIL1', 'MAGEA4'],
    'Spermatocytes': ['SYCP1', 'SYCP3', 'REC8', 'DMC1', 'SPO11', 'MLH1', 'TEX11', 'MEIOB', 'HORMAD1', 'RAD51'],
    'Spermatids': ['PRM1', 'PRM2', 'TNP1', 'TNP2', 'AKAP4', 'ODF1', 'SPATA18', 'GAPDHS', 'CATSPER1', 'IZUMO1'],
    'Sertoli': ['SOX9', 'AMH', 'GDNF', 'FSHR', 'WT1', 'GATA4', 'KITLG', 'CLDN11', 'DMRT1', 'FGF9'],
    'Leydig': ['STAR', 'CYP11A1', 'HSD3B1', 'CYP17A1', 'LHCGR', 'INSL3', 'NR5A1', 'CYP11B2', 'HSD17B3', 'POR'],
    'Peritubular': ['ACTA2', 'MYH11', 'DES', 'PDGFRB', 'COL1A1', 'COL1A2', 'POSTN', 'FN1', 'TAGLN', 'MYL9'],
    'Macrophage': ['CD68', 'CSF1R', 'CD163', 'LYZ', 'C1QA', 'C1QB', 'IL1B', 'TNF', 'CCL4', 'CXCL8'],
    'Endothelial': ['PECAM1', 'VWF', 'CDH5', 'CLDN5', 'FLT1', 'KDR', 'TEK', 'ESAM', 'ROBO4', 'CD34'],
}
stress_genes = ['HSPA1A', 'HSPA8', 'HSP90AA1', 'HSPB1', 'DNAJB1', 'JUN', 'FOS', 'ATF3', 'HIF1A', 'NFKB1', 'RELA', 'DDIT3', 'XBP1', 'ATF4', 'EIF2AK3', 'ERN1', 'BAX', 'CASP3', 'CASP8', 'CASP9', 'BCL2']
prolif_genes = ['MKI67', 'PCNA', 'CCND1', 'CCNB1', 'CDK1', 'TOP2A', 'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7', 'E2F1', 'CDC20', 'BUB1', 'AURKA', 'PLK1', 'CCNA2', 'CDK2', 'CDKN1A']
meiotic_genes = markers['Spermatocytes'] + ['SYCP2', 'TEX12', 'MNS1', 'SPATA22', 'RAD21L', 'MEI1', 'MEI4', 'STAG3', 'SMC1B', 'REC8']

genes = []
for ct in CELL_TYPES:
    genes.extend(markers[ct])
genes.extend(stress_genes)
genes.extend(prolif_genes)
genes.extend(meiotic_genes)
seen = set()
uniq = []
for g in genes:
    if g not in seen:
        seen.add(g)
        uniq.append(g)
genes = uniq
i = 1
while len(genes) < N_GENES:
    g = f'GENE{i:04d}'
    if g not in seen:
        seen.add(g)
        genes.append(g)
    i += 1
genes = genes[:N_GENES]
gene_idx = {g: i for i, g in enumerate(genes)}

pd.Series(genes).to_csv(os.path.join(OUT, 'genes.txt'), index=False, header=False)

base_log_mean = np.random.uniform(-2.0, 0.5, size=N_GENES)
type_log_mean = {}
for ct in CELL_TYPES:
    m = base_log_mean.copy()
    for g in markers[ct]:
        m[gene_idx[g]] = np.random.uniform(3.0, 5.5)
    type_log_mean[ct] = m

rng = np.random.default_rng(42)

ref_props = {
    'Spermatogonia': 0.25, 'Spermatocytes': 0.20, 'Spermatids': 0.15,
    'Sertoli': 0.15, 'Leydig': 0.10, 'Peritubular': 0.05,
    'Macrophage': 0.05, 'Endothelial': 0.05,
}
ref_types = []
for ct, p in ref_props.items():
    ref_types.extend([ct] * int(round(p * N_REF)))
while len(ref_types) < N_REF:
    ref_types.append(rng.choice(CELL_TYPES))
while len(ref_types) > N_REF:
    ref_types.pop()
rng.shuffle(ref_types)

ref_mats = []
for ct in ref_types:
    log_mean = type_log_mean[ct].copy()
    noise = rng.normal(0, 0.5, size=N_GENES)
    lam = np.exp(log_mean + noise)
    ref_mats.append(rng.poisson(lam))
ref_mat = np.array(ref_mats).T
ref_sparse = sparse.csr_matrix(ref_mat)
mmwrite(os.path.join(OUT, 'reference_counts.mtx'), ref_sparse)
ref_cells = [f'ref_{i:04d}' for i in range(N_REF)]
pd.DataFrame({'cell_id': ref_cells, 'cell_type': ref_types}).to_csv(os.path.join(OUT, 'reference_cells.csv'), index=False)

cond_props = {
    'EGF_IGF': {'Spermatogonia': 0.55, 'Spermatocytes': 0.15, 'Spermatids': 0.05, 'Sertoli': 0.15, 'Leydig': 0.03, 'Peritubular': 0.03, 'Macrophage': 0.02, 'Endothelial': 0.02},
    'Basal': {'Spermatogonia': 0.15, 'Spermatocytes': 0.05, 'Spermatids': 0.02, 'Sertoli': 0.60, 'Leydig': 0.08, 'Peritubular': 0.05, 'Macrophage': 0.03, 'Endothelial': 0.02},
    'Differentiation': {'Spermatogonia': 0.25, 'Spermatocytes': 0.35, 'Spermatids': 0.15, 'Sertoli': 0.15, 'Leydig': 0.05, 'Peritubular': 0.03, 'Macrophage': 0.01, 'Endothelial': 0.01},
}
for c in cond_props:
    s = sum(cond_props[c].values())
    for ct in cond_props[c]:
        cond_props[c][ct] /= s

cult_mats = []
cult_cells = []
cult_conditions = []
n_per_cond = N_CULT // 3
for cond in CONDITIONS:
    props = cond_props[cond]
    types = list(props.keys())
    probs = [props[t] for t in types]
    for i in range(n_per_cond):
        ct = rng.choice(types, p=probs)
        log_mean = type_log_mean[ct].copy()
        if cond == 'EGF_IGF':
            for g in prolif_genes:
                log_mean[gene_idx[g]] += np.log(2.5)
            for g in stress_genes:
                log_mean[gene_idx[g]] -= np.log(1.5)
        elif cond == 'Basal':
            if ct == 'Sertoli':
                for g in stress_genes:
                    log_mean[gene_idx[g]] += np.log(4.0)
                for g in markers['Sertoli']:
                    log_mean[gene_idx[g]] -= np.log(2.0)
            for g in stress_genes[:10]:
                log_mean[gene_idx[g]] += np.log(1.8)
        elif cond == 'Differentiation':
            for g in meiotic_genes:
                log_mean[gene_idx[g]] += np.log(1.3)
            for g in stress_genes[10:]:
                log_mean[gene_idx[g]] += np.log(1.2)
        noise = rng.normal(0, 0.5, size=N_GENES)
        lam = np.exp(log_mean + noise)
        cult_mats.append(rng.poisson(lam))
        cult_cells.append(f'cell_{cond}_{i:04d}')
        cult_conditions.append(cond)

cult_mat = np.array(cult_mats).T
cult_sparse = sparse.csr_matrix(cult_mat)
mmwrite(os.path.join(OUT, 'cultured_counts.mtx'), cult_sparse)
pd.DataFrame({'cell_id': cult_cells, 'condition': cult_conditions}).to_csv(os.path.join(OUT, 'cultured_cells.csv'), index=False)
print('Generated synthetic scRNA-seq data in', OUT)
