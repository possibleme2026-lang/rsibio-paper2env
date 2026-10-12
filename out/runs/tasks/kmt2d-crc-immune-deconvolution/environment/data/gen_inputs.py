#!/usr/bin/env python3
"""Deterministic generator of the synthetic KMT2D/CRC cohort.

Running this script recreates every input file deterministically (fixed seed).
"""
import os
import sys
import numpy as np
import pandas as pd

SEED = 20260731
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

N = 240
samples = [f'TCGA-CRC-{i:04d}' for i in range(1, N + 1)]

raw = rng.random(N)
status = np.where(raw < 0.25, 'LOF', np.where(raw < 0.35, 'MISSENSE', 'WT'))
is_lof = status == 'LOF'

checkpoint = ['CD274', 'CTLA4', 'PDCD1', 'LAG3', 'HAVCR2']
t_effector = ['GZMA', 'GZMB', 'PRF1', 'IFNG', 'CXCL9', 'CXCL10']
ifn_gamma = ['IDO1', 'STAT1', 'HLA-DRA', 'CXCL11', 'IFNG']
cd8_t = ['CD8A', 'CD8B', 'GZMK']
nk_cell = ['NKG7', 'KLRD1', 'GNLY']
macrophage = ['CD68', 'CD163', 'CSF1R', 'MSR1']
treg = ['FOXP3', 'IL2RA', 'IKZF2']
housekeeping = ['ACTB', 'GAPDH', 'TUBB', 'RPL13A', 'RPS18', 'VIM', 'PCNA',
                'MKI67', 'TOP2A', 'BIRC5', 'COL1A1', 'FN1', 'SPARC', 'THBS1',
                'EGFR', 'KRAS', 'TP53', 'APC', 'SMAD4', 'MYC', 'CCND1', 'CDX2',
                'MUC2', 'TFF3', 'LGALS4', 'CDH17', 'KRT20', 'CEACAM5', 'EPCAM',
                'B2M']

signal = {}
for g in checkpoint:
    signal[g] = rng.uniform(0.5, 1.1)
for g in t_effector:
    signal[g] = rng.uniform(0.6, 1.2)
for g in ifn_gamma:
    signal[g] = rng.uniform(0.5, 1.0)
for g in cd8_t:
    signal[g] = rng.uniform(0.6, 1.1)
for g in nk_cell:
    signal[g] = rng.uniform(0.4, 0.9)
for g in macrophage:
    signal[g] = rng.uniform(0.3, 0.8)
for g in treg:
    signal[g] = -rng.uniform(0.4, 0.9)
for g in housekeeping:
    signal[g] = 0.0

genes = list(dict.fromkeys(checkpoint + t_effector + ifn_gamma + cd8_t +
                           nk_cell + macrophage + treg + housekeeping))
base = {g: (rng.uniform(6.0, 9.0) if signal[g] == 0.0 else rng.uniform(3.0, 7.0))
        for g in genes}
sample_eff = rng.normal(0.0, 0.3, N)

mat = np.zeros((len(genes), N))
for i, g in enumerate(genes):
    mat[i] = base[g] + signal[g] * is_lof + sample_eff + rng.normal(0.0, 0.6, N)
mat = np.clip(mat, 0.01, None)
expr = pd.DataFrame(mat, index=genes, columns=samples)
expr.index.name = 'gene'
expr.round(4).to_csv(os.path.join(OUT, 'expression_matrix.tsv'), sep='\t')

lof_classes = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'Splice_Site']
rows = []
for i, s in enumerate(samples):
    if status[i] == 'LOF':
        rows.append((s, 'KMT2D', str(rng.choice(lof_classes)),
                     'p.Gln%dTer' % int(rng.integers(50, 1500))))
    elif status[i] == 'MISSENSE':
        rows.append((s, 'KMT2D', 'Missense_Mutation',
                     'p.Arg%dHis' % int(rng.integers(50, 1500))))
mut = pd.DataFrame(rows, columns=['sample_id', 'gene',
                                  'variant_classification', 'hgvsp'])
mut.to_csv(os.path.join(OUT, 'kmt2d_mutations.tsv'), sep='\t', index=False)

ici = np.where(is_lof, rng.random(N) < 0.55, rng.random(N) < 0.45)
resp = []
for i in range(N):
    if not ici[i]:
        resp.append('NA')
    else:
        p = 0.55 if is_lof[i] else 0.22
        resp.append('PR' if rng.random() < p else 'SD')
os_m = np.zeros(N)
os_e = np.zeros(N, dtype=int)
for i in range(N):
    if ici[i]:
        scale = 34.0 if is_lof[i] else 16.0
        os_m[i] = float(np.clip(rng.exponential(scale), 1.0, 120.0))
        os_e[i] = int(rng.random() < 0.75)
    else:
        os_m[i] = float(np.clip(rng.exponential(22.0), 1.0, 120.0))
        os_e[i] = int(rng.random() < 0.70)
cancer = [str(rng.choice(['COAD', 'READ'])) for _ in range(N)]
clin = pd.DataFrame({'sample_id': samples, 'cancer_type': cancer,
                     'ici_treated': np.where(ici, 'yes', 'no'),
                     'response': resp, 'os_months': np.round(os_m, 2),
                     'os_event': os_e})
clin.to_csv(os.path.join(OUT, 'clinical.tsv'), sep='\t', index=False)

M = 60
cl_status = np.where(rng.random(M) < 0.33, 'LOF', 'WT')
auc = np.where(cl_status == 'LOF', rng.normal(6.3, 1.0, M),
               rng.normal(8.6, 1.0, M))
auc = np.clip(auc, 0.5, None)
drug = pd.DataFrame({'cell_line': ['CRC-CL-%03d' % i for i in range(1, M + 1)],
                     'kmt2d_status': cl_status,
                     'cisplatin_auc': np.round(auc, 3)})
drug.to_csv(os.path.join(OUT, 'cell_line_drug_screen.tsv'), sep='\t', index=False)

print('inputs written to', OUT)
