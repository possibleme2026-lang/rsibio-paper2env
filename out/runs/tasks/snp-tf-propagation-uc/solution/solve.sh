#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python << 'EOF'
import numpy as np
import pandas as pd
from scipy import stats

geno = pd.read_csv('/app/data/genotypes.tsv', sep='\t')
snp_map = pd.read_csv('/app/data/snp_tf_map.tsv', sep='\t')
net = pd.read_csv('/app/data/tf_network.tsv', sep='\t', index_col=0)
clin = pd.read_csv('/app/data/clinical.tsv', sep='\t')

sample_ids = geno['sample_id'].values
snp_cols = [c for c in geno.columns if c != 'sample_id']
geno_mat = geno[snp_cols].values

tf_ids = net.index.tolist()
n_tfs = len(tf_ids)
n_samples = geno_mat.shape[0]

P = np.zeros((n_samples, n_tfs))
tf_to_idx = {tf: i for i, tf in enumerate(tf_ids)}
for _, row in snp_map.iterrows():
    snp = row['snp_id']
    tf = row['tf_id']
    w = row['effect_weight']
    if snp not in snp_cols or tf not in tf_to_idx:
        continue
    snp_idx = snp_cols.index(snp)
    tf_idx = tf_to_idx[tf]
    P[:, tf_idx] += geno_mat[:, snp_idx] * w

W = net.values.astype(float)
row_sums = W.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
W_norm = W / row_sums

alpha = 0.7
I = np.eye(n_tfs)
A = I - alpha * W_norm
B = (1 - alpha) * P.T
F_T = np.linalg.solve(A, B)
F = F_T.T

Z = (F - F.mean(axis=0)) / F.std(axis=0, ddof=0)

severity = clin.set_index('sample_id').loc[sample_ids, 'severity'].values
correlations = []
for i, tf in enumerate(tf_ids):
    r, p = stats.pearsonr(Z[:, i], severity)
    correlations.append({'tf': tf, 'pearson_r': r, 'p_value': p})
corr_df = pd.DataFrame(correlations)

corr_df['abs_r'] = corr_df['pearson_r'].abs()
top = corr_df.sort_values('abs_r', ascending=False).head(3).reset_index(drop=True)
top['rank'] = top.index + 1
top = top[['rank', 'tf', 'abs_r']]

Z_df = pd.DataFrame(Z, columns=tf_ids)
Z_df.insert(0, 'sample_id', sample_ids)
Z_df.to_csv('/app/output/tf_activity.tsv', sep='\t', index=False)
corr_df[['tf', 'pearson_r', 'p_value']].to_csv('/app/output/correlations.tsv', sep='\t', index=False)
top.to_csv('/app/output/top_tfs.tsv', sep='\t', index=False)
EOF
