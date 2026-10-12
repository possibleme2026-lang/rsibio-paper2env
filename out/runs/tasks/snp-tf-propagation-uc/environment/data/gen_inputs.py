import numpy as np
import pandas as pd
import os

np.random.seed(42)
n_samples = 200
n_snps = 50
n_tfs = 20

outdir = '/app/data'
os.makedirs(outdir, exist_ok=True)

mafs = np.random.uniform(0.1, 0.5, n_snps)
genotypes = np.zeros((n_samples, n_snps), dtype=int)
for j in range(n_snps):
    genotypes[:, j] = np.random.binomial(2, mafs[j], n_samples)
snp_ids = [f'SNP_{i+1:03d}' for i in range(n_snps)]
sample_ids = [f'Sample_{i+1:03d}' for i in range(n_samples)]
df_geno = pd.DataFrame(genotypes, columns=snp_ids)
df_geno.insert(0, 'sample_id', sample_ids)
df_geno.to_csv(os.path.join(outdir, 'genotypes.tsv'), sep='\t', index=False)

tf_ids = [f'TF_{i+1:02d}' for i in range(n_tfs)]
snp_to_tf = np.random.choice(tf_ids, n_snps)
effect_weights = np.random.normal(0, 1, n_snps)
df_map = pd.DataFrame({'snp_id': snp_ids, 'tf_id': snp_to_tf, 'effect_weight': effect_weights})
df_map.to_csv(os.path.join(outdir, 'snp_tf_map.tsv'), sep='\t', index=False)

W = np.random.uniform(0, 1, (n_tfs, n_tfs))
W = (W + W.T) / 2
np.fill_diagonal(W, 0)
df_net = pd.DataFrame(W, index=tf_ids, columns=tf_ids)
df_net.to_csv(os.path.join(outdir, 'tf_network.tsv'), sep='\t')

severity = np.random.normal(0, 1, n_samples)
df_clin = pd.DataFrame({'sample_id': sample_ids, 'severity': severity})
df_clin.to_csv(os.path.join(outdir, 'clinical.tsv'), sep='\t', index=False)
