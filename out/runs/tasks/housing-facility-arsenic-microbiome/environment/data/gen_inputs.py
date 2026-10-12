#!/usr/bin/env python3
import os, sys, json
import numpy as np
import pandas as pd

def generate(data_dir, seed=20260803):
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = 80
    facility = np.array(['A']*40 + ['B']*40)
    treatment = np.array(['control','iAs']*20 + ['control','iAs']*20)
    order = rng.permutation(n)
    facility = facility[order]
    treatment = treatment[order]
    sample_ids = [f'S{i+1:03d}' for i in range(n)]
    batch = rng.integers(1, 4, size=n)
    auc = np.zeros(n)
    for i in range(n):
        f = facility[i]
        t = treatment[i]
        base = 1000.0
        if f == 'B':
            base -= 50.0
        if t == 'iAs':
            if f == 'A':
                base += 150.0
            else:
                base -= 50.0
        auc[i] = base + rng.normal(0, 40)
    weight = 25.0 + (facility == 'B')*2.0 + rng.normal(0, 1.5, n)
    meta = pd.DataFrame({
        'sample_id': sample_ids,
        'facility': facility,
        'treatment': treatment,
        'batch': batch,
        'glucose_auc': auc,
        'weight_g': weight
    })
    meta.to_csv(os.path.join(data_dir, 'metadata.csv'), index=False)
    n_sp = 200
    sp_ids = [f'sp_{i:03d}' for i in range(n_sp)]
    mu_sp = rng.normal(5, 1.5, n_sp)
    fac_sp = rng.normal(0, 0.1, n_sp)
    trt_sp = rng.normal(0, 0.02, n_sp)
    int_sp = rng.normal(0, 0.02, n_sp)
    sig_fac_sp = rng.choice(n_sp, 22, replace=False)
    fac_sp[sig_fac_sp] = rng.normal(0, 1.5, 22)
    sig_trt_sp = rng.choice(n_sp, 8, replace=False)
    trt_sp[sig_trt_sp] = rng.normal(0, 0.2, 8)
    sig_int_sp = rng.choice(n_sp, 5, replace=False)
    int_sp[sig_int_sp] = rng.normal(0, 0.2, 5)
    X_sp = np.zeros((n, n_sp), dtype=int)
    lib_sp = rng.lognormal(13, 0.5, n).astype(int)
    for i in range(n):
        f = 1.0 if facility[i] == 'B' else 0.0
        t = 1.0 if treatment[i] == 'iAs' else 0.0
        log_mu = mu_sp + fac_sp*(f-0.5) + trt_sp*(t-0.5) + int_sp*(f-0.5)*(t-0.5)
        log_mu += rng.normal(0, 0.1, n_sp)
        p = np.exp(log_mu - log_mu.max())
        p /= p.sum()
        X_sp[i] = rng.multinomial(lib_sp[i], p)
    df_sp = pd.DataFrame(X_sp, columns=sp_ids)
    df_sp.insert(0, 'sample_id', sample_ids)
    df_sp.to_csv(os.path.join(data_dir, 'microbiome_species.csv'), index=False)
    n_met = 300
    met_ids = [f'met_{i:03d}' for i in range(n_met)]
    base_met = rng.normal(10, 1.5, n_met)
    fac_met = rng.normal(0, 0.1, n_met)
    trt_met = rng.normal(0, 0.05, n_met)
    int_met = rng.normal(0, 0.02, n_met)
    sig_fac_met = rng.choice(n_met, 100, replace=False)
    fac_met[sig_fac_met] = rng.normal(0, 1.2, 100)
    sig_trt_met = rng.choice(n_met, 20, replace=False)
    trt_met[sig_trt_met] = rng.normal(0, 0.3, 20)
    sig_int_met = rng.choice(n_met, 10, replace=False)
    int_met[sig_int_met] = rng.normal(0, 0.3, 10)
    M = np.zeros((n, n_met))
    for i in range(n):
        f = 1.0 if facility[i] == 'B' else 0.0
        t = 1.0 if treatment[i] == 'iAs' else 0.0
        M[i] = base_met + fac_met*(f-0.5) + trt_met*(t-0.5) + int_met*(f-0.5)*(t-0.5) + rng.normal(0, 0.3, n_met)
    df_met = pd.DataFrame(M, columns=met_ids)
    df_met.insert(0, 'sample_id', sample_ids)
    df_met.to_csv(os.path.join(data_dir, 'metabolites.csv'), index=False)
    n_gene = 2000
    gene_ids = [f'gene_{i:04d}' for i in range(n_gene)]
    mu_gene = rng.normal(4, 1.0, n_gene)
    fac_gene = rng.normal(0, 0.02, n_gene)
    trt_gene = rng.normal(0, 0.01, n_gene)
    int_gene = rng.normal(0, 0.01, n_gene)
    sig_fac_gene = rng.choice(n_gene, 1200, replace=False)
    fac_gene[sig_fac_gene] = rng.normal(0, 0.6, 1200)
    sig_trt_gene = rng.choice(n_gene, 50, replace=False)
    trt_gene[sig_trt_gene] = rng.normal(0, 0.15, 50)
    sig_int_gene = rng.choice(n_gene, 30, replace=False)
    int_gene[sig_int_gene] = rng.normal(0, 0.15, 30)
    G = np.zeros((n, n_gene), dtype=int)
    lib_gene = rng.lognormal(14, 0.5, n).astype(int)
    for i in range(n):
        f = 1.0 if facility[i] == 'B' else 0.0
        t = 1.0 if treatment[i] == 'iAs' else 0.0
        log_mu = mu_gene + fac_gene*(f-0.5) + trt_gene*(t-0.5) + int_gene*(f-0.5)*(t-0.5)
        log_mu += rng.normal(0, 0.1, n_gene)
        p = np.exp(log_mu - log_mu.max())
        p /= p.sum()
        G[i] = rng.multinomial(lib_gene[i], p)
    df_gene = pd.DataFrame(G, columns=gene_ids)
    df_gene.insert(0, 'sample_id', sample_ids)
    df_gene.to_csv(os.path.join(data_dir, 'gene_families.csv'), index=False)
    readme = '''# Synthetic multi-omics dataset
This dataset mimics a two-facility mouse study with arsenic exposure.
Files:
- metadata.csv: sample_id, facility (A/B), treatment (control/iAs), batch, glucose_auc, weight_g
- microbiome_species.csv: sample_id followed by 200 species count columns (sp_000 ... sp_199)
- metabolites.csv: sample_id followed by 300 metabolite intensity columns (met_000 ... met_299)
- gene_families.csv: sample_id followed by 2000 gene family count columns (gene_0000 ... gene_1999)
'''
    with open(os.path.join(data_dir, 'README.txt'), 'w') as f:
        f.write(readme)

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    generate(data_dir)
