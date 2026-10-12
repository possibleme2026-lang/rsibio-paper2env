#!/bin/bash
set -euo pipefail
mkdir -p /app/output
cat > /app/analysis.py << 'PYEOF'
#!/usr/bin/env python3
import argparse, json, os
import numpy as np
from scipy.spatial.distance import braycurtis
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import mannwhitneyu, false_discovery_control
from skbio.stats.distance import permanova
from skbio import DistanceMatrix

def compute_permanova_r2(dist_matrix, labels):
    labels = np.asarray(labels)
    n = len(labels)
    D2 = dist_matrix.to_data_frame() ** 2
    SS_T = D2.values.sum() / (2 * n)
    SS_W = 0.0
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        n_g = len(idx)
        sub = D2.iloc[idx, idx]
        SS_W += sub.values.sum() / (2 * n_g)
    SS_A = SS_T - SS_W
    return SS_A / SS_T if SS_T > 0 else 0.0

def diff_abundance(df, group):
    results = []
    for col in df.columns:
        a = df.loc[group==0, col].values
        b = df.loc[group==1, col].values
        mean_a = np.mean(a) + 1e-9
        mean_b = np.mean(b) + 1e-9
        log2fc = np.log2(mean_b / mean_a)
        stat, p = mannwhitneyu(a, b, alternative='two-sided')
        results.append((col, log2fc, p))
    res = pd.DataFrame(results, columns=['feature', 'log2FC', 'pvalue'])
    res['qvalue'] = false_discovery_control(res['pvalue'])
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(args.data_dir, 'metadata.csv')).set_index('sample_id')
    sp = pd.read_csv(os.path.join(args.data_dir, 'microbiome_species.csv')).set_index('sample_id')
    met = pd.read_csv(os.path.join(args.data_dir, 'metabolites.csv')).set_index('sample_id')
    gene = pd.read_csv(os.path.join(args.data_dir, 'gene_families.csv')).set_index('sample_id')
    common = meta.index.intersection(sp.index)
    meta = meta.loc[common]
    sp = sp.loc[common]
    met = met.loc[common]
    gene = gene.loc[common]
    y_fac = (meta['facility'] == 'B').astype(int).values
    y_trt = (meta['treatment'] == 'iAs').astype(int).values
    sp_rel = sp.div(sp.sum(axis=1), axis=0)
    prevalence = (sp > 0).mean(axis=0)
    keep = prevalence >= 0.2
    sp_filt = sp_rel.loc[:, keep]
    dist = DistanceMatrix.from_iterable(sp_filt.values, metric=braycurtis, keys=sp_filt.index)
    perm_fac = permanova(dist, meta['facility'], permutations=999)
    perm_trt = permanova(dist, meta['treatment'], permutations=999)
    r2_fac = compute_permanova_r2(dist, meta['facility'].values)
    r2_trt = compute_permanova_r2(dist, meta['treatment'].values)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_fac = RandomForestClassifier(n_estimators=500, random_state=42)
    acc_fac = cross_val_score(rf_fac, sp_filt, y_fac, cv=cv, scoring='accuracy').mean()
    rf_trt = RandomForestClassifier(n_estimators=500, random_state=42)
    acc_trt = cross_val_score(rf_trt, sp_filt, y_trt, cv=cv, scoring='accuracy').mean()
    rf_fac.fit(sp_filt, y_fac)
    imp = pd.Series(rf_fac.feature_importances_, index=sp_filt.columns)
    sig = imp.sort_values(ascending=False).head(22).index.tolist()
    diff_sp_fac = diff_abundance(sp_filt, y_fac)
    diff_sp_trt = diff_abundance(sp_filt, y_trt)
    sp_diff = pd.DataFrame({'species_id': sp_filt.columns})
    sp_diff['log2FC_facility'] = diff_sp_fac.set_index('feature').loc[sp_filt.columns, 'log2FC'].values
    sp_diff['pvalue_facility'] = diff_sp_fac.set_index('feature').loc[sp_filt.columns, 'pvalue'].values
    sp_diff['qvalue_facility'] = diff_sp_fac.set_index('feature').loc[sp_filt.columns, 'qvalue'].values
    sp_diff['log2FC_treatment'] = diff_sp_trt.set_index('feature').loc[sp_filt.columns, 'log2FC'].values
    sp_diff['pvalue_treatment'] = diff_sp_trt.set_index('feature').loc[sp_filt.columns, 'pvalue'].values
    sp_diff['qvalue_treatment'] = diff_sp_trt.set_index('feature').loc[sp_filt.columns, 'qvalue'].values
    sp_diff.to_csv(os.path.join(args.out_dir, 'differential_species.csv'), index=False)
    gene_rel = gene.div(gene.sum(axis=1), axis=0)
    diff_gene_fac = diff_abundance(gene_rel, y_fac)
    diff_gene_trt = diff_abundance(gene_rel, y_trt)
    diff_met_fac = diff_abundance(met, y_fac)
    diff_met_trt = diff_abundance(met, y_trt)
    n_sp_fac = int((sp_diff['qvalue_facility'] < 0.05).sum())
    n_sp_trt = int((sp_diff['qvalue_treatment'] < 0.05).sum())
    n_gene_fac = int((diff_gene_fac['qvalue'] < 0.05).sum())
    n_gene_trt = int((diff_gene_trt['qvalue'] < 0.05).sum())
    n_met_fac = int((diff_met_fac['qvalue'] < 0.05).sum())
    n_met_trt = int((diff_met_trt['qvalue'] < 0.05).sum())
    vp = pd.DataFrame({
        'term': ['facility', 'treatment'],
        'R2': [r2_fac, r2_trt],
        'pvalue': [perm_fac['p-value'], perm_trt['p-value']]
    })
    vp.to_csv(os.path.join(args.out_dir, 'variance_partition.csv'), index=False)
    with open(os.path.join(args.out_dir, 'signature_species.txt'), 'w') as f:
        for s in sig:
            f.write(s + '\n')
    results = {
        'facility_cv_accuracy': float(acc_fac),
        'treatment_cv_accuracy': float(acc_trt),
        'facility_variance_explained': float(r2_fac),
        'treatment_variance_explained': float(r2_trt),
        'n_signature_species': len(sig),
        'n_differential_species_facility': n_sp_fac,
        'n_differential_species_treatment': n_sp_trt,
        'n_differential_genes_facility': n_gene_fac,
        'n_differential_genes_treatment': n_gene_trt,
        'n_differential_metabolites_facility': n_met_fac,
        'n_differential_metabolites_treatment': n_met_trt
    }
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
PYEOF
python /app/analysis.py --data-dir /app/data --out-dir /app/output
