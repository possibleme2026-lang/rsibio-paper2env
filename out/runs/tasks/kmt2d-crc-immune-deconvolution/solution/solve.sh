#!/bin/bash
set -euo pipefail

cat > /app/solve_kmt2d.py <<'PYEOF'
import json
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

DATA = '/app/data'
LOF_CLASSES = {'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'Splice_Site'}
CHECKPOINTS = ['CD274', 'CTLA4', 'CD8A']
SIGNATURES = {
    'T_effector': ['GZMA', 'GZMB', 'PRF1', 'IFNG', 'CXCL9', 'CXCL10'],
    'IFN_gamma': ['IDO1', 'STAT1', 'HLA-DRA', 'CXCL11', 'IFNG'],
}
CELL_TYPES = {
    'CD8_T_cell': ['CD8A', 'CD8B', 'GZMK'],
    'NK_cell': ['NKG7', 'KLRD1', 'GNLY'],
    'Macrophage': ['CD68', 'CD163', 'CSF1R'],
    'Treg': ['FOXP3', 'IL2RA', 'IKZF2'],
}


def mwu(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    delta = float(a.mean() - b.mean())
    p = float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue)
    return delta, p


def main():
    expr = pd.read_csv(DATA + '/expression_matrix.tsv', sep='\t', index_col=0)
    mut = pd.read_csv(DATA + '/kmt2d_mutations.tsv', sep='\t')
    clin = pd.read_csv(DATA + '/clinical.tsv', sep='\t')
    drug = pd.read_csv(DATA + '/cell_line_drug_screen.tsv', sep='\t')

    lof_s = set(mut.loc[mut.variant_classification.isin(LOF_CLASSES), 'sample_id'])
    samples = list(expr.columns)
    lof = [s for s in samples if s in lof_s]
    wt = [s for s in samples if s not in lof_s]
    expr = expr.astype(float)

    def gene_cmp(g):
        return mwu(expr.loc[g, lof].values, expr.loc[g, wt].values)

    z = expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1, ddof=0), axis=0)

    def sig_cmp(genes):
        score = z.loc[genes].mean(axis=0)
        return mwu(score[lof].values, score[wt].values)

    res = {'n_lof': len(lof), 'n_wt': len(wt),
           'immune_markers': {}, 'signatures': {}, 'deconvolution': {}}
    for g in CHECKPOINTS:
        d, p = gene_cmp(g)
        res['immune_markers'][g] = {'delta': d, 'p_value': p}
    for name, genes in SIGNATURES.items():
        d, p = sig_cmp(genes)
        res['signatures'][name] = {'delta': d, 'p_value': p}
    for name, genes in CELL_TYPES.items():
        d, p = sig_cmp(genes)
        res['deconvolution'][name] = {'delta': d, 'p_value': p}

    ici = clin[clin.ici_treated == 'yes'].copy()
    ici['grp'] = np.where(ici.sample_id.isin(lof_s), 'LOF', 'WT')
    ici['resp'] = ici.response.isin(['CR', 'PR'])
    a = int(((ici.grp == 'LOF') & ici.resp).sum())
    b = int(((ici.grp == 'LOF') & ~ici.resp).sum())
    c = int(((ici.grp == 'WT') & ici.resp).sum())
    d2 = int(((ici.grp == 'WT') & ~ici.resp).sum())
    odds, p_fish = stats.fisher_exact([[a, b], [c, d2]])
    res['ici_response'] = {
        'odds_ratio': float(odds),
        'p_value': float(p_fish),
        'response_rate_lof': a / (a + b),
        'response_rate_wt': c / (c + d2),
    }

    t1 = ici.loc[ici.grp == 'LOF', 'os_months']
    e1 = ici.loc[ici.grp == 'LOF', 'os_event']
    t2 = ici.loc[ici.grp == 'WT', 'os_months']
    e2 = ici.loc[ici.grp == 'WT', 'os_event']
    lr = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
    dd = pd.DataFrame({'T': ici.os_months.values,
                       'E': ici.os_event.values,
                       'lof': (ici.grp == 'LOF').astype(int).values})
    cph = CoxPHFitter().fit(dd, 'T', 'E')
    res['survival'] = {'logrank_p': float(lr.p_value),
                       'hazard_ratio': float(np.exp(cph.params_['lof']))}

    a_auc = drug.loc[drug.kmt2d_status == 'LOF', 'cisplatin_auc'].values
    b_auc = drug.loc[drug.kmt2d_status == 'WT', 'cisplatin_auc'].values
    d_auc, p_auc = mwu(a_auc, b_auc)
    res['drug_sensitivity'] = {'delta': d_auc, 'p_value': p_auc}

    with open('/app/results.json', 'w') as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
PYEOF

python /app/solve_kmt2d.py
