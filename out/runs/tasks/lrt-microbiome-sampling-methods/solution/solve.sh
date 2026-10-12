#!/bin/bash
set -euo pipefail
mkdir -p /app/output
cat > /tmp/pipeline.py <<'PY'
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from skbio.diversity import beta_diversity
from skbio.stats.distance import DistanceMatrix, permanova
from skbio.stats.ordination import pcoa

DATA = '/app/data'
OUT = '/app/output'
SEED = 42
N_PERM = 999

asv = pd.read_csv(DATA + '/asv_table.tsv', sep='\t', index_col=0)
meta = pd.read_csv(DATA + '/metadata.tsv', sep='\t').set_index('sample_id')
tax = pd.read_csv(DATA + '/taxonomy.tsv', sep='\t', index_col=0)

ctrl = [s for s in asv.columns if meta.loc[s, 'sample_type'] == 'control']
part = [s for s in asv.columns if meta.loc[s, 'sample_type'] != 'control']
prev_c = (asv[ctrl] >= 1).mean(axis=1)
prev_p = (asv[part] >= 1).mean(axis=1)
contaminant = (prev_c >= 0.5) & (prev_p < 0.5)
keep = (~contaminant) & (asv[part].sum(axis=1) >= 10)
sub = asv.loc[keep, part].sort_index()
sub.to_csv(OUT + '/filtered_asv_table.tsv', sep='\t')

rel = sub / sub.sum(axis=0)
X = rel.T
dm = beta_diversity('braycurtis', X.values, ids=list(X.index))
pc1 = float(pcoa(dm).proportion_explained.iloc[0])

stype = meta.loc[part, 'sample_type'].values
np.random.seed(SEED)
res = permanova(dm, stype, permutations=N_PERM)
F = float(res[0]['test statistic'])
p_glob = float(res[0]['p-value'])
N = int(res[0]['sample size'])
a = len(set(stype))
R2 = F * (a - 1) / (F * (a - 1) + (N - a))

arr = np.asarray(dm)
idx = {s: i for i, s in enumerate(part)}
pairs = [('oral', 'bal'), ('oral', 'brush'), ('bal', 'brush')]
raw, keys, pwres = [], [], {}
for x, y in pairs:
    ids = [s for s in part if meta.loc[s, 'sample_type'] in (x, y)]
    sel = np.array([idx[s] for s in ids])
    np.random.seed(SEED)
    r = permanova(DistanceMatrix(arr[np.ix_(sel, sel)], ids=ids),
                  meta.loc[ids, 'sample_type'].values, permutations=N_PERM)
    key = x + '_vs_' + y
    pwres[key] = {'p_value': float(r[0]['p-value'])}
    raw.append(float(r[0]['p-value']))
    keys.append(key)
adj = multipletests(raw, method='fdr_bh')[1]
for k, key in enumerate(keys):
    pwres[key]['p_adjusted'] = float(adj[k])

Xv = X.values
groups = np.asarray(stype)
types = sorted(set(groups))
means = np.array([Xv[groups == t].mean(axis=0) for t in types])
tot = means.sum(axis=0)
tot[tot <= 0] = 1.0
obs = np.zeros(means.shape)
for k, t in enumerate(types):
    obs[k] = 100.0 * means[k] / tot * (Xv[groups == t] > 0).mean(axis=0)
rng = np.random.default_rng(SEED)
ge = np.zeros(obs.T.shape)
for _ in range(N_PERM):
    perm = rng.permutation(groups)
    pm = np.array([Xv[perm == t].mean(axis=0) for t in types])
    pt = pm.sum(axis=0)
    pt[pt <= 0] = 1.0
    mx = np.zeros(Xv.shape[1])
    for k, t in enumerate(types):
        np.maximum(mx, 100.0 * pm[k] / pt * (Xv[perm == t] > 0).mean(axis=0), out=mx)
    for k in range(len(types)):
        ge[:, k] += mx >= obs[k]
pv = ge / N_PERM
ids_all = np.array(sub.index)
sets = {}
for k, t in enumerate(types):
    mask = (obs[k] >= 40.0) & (pv[:, k] < 0.05)
    sets[t] = sorted(ids_all[mask].tolist())
fam = tax['Family']
sph = [a2 for a2 in sets['brush'] if str(fam.get(a2, '')).strip().lower() == 'sphingomonadaceae']

bal_ids = [s for s in part if meta.loc[s, 'sample_type'] == 'bal']
grp = (meta.loc[bal_ids, 'cancer_status'] == 'cancer').values
relbal = rel[bal_ids]
pvals = []
for a2 in relbal.index:
    v = relbal.loc[a2].values
    pvals.append(mannwhitneyu(v[grp], v[~grp], alternative='two-sided',
                              method='asymptotic').pvalue)
pvals = np.nan_to_num(np.array(pvals, dtype=float), nan=1.0)
n_sig = int((multipletests(pvals, method='fdr_bh')[1] < 0.05).sum())
relc = (meta.loc[part, 'cancer_status'] == 'cancer').values
np.random.seed(SEED)
resc = permanova(dm, relc, permutations=N_PERM)
p_cancer = float(resc[0]['p-value'])

report = {
    'n_asvs_retained': int(len(sub)),
    'n_contaminants_removed': int(contaminant.sum()),
    'beta_diversity': {
        'pcoa_pc1_variance_explained': round(pc1, 4),
        'permanova_sample_type': {'pseudo_f': round(F, 4), 'r2': round(R2, 4),
                                  'p_value': round(p_glob, 4)},
        'pairwise_permanova': {
            k: {'p_value': round(v['p_value'], 4), 'p_adjusted': round(v['p_adjusted'], 4)}
            for k, v in pwres.items()},
    },
    'indicator_analysis': {
        'oral': sets['oral'], 'bal': sets['bal'], 'brush': sets['brush'],
        'overlaps': {
            'oral_bal': len(set(sets['oral']) & set(sets['bal'])),
            'oral_brush': len(set(sets['oral']) & set(sets['brush'])),
            'bal_brush': len(set(sets['bal']) & set(sets['brush'])),
        },
        'brush_specific_sphingomonadaceae_asv': sph[0] if sph else None,
    },
    'cancer_status': {'permanova_p_value': round(p_cancer, 4),
                      'n_asvs_fdr_significant': n_sig},
}
with open(OUT + '/report.json', 'w') as fh:
    json.dump(report, fh, indent=2)
print(json.dumps(report)[:800])
PY
python /tmp/pipeline.py
