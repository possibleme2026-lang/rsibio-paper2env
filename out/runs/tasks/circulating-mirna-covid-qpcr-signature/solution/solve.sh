#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python3 - <<'PY'
import json
import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests

D = '/app/data'
O = '/app/output'
SPIKES = {'cel-miR-39-3p', 'UniSp6'}

# ---- 1. plate QC -------------------------------------------------------
cq = pd.read_csv(f'{D}/plate_cq.csv')
meta = pd.read_csv(f'{D}/sample_metadata.csv')
cq['sample_type'] = cq['sample_type'].astype(str).str.strip().str.upper()
cq = cq[cq['sample_type'] == 'SAMPLE'].copy()
ct = pd.to_numeric(cq['cq'], errors='coerce')
cq['ct'] = np.where(ct > 38.0, np.nan, ct)

spk = cq[cq['mirna'].isin(SPIKES)].groupby('sample_id')['ct'].agg(['mean', 'count'])
spk = spk[spk['count'] >= 2]['mean']

agg = cq[~cq['mirna'].isin(SPIKES)].groupby(['sample_id', 'mirna'])['ct'].agg(['mean', 'count']).reset_index()
agg = agg[agg['count'] >= 1]
agg = agg.merge(spk.rename('spike_mean'), left_on='sample_id', right_index=True, how='inner')
agg['dct'] = agg['mean'] - agg['spike_mean']

n_samples = int(spk.shape[0])
detected = agg.groupby('mirna')['dct'].count()
retained = sorted(detected[detected >= 0.70 * n_samples].index)

dct = agg[agg['mirna'].isin(retained)][['sample_id', 'mirna', 'dct']]
dct = dct.merge(meta[['sample_id', 'group']], on='sample_id', how='inner')
dct = dct.sort_values(['sample_id', 'mirna']).reset_index(drop=True)
dct.to_csv(f'{O}/normalized_dct.tsv', sep='\t', index=False, float_format='%.6f')

# ---- 2. differential expression ---------------------------------------
rows = []
for m in retained:
    s = dct[dct['mirna'] == m]
    con = s[s['group'] == 'Control']['dct'].values
    cov = s[s['group'] == 'COVID-19']['dct'].values
    t, p = stats.ttest_ind(cov, con, equal_var=False)
    rows.append({'mirna': m, 'mean_dct_control': con.mean(), 'mean_dct_covid': cov.mean(),
                 'log2fc': con.mean() - cov.mean(), 't_statistic': t, 'p_value': p})
de = pd.DataFrame(rows)
de['q_value'] = multipletests(de['p_value'].values, method='fdr_bh')[1]
de['significant'] = np.where((de['q_value'] < 0.05) & (de['log2fc'] > math.log2(1.5)), 'yes', 'no')
de.to_csv(f'{O}/de_results.tsv', sep='\t', index=False, float_format='%.6g')

# ---- 3. ROC / Youden ---------------------------------------------------
roc_rows = []
for m in retained:
    s = dct[dct['mirna'] == m]
    y = (s['group'] == 'COVID-19').astype(int).values
    score = -s['dct'].values
    fpr, tpr, thr = roc_curve(y, score)
    i = int(np.argmax(tpr - fpr))
    roc_rows.append({'mirna': m, 'auc': roc_auc_score(y, score),
                     'youden_cutoff_score': thr[i], 'sensitivity': tpr[i],
                     'specificity': 1.0 - fpr[i]})
roc = pd.DataFrame(roc_rows)
roc.to_csv(f'{O}/roc_results.tsv', sep='\t', index=False, float_format='%.6f')

sig = sorted(de.loc[de['significant'] == 'yes', 'mirna'])
sig_sorted = list(roc[roc['mirna'].isin(sig)].sort_values(['auc', 'mirna'], ascending=[False, True])['mirna'])
with open(f'{O}/signature.json', 'w') as fh:
    json.dump({'signature': sig_sorted, 'n_signature': len(sig_sorted),
               'best_mirna': sig_sorted[0] if sig_sorted else None}, fh, indent=2)

# ---- 4. target network -------------------------------------------------
pairs = pd.read_csv(f'{D}/mirna_target_pairs.tsv', sep='\t')
pairs['confidence'] = pd.to_numeric(pairs['confidence'], errors='coerce')
edges = pairs[pairs['mirna'].isin(sig) & (pairs['confidence'] >= 0.75)].drop_duplicates(['mirna', 'gene'])
deg = edges.groupby('gene')['mirna'].nunique()
panel = {l.strip() for l in open(f'{D}/immune_gene_panel.txt') if l.strip()}
net = {'n_edges': int(len(edges)), 'n_target_genes': int(len(deg)),
       'gene_degrees': {g: int(d) for g, d in deg.items()},
       'top_hub_genes': [g for g, _ in sorted(deg.items(), key=lambda kv: (-kv[1], kv[0]))[:5]],
       'immune_overlap': sorted(set(deg.index) & panel)}
with open(f'{O}/network.json', 'w') as fh:
    json.dump(net, fh, indent=2)

print('signature:', sig_sorted)
print('network edges:', net['n_edges'], 'genes:', net['n_target_genes'])
PY
