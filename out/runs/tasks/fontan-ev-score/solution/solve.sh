#!/bin/bash
set -euo pipefail
mkdir -p /app/outputs
python3 - <<'PY'
import json
import math

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.miscmodels.ordinal_model import OrderedModel

DATA = '/app/data'
OUT = '/app/outputs'

clin = pd.read_csv(DATA + '/clinical.csv')
counts = pd.read_csv(DATA + '/mirna_counts.csv', index_col=0).loc[clin['sample_id']]
n = len(clin)
samples = clin['sample_id'].tolist()

# 1. clinical indices
apri = (clin['ast_u_l'] / 40.0) / clin['platelets_10e9_l'] * 100.0
fib4 = (clin['age_years'] * clin['ast_u_l']) / (clin['platelets_10e9_l'] * np.sqrt(clin['alt_u_l']))
pd.DataFrame({'sample_id': samples, 'apri': np.round(apri.values, 6),
              'fib4': np.round(fib4.values, 6)}).to_csv(OUT + '/clinical_indices.csv', index=False)

# 2. filtering + CPM + log2
keep = (counts >= 10).sum(axis=0) >= 0.30 * n
kept = counts.loc[:, keep]
cpm = kept.div(kept.sum(axis=1), axis=0) * 1e6
logcpm = np.log2(cpm + 1.0)
logcpm.round(6).to_csv(OUT + '/normalized_expression.csv')
train = (clin['split'] == 'train').values
grade = clin['fibrosis_grade'].values.astype(int)

# 3. L1 feature selection on the training split
mu = logcpm.loc[train].mean(axis=0)
sd = logcpm.loc[train].std(axis=0, ddof=1)
Z = (logcpm.loc[train] - mu) / sd
ybin = (grade[train] >= 2).astype(int)
panel, c_used, nz = None, None, 0
ranked = []
for C in [0.5, 0.05, 0.005, 0.0005]:
    clf = LogisticRegression(penalty='l1', solver='liblinear', C=C,
                             max_iter=10000, random_state=0)
    clf.fit(Z.values, ybin)
    cm = {str(k): float(v) for k, v in zip(list(Z.columns), clf.coef_[0])}
    ranked = sorted(cm.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    nz = sum(1 for v in cm.values() if v != 0.0)
    if nz >= 11:
        panel = sorted([k for k, _ in ranked[:11]])
        c_used = C
        break
if panel is None:
    panel = sorted([k for k, _ in ranked[:11]])
    c_used = 0.0005
json.dump({'panel': panel, 'c_used': c_used, 'n_nonzero': nz},
          open(OUT + '/panel.json', 'w'), indent=2)

# 4. ordinal proportional-odds model on the training split
feats = ['log1p_months'] + panel
X = pd.DataFrame(index=clin.index)
X['log1p_months'] = np.log1p(clin['months_post_surgery'].values.astype(float))
for m in panel:
    X[m] = logcpm[m].values
res = OrderedModel(grade[train], X.loc[train].values, distr='logit').fit(
    method='bfgs', disp=False, maxiter=2000)
k = len(feats)
coefs = {f: float(res.params[i]) for i, f in enumerate(feats)}
thr = sorted([float(v) for v in res.params[k:k + 3]])
json.dump({'coefficients': coefs, 'thresholds': thr},
          open(OUT + '/model.json', 'w'), indent=2)

# 5. FES probabilities for every sample
eta = X[feats].values @ np.array([coefs[f] for f in feats])
p_mod = 1.0 - expit(thr[1] - eta)
p_sev = 1.0 - expit(thr[2] - eta)
pd.DataFrame({'sample_id': samples, 'split': clin['split'].values,
              'fes_moderate': np.round(p_mod, 6),
              'fes_severe': np.round(p_sev, 6)}).to_csv(OUT + '/predictions.csv', index=False)

# 6. held-out benchmark
test = (clin['split'] == 'test').values
y_mod = (grade[test] >= 2).astype(int)
y_sev = (grade[test] >= 3).astype(int)
metrics = {
    'test_n': int(test.sum()),
    'auc_moderate_fes': float(roc_auc_score(y_mod, p_mod[test])),
    'auc_severe_fes': float(roc_auc_score(y_sev, p_sev[test])),
    'auc_moderate_apri': float(roc_auc_score(y_mod, apri.values[test])),
    'auc_severe_apri': float(roc_auc_score(y_sev, apri.values[test])),
    'auc_moderate_fib4': float(roc_auc_score(y_mod, fib4.values[test])),
    'auc_severe_fib4': float(roc_auc_score(y_sev, fib4.values[test])),
}
json.dump(metrics, open(OUT + '/metrics.json', 'w'), indent=2)
print(json.dumps(metrics, indent=2))
PY
