#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python3 - <<'PY'
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pro = pd.read_csv('/app/data/proteome_npx.csv')
meta = pd.read_csv('/app/data/sample_metadata.csv')
clin = pd.read_csv('/app/data/clinical_train.csv')
proteins = [c for c in pro.columns if c != 'sample_id']
X = pro.set_index('sample_id')[proteins].astype(float)

# QC: drop proteins with more than 20% missing, median-impute the rest
miss_frac = X.isna().mean()
keep = [c for c in proteins if miss_frac[c] <= 0.20]
X = X[keep].fillna(X[keep].median())

# per-plate offset correction (plate is listed in the metadata)
plate = meta.set_index('sample_id')['plate']
for c in keep:
    X[c] = X[c] - X[c].groupby(plate).transform('mean') + X[c].mean()

meta = meta.set_index('sample_id')
train_ids = [s for s in X.index if meta.loc[s, 'cohort'] == 'train']
test_ids = [s for s in X.index if meta.loc[s, 'cohort'] == 'test']
clin = clin.set_index('sample_id')
y = clin.loc[train_ids, 'diagnosis'].map({'MPN': 1, 'non-MPN': 0}).values
Xt = X.loc[train_ids].values
Xs = X.loc[test_ids].values

def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / float(rank + 1))
        q[i] = prev
    return np.clip(q, 0.0, 1.0)

# differential abundance in the training cohort
rows = []
for j, c in enumerate(keep):
    a, b = Xt[y == 1, j], Xt[y == 0, j]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    rows.append((c, float(a.mean() - b.mean()), float(p)))
qvals = bh_adjust([r[2] for r in rows])
de = pd.DataFrame({'protein': [r[0] for r in rows], 'log2_fc': [r[1] for r in rows],
                   'p_value': [r[2] for r in rows], 'q_value': qvals})
de.to_csv('/app/output/differential_expression.csv', index=False)

# Lasso logistic regression, penalty chosen by 5-fold CV on ROC-AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
best_c, best_auc = None, -1.0
for C in [0.02, 0.05, 0.1, 0.2, 0.5]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear',
                                                             C=C, max_iter=5000))
    auc = cross_val_score(clf, Xt, y, cv=cv, scoring='roc_auc').mean()
    if auc > best_auc:
        best_auc, best_c = float(auc), C
clf = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear',
                                                         C=best_c, max_iter=5000))
clf.fit(Xt, y)
coefs = dict(zip(keep, clf.named_steps['logisticregression'].coef_[0]))
ranked = sorted(coefs, key=lambda k: -abs(coefs[k]))
sel = [c for c in ranked if abs(coefs[c]) > 1e-8][:12]
if len(sel) < 3:
    sel = ranked[:3]
pred = clf.predict_proba(Xs)[:, 1]
with open('/app/output/predictions.csv', 'w') as fh:
    fh.write('sample_id,mpn_probability\n')
    for sid, pr in zip(test_ids, pred):
        fh.write('%s,%.6f\n' % (sid, pr))
with open('/app/output/panel.json', 'w') as fh:
    json.dump({'selected_proteins': sel, 'cv_auc': best_auc, 'n_cv_folds': 5,
               'model': 'lasso_logistic', 'C': best_c}, fh)

# PV vs non-PV within the MPN training subjects
mpn_train = [s for s in train_ids if clin.loc[s, 'diagnosis'] == 'MPN']
sublabels = clin.loc[mpn_train, 'mpn_subtype']
pv = (sublabels == 'PV').values
Xp = X.loc[mpn_train].values
prows = []
for j, c in enumerate(keep):
    a, b = Xp[pv, j], Xp[~pv, j]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    prows.append((c, float(a.mean() - b.mean()), float(p)))
pq = bh_adjust([r[2] for r in prows])
pd.DataFrame({'protein': [r[0] for r in prows], 'log2_fc': [r[1] for r in prows],
              'q_value': pq}).to_csv('/app/output/pv_markers.csv', index=False)

tfrc = X.loc[mpn_train, 'TFRC'].values
pos, neg = tfrc[pv], tfrc[~pv]
tfrc_auc = float(np.mean([(1.0 if a > b else 0.5 if a == b else 0.0)
                          for a in pos for b in neg]))
il6_row = de[de['protein'] == 'IL6'].iloc[0]
report = {'n_de_significant': int((de['q_value'] < 0.05).sum()),
          'tfrc_pv_auc': tfrc_auc,
          'il6_q_mpn_vs_nonmpn': float(il6_row['q_value']),
          'il6_log2_fc': float(il6_row['log2_fc']),
          'n_train_samples': len(train_ids),
          'n_test_samples': len(test_ids),
          'n_proteins_analyzed': len(keep),
          'panel_size': len(sel),
          'lasso_cv_auc': best_auc}
with open('/app/output/report.json', 'w') as fh:
    json.dump(report, fh, indent=2)
print(json.dumps(report, indent=2))
PY
