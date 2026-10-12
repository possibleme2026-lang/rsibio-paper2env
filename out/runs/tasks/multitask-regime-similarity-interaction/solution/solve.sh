#!/bin/bash
set -euo pipefail

cat > /tmp/analysis.py <<'PY'
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/app/data/behavior.csv')

# ---- E1: practice accuracy on the raw practice rows --------------------------
prac = df[df['block'] == 'practice']
prac_acc = prac.groupby('participant_id')['correct'].mean()
retained = set(prac_acc.index[prac_acc >= 0.60])
d = df[df['participant_id'].isin(retained)].copy()

# ---- E2 / E3 ----------------------------------------------------------------
d = d[(d['rt_ms'] >= 150.0) & (d['rt_ms'] <= 4000.0)].copy()
d = d[d['trial_in_block'] > 5].copy()

# ---- E4: at least 30 usable post-test trials on both tasks ------------------
post = d[d['block'] == 'posttest']
counts = post.groupby(['participant_id', 'task'])['correct'].size().unstack(fill_value=0)
valid = set(counts.index[(counts[['conceptual', 'spatial']] >= 30).all(axis=1)])
d = d[d['participant_id'].isin(valid)].copy()
post = d[d['block'] == 'posttest']

# ---- participant-level post-test accuracy -----------------------------------
meta = d.groupby('participant_id')[['order', 'similarity', 'regime']].first()
meta['first_task'] = np.where(meta['order'] == 'conceptual_first', 'conceptual', 'spatial')
meta['second_task'] = np.where(meta['first_task'] == 'conceptual', 'spatial', 'conceptual')
meta['cell'] = meta['order'] + '|' + meta['similarity'] + '|' + meta['regime']

acc = post.groupby(['participant_id', 'task'])['correct'].mean().unstack()
meta['acc_second'] = [float(acc.at[p, t]) for p, t in zip(meta.index, meta['second_task'])]
meta['acc_first'] = [float(acc.at[p, t]) for p, t in zip(meta.index, meta['first_task'])]

keys = [o + '|' + s + '|' + r for o in ['conceptual_first', 'spatial_first']
        for s in ['similar', 'distinct'] for r in ['blocked', 'interleaved']]
m2 = meta.groupby('cell')['acc_second'].mean()
m1 = meta.groupby('cell')['acc_first'].mean()
ncell = meta.groupby('cell').size()


def delta(m, order):
    return float(m[order + '|similar|blocked'] - m[order + '|distinct|blocked']
                 - m[order + '|similar|interleaved'] + m[order + '|distinct|interleaved'])


dc = delta(m2, 'conceptual_first')
ds = delta(m2, 'spatial_first')

# ---- saturated OLS on participant-level second-task accuracy ----------------
y = meta['acc_second'].to_numpy(dtype=float)
o = (meta['order'] == 'spatial_first').to_numpy(dtype=float)
s = (meta['similarity'] == 'similar').to_numpy(dtype=float)
r = (meta['regime'] == 'interleaved').to_numpy(dtype=float)
X = np.column_stack([np.ones_like(o), o, s, r, o * s, o * r, s * r, o * s * r])
fit = sm.OLS(y, X).fit()
L_c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
L_s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0])
p_c = float(np.ravel(np.asarray(fit.t_test(L_c).pvalue))[0])
p_s = float(np.ravel(np.asarray(fit.t_test(L_s).pvalue))[0])

out = {
    'n_participants_retained': int(len(retained)),
    'n_participants_analyzed': int(meta.shape[0]),
    'n_trials_analyzed': int(d.shape[0]),
    'n_participants_per_cell': {k: int(ncell[k]) for k in keys},
    'second_task_posttest_accuracy': {k: float(m2[k]) for k in keys},
    'first_task_posttest_accuracy': {k: float(m1[k]) for k in keys},
    'interaction_conceptual_first': dc,
    'interaction_spatial_first': ds,
    'interaction_three_way_difference': float(dc - ds),
    'p_interaction_conceptual_first': p_c,
    'p_interaction_spatial_first': p_s,
    'asymmetric_interaction_supported': bool(abs(dc) >= 0.10 and p_c < 0.01
                                             and abs(ds) < 0.5 * abs(dc)),
}

with open('/app/results.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
PY

python /tmp/analysis.py
