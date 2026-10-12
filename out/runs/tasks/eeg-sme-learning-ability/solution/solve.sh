#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

df = pd.read_csv('/app/data/encoding_events.csv')
df['serial_position_sq'] = df['serial_position'] ** 2

formula = 'recalled ~ theta_power + alpha_power + gamma_power + serial_position + serial_position_sq'
model = smf.logit(formula, data=df)
res = model.fit(disp=False, cov_type='cluster', cov_kwds={'groups': df['subject_id']})

group_level = {}
for var in ['theta_power', 'alpha_power', 'gamma_power', 'serial_position', 'serial_position_sq']:
    group_level[var] = {
        'coef': round(float(res.params[var]), 6),
        'se': round(float(res.bse[var]), 6),
        'z': round(float(res.tvalues[var]), 4),
        'p': round(float(res.pvalues[var]), 6)
    }

per_subject = []
for subj in sorted(df['subject_id'].unique()):
    sub_df = df[df['subject_id'] == subj].copy()
    m = smf.logit(formula, data=sub_df)
    r = m.fit(disp=False)
    per_subject.append({
        'subject_id': int(subj),
        'theta': round(float(r.params['theta_power']), 6),
        'alpha': round(float(r.params['alpha_power']), 6),
        'gamma': round(float(r.params['gamma_power']), 6)
    })

acc = df.groupby('subject_id')['recalled'].mean().to_dict()
theta_sme = np.array([abs(p['theta']) for p in per_subject])
alpha_sme = np.array([abs(p['alpha']) for p in per_subject])
gamma_sme = np.array([abs(p['gamma']) for p in per_subject])
accs = np.array([acc[p['subject_id']] for p in per_subject])

r_theta, p_theta = pearsonr(accs, theta_sme)
r_alpha, p_alpha = pearsonr(accs, alpha_sme)
r_gamma, p_gamma = pearsonr(accs, gamma_sme)

correlations = {
    'theta': {'r': round(float(r_theta), 6), 'p': round(float(p_theta), 6)},
    'alpha': {'r': round(float(r_alpha), 6), 'p': round(float(p_alpha), 6)},
    'gamma': {'r': round(float(r_gamma), 6), 'p': round(float(p_gamma), 6)}
}

result = {
    'group_level': group_level,
    'per_subject': per_subject,
    'correlations': correlations
}

with open('/app/output/results.json', 'w') as f:
    json.dump(result, f, indent=2)
PY
