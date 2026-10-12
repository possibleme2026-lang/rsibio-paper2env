#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import json
import math

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

DAYS = (0, 3, 7, 11, 14)
Z95 = 1.959963984540054

cohort = pd.read_csv('/app/data/mouse_cohort.csv')
pet = pd.read_csv('/app/data/pet_ct_roi.csv')
df = cohort.merge(pet, on='mouse_id', how='inner')
df['suv_max'] = df['roi_activity_max_kbq_ml'] * df['body_weight_g'] / (df['injected_dose_mbq'] * 1000.0)
df['suv_mean'] = df['roi_activity_mean_kbq_ml'] * df['body_weight_g'] / (df['injected_dose_mbq'] * 1000.0)
df['rel11'] = df['day11_volume_mm3'] / df['day0_volume_mm3']
df['responder'] = (df['rel11'] < 1.0).astype(int)

long = df.melt(id_vars=['mouse_id', 'model', 'responder'],
               value_vars=['day{}_volume_mm3'.format(d) for d in DAYS],
               var_name='day_col', value_name='volume_mm3')
long['day'] = [int(c.split('_')[0][3:]) for c in long['day_col']]
long['log_volume'] = np.log(long['volume_mm3'])


def mix(formula):
    return smf.mixedlm(formula, long, groups=long['mouse_id']).fit(reml=False)


m_full = mix('log_volume ~ C(model) * day')
m_null = mix('log_volume ~ C(model) + day')
lr_model = 2.0 * (m_full.llf - m_null.llf)
r_full = mix('log_volume ~ responder * day')
r_null = mix('log_volume ~ responder + day')
lr_resp = 2.0 * (r_full.llf - r_null.llf)

p_models = multivariate_logrank_test(df['survival_days'], df['model'], df['event']).p_value
resp = df[df['responder'] == 1]
non = df[df['responder'] == 0]
p_groups = logrank_test(resp['survival_days'], non['survival_days'],
                        resp['event'], non['event']).p_value

medians = {}
for m in ('MOC1', 'MOC2', 'MLM1'):
    sub = df[df['model'] == m]
    kmf = KaplanMeierFitter().fit(sub['survival_days'], sub['event'])
    med = float(kmf.median_survival_time_)
    medians[m] = med if np.isfinite(med) else float(sub['survival_days'].max())


def wald(cph, cov):
    row = cph.summary.loc[cov]
    c = float(row['coef'])
    s = float(row['se(coef)'])
    return {'coef': c, 'se': s, 'hr': math.exp(c),
            'hr_ci_low': math.exp(c - Z95 * s),
            'hr_ci_high': math.exp(c + Z95 * s),
            'z': c / s, 'p': 2.0 * float(stats.norm.sf(abs(c / s)))}


uni = {}
for cov in ('suv_max', 'suv_mean', 'roi_volume_mm3'):
    cph = CoxPHFitter().fit(df[['survival_days', 'event', cov]], 'survival_days', 'event')
    uni[cov] = wald(cph, cov)

multi = CoxPHFitter().fit(df[['survival_days', 'event', 'suv_max', 'roi_volume_mm3']],
                          'survival_days', 'event')
times = df['survival_days'].to_numpy()
events = df['event'].to_numpy()
order = np.argsort(times)
t_sorted = times[order]
e_sorted = events[order]
ll_null = -sum(math.log(float((t_sorted >= t_sorted[i]).sum()))
               for i in range(len(t_sorted)) if e_sorted[i] == 1)
ll_full = float(multi.log_likelihood_)
lr = 2.0 * (ll_full - ll_null)

out = {
    'n_mice': int(df.shape[0]),
    'n_by_model': {m: int((df['model'] == m).sum()) for m in ('MOC1', 'MOC2', 'MLM1')},
    'n_responders': int(df['responder'].sum()),
    'n_non_responders': int((df['responder'] == 0).sum()),
    'median_relative_volume_day11': {
        'responder': float(resp['rel11'].median()),
        'non_responder': float(non['rel11'].median())},
    'growth_model': {
        'model_by_day': {'lr_stat': float(lr_model), 'df': 2,
                         'p': float(stats.chi2.sf(lr_model, 2))},
        'response_by_day': {'lr_stat': float(lr_resp), 'df': 1,
                            'p': float(stats.chi2.sf(lr_resp, 1))}},
    'survival': {
        'logrank_p_across_models': float(p_models),
        'logrank_p_response_groups': float(p_groups),
        'median_survival_days': medians},
    'cox_univariable': uni,
    'cox_multivariable': {
        'suv_max': wald(multi, 'suv_max'),
        'roi_volume_mm3': wald(multi, 'roi_volume_mm3'),
        'll_null': float(ll_null),
        'll_full': ll_full,
        'lr_stat': float(lr),
        'lr_df': 2,
        'lr_p': float(stats.chi2.sf(lr, 2)),
        'concordance_index': float(multi.concordance_index_)},
    'mannwhitney_p': {
        'suv_max': float(stats.mannwhitneyu(resp['suv_max'], non['suv_max'],
                                            alternative='two-sided').pvalue),
        'suv_mean': float(stats.mannwhitneyu(resp['suv_mean'], non['suv_mean'],
                                             alternative='two-sided').pvalue)},
}

with open('/app/output/results.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print('wrote /app/output/results.json')
PY
