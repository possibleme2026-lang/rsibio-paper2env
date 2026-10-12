#!/usr/bin/env python3
'''Hidden reference implementation: regenerates the cohort from a protected seed
and recomputes every graded quantity.  Only present during verification.'''

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260803
MODELS = ('MOC1', 'MOC2', 'MLM1')
N_BY_MODEL = {'MOC1': 40, 'MOC2': 41, 'MLM1': 40}
TIME_DAYS = (0, 3, 7, 11, 14)
VOL_COLS = {d: 'day{}_volume_mm3'.format(d) for d in TIME_DAYS}
MODEL_LOG_HAZARD = {'MOC1': 0.0, 'MOC2': 0.45, 'MLM1': -0.30}


def generate_tables(seed=SEED):
    rng = np.random.default_rng(seed)
    cohort_rows, pet_rows = [], []
    for model in MODELS:
        for idx in range(N_BY_MODEL[model]):
            mouse_id = '{}-{:02d}'.format(model, idx + 1)

            v0 = float(np.clip(rng.lognormal(4.15, 0.22), 50.0, 150.0))
            u = float(rng.uniform())
            shrink = 0.02 + 0.13 * u + float(rng.normal(0.0, 0.006))
            grow = 0.085 - 0.075 * u + float(rng.normal(0.0, 0.008))
            if model == 'MOC2':
                shrink -= 0.012
                grow += 0.022
            elif model == 'MLM1':
                shrink += 0.015
                grow -= 0.016

            vol = {0: v0}
            vol[3] = vol[0] * math.exp(-3.0 * shrink)
            vol[7] = vol[3] * math.exp(4.0 * grow)
            vol[11] = vol[3] * math.exp(8.0 * grow)
            vol[14] = vol[3] * math.exp(11.0 * grow)
            vol = {d: float(x * math.exp(float(rng.normal(0.0, 0.03))))
                   for d, x in vol.items()}

            roi_volume = float(vol[14] * math.exp(float(rng.normal(0.0, 0.05))))
            log_roi = math.log(roi_volume)

            body_weight = float(np.clip(rng.normal(27.0, 2.5), 20.0, 35.0))
            dose = float(np.clip(rng.normal(9.5, 1.0), 6.0, 13.0))
            uptake = float(np.clip(rng.normal(60.0, 4.0), 45.0, 75.0))

            suv_max_true = float(math.exp(1.0
                                          + 0.45 * (log_roi - math.log(160.0))
                                          + 0.22 * float(rng.normal())
                                          + float(rng.normal(0.0, 0.06))))
            suv_mean_true = float(np.clip(0.55 * suv_max_true
                                          + float(rng.normal(0.0, 0.35)),
                                          0.35, 0.95 * suv_max_true))

            conc_max = suv_max_true * dose * 1000.0 / body_weight
            conc_mean = suv_mean_true * dose * 1000.0 / body_weight

            lp = (0.70 * (log_roi - math.log(160.0))
                  + 0.35 * (suv_max_true - 2.7)
                  + MODEL_LOG_HAZARD[model]
                  + float(rng.normal(0.0, 0.20)))
            scale = 38.0 * math.exp(-lp / 1.5)
            t_event = float(rng.weibull(1.5)) * scale
            t_censor = float(rng.uniform(60.0, 100.0))
            observed = min(t_event, t_censor)
            event = 1 if t_event <= t_censor else 0

            row = {'mouse_id': mouse_id, 'model': model,
                   'treatment': 'cisplatin_5mgkg+RT_8Gy'}
            for d in TIME_DAYS:
                row[VOL_COLS[d]] = round(vol[d], 2)
            row['survival_days'] = round(observed, 1)
            row['event'] = int(event)
            cohort_rows.append(row)

            pet_rows.append({
                'mouse_id': mouse_id,
                'scan_day': 14,
                'body_weight_g': round(body_weight, 2),
                'injected_dose_mbq': round(dose, 3),
                'uptake_time_min': round(uptake, 1),
                'roi_volume_mm3': round(roi_volume, 2),
                'roi_activity_mean_kbq_ml': round(conc_mean, 2),
                'roi_activity_max_kbq_ml': round(conc_max, 2),
            })
    return pd.DataFrame(cohort_rows), pd.DataFrame(pet_rows)


def prepare_analysis_frame(cohort, pet):
    df = cohort.merge(pet, on='mouse_id', how='inner')
    df['suv_max'] = (df['roi_activity_max_kbq_ml'] * df['body_weight_g']
                     / (df['injected_dose_mbq'] * 1000.0))
    df['suv_mean'] = (df['roi_activity_mean_kbq_ml'] * df['body_weight_g']
                      / (df['injected_dose_mbq'] * 1000.0))
    df['relative_volume_day11'] = df['day11_volume_mm3'] / df['day0_volume_mm3']
    df['responder'] = (df['relative_volume_day11'] < 1.0).astype(int)
    return df


def build_long(df):
    long = df.melt(id_vars=['mouse_id', 'model', 'responder'],
                   value_vars=[VOL_COLS[d] for d in TIME_DAYS],
                   var_name='day_col', value_name='volume_mm3')
    long['day'] = [int(c.split('_')[0][3:]) for c in long['day_col']]
    long['log_volume'] = np.log(long['volume_mm3'])
    return long


def _fit_mixed(formula, data):
    import statsmodels.formula.api as smf
    last_error = None
    for method in ('bfgs', 'lbfgs', 'powell'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = smf.mixedlm(formula, data, groups=data['mouse_id'])
                res = model.fit(reml=False, method=method)
            if np.isfinite(res.llf):
                return res
        except Exception as exc:
            last_error = exc
    raise RuntimeError('mixed model did not converge: {}'.format(last_error))


def _wald_entry(cph, covariate):
    row = cph.summary.loc[covariate]
    coef = float(row['coef'])
    se = float(row['se(coef)'])
    z = float(row['z'])
    return {
        'coef': coef,
        'se': se,
        'hr': math.exp(coef),
        'hr_ci_low': float(row['exp(coef) lower 95%']),
        'hr_ci_high': float(row['exp(coef) upper 95%']),
        'z': z,
        'p': float(row['p']),
    }


def _null_partial_loglik(times, events):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    order = np.argsort(times)
    t = times[order]
    e = events[order]
    total = 0.0
    for i in range(len(t)):
        if e[i] == 1:
            total -= math.log(float(np.sum(t >= t[i])))
    return total


def compute_gold():
    cohort, pet = generate_tables()
    df = prepare_analysis_frame(cohort, pet)
    long = build_long(df)

    gold = {}
    gold['n_mice'] = int(df.shape[0])
    gold['n_by_model'] = {m: int((df['model'] == m).sum()) for m in MODELS}
    n_resp = int(df['responder'].sum())
    gold['n_responders'] = n_resp
    gold['n_non_responders'] = int(df.shape[0] - n_resp)
    gold['median_relative_volume_day11'] = {
        'responder': float(df.loc[df['responder'] == 1, 'relative_volume_day11'].median()),
        'non_responder': float(df.loc[df['responder'] == 0, 'relative_volume_day11'].median()),
    }

    full = _fit_mixed('log_volume ~ C(model) * day', long)
    null = _fit_mixed('log_volume ~ C(model) + day', long)
    lr_model = 2.0 * float(full.llf - null.llf)

    full_r = _fit_mixed('log_volume ~ responder * day', long)
    null_r = _fit_mixed('log_volume ~ responder + day', long)
    lr_resp = 2.0 * float(full_r.llf - null_r.llf)

    gold['growth_model'] = {
        'model_by_day': {'lr_stat': lr_model, 'df': 2,
                         'p': float(stats.chi2.sf(lr_model, 2))},
        'response_by_day': {'lr_stat': lr_resp, 'df': 1,
                            'p': float(stats.chi2.sf(lr_resp, 1))},
    }

    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test

    p_models = float(multivariate_logrank_test(
        df['survival_days'], df['model'], df['event']).p_value)
    resp = df[df['responder'] == 1]
    non = df[df['responder'] == 0]
    p_groups = float(logrank_test(resp['survival_days'], non['survival_days'],
                                  resp['event'], non['event']).p_value)

    medians = {}
    for m in MODELS:
        sub = df[df['model'] == m]
        kmf = KaplanMeierFitter().fit(sub['survival_days'], sub['event'])
        med = float(kmf.median_survival_time_)
        if not np.isfinite(med):
            med = float(sub['survival_days'].max())
        medians[m] = med

    gold['survival'] = {
        'logrank_p_across_models': p_models,
        'logrank_p_response_groups': p_groups,
        'median_survival_days': medians,
    }

    uni = {}
    for cov in ('suv_max', 'suv_mean', 'roi_volume_mm3'):
        cph = CoxPHFitter()
        cph.fit(df[['survival_days', 'event', cov]],
                duration_col='survival_days', event_col='event')
        uni[cov] = _wald_entry(cph, cov)
    gold['cox_univariable'] = uni

    multi = CoxPHFitter()
    multi.fit(df[['survival_days', 'event', 'suv_max', 'roi_volume_mm3']],
              duration_col='survival_days', event_col='event')
    ll_full = float(multi.log_likelihood_)
    ll_null = float(_null_partial_loglik(df['survival_days'].to_numpy(),
                                         df['event'].to_numpy()))
    lr_stat = 2.0 * (ll_full - ll_null)
    gold['cox_multivariable'] = {
        'suv_max': _wald_entry(multi, 'suv_max'),
        'roi_volume_mm3': _wald_entry(multi, 'roi_volume_mm3'),
        'll_null': ll_null,
        'll_full': ll_full,
        'lr_stat': lr_stat,
        'lr_df': 2,
        'lr_p': float(stats.chi2.sf(lr_stat, 2)),
        'concordance_index': float(multi.concordance_index_),
    }

    gold['mannwhitney_p'] = {
        'suv_max': float(stats.mannwhitneyu(resp['suv_max'], non['suv_max'],
                                            alternative='two-sided').pvalue),
        'suv_mean': float(stats.mannwhitneyu(resp['suv_mean'], non['suv_mean'],
                                             alternative='two-sided').pvalue),
    }
    return gold


if __name__ == '__main__':
    import json
    print(json.dumps(compute_gold(), indent=2, default=float))
