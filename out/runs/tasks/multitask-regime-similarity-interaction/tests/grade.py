#!/usr/bin/env python3
"""Grader for the multi-task training-regime / structural-similarity re-analysis.

Recomputes the reference answer from a protected copy of the data generator and
scores ten independent constraints with partial credit.
"""
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gold_gen

AMBIENT_CSV = '/app/data/behavior.csv'
RESULTS_JSON = '/app/results.json'
OUT_DIR = '/logs/verifier'

CELL_KEYS = ['%s|%s|%s' % (o, s, r)
             for o in gold_gen.ORDERS
             for s in gold_gen.SIMILARITIES
             for r in gold_gen.REGIMES]

TOL_MEAN = 0.005
TOL_DELTA = 0.010
TOL_SELF = 0.005
N_MIN_POSTTEST = 30


def reference_analysis(df):
    """Independent re-implementation of the requested pipeline."""
    prac = df[df['block'] == 'practice']
    prac_acc = prac.groupby('participant_id')['correct'].mean()
    retained = set(prac_acc.index[prac_acc >= 0.60])

    d = df[df['participant_id'].isin(retained)].copy()
    d = d[(d['rt_ms'] >= 150.0) & (d['rt_ms'] <= 4000.0)].copy()
    d = d[d['trial_in_block'] > 5].copy()

    post = d[d['block'] == 'posttest']
    counts = post.groupby(['participant_id', 'task'])['correct'].size().unstack(fill_value=0)
    for tk in ['conceptual', 'spatial']:
        if tk not in counts.columns:
            counts[tk] = 0
    valid = set(counts.index[(counts[['conceptual', 'spatial']] >= N_MIN_POSTTEST).all(axis=1)])

    d = d[d['participant_id'].isin(valid)].copy()
    post = d[d['block'] == 'posttest']

    meta = d.groupby('participant_id')[['order', 'similarity', 'regime']].first()
    meta['first_task'] = np.where(meta['order'] == 'conceptual_first', 'conceptual', 'spatial')
    meta['second_task'] = np.where(meta['first_task'] == 'conceptual', 'spatial', 'conceptual')
    meta['cell'] = meta['order'] + '|' + meta['similarity'] + '|' + meta['regime']

    acc = post.groupby(['participant_id', 'task'])['correct'].mean().unstack()
    meta['acc_second'] = [float(acc.at[p, t]) for p, t in zip(meta.index, meta['second_task'])]
    meta['acc_first'] = [float(acc.at[p, t]) for p, t in zip(meta.index, meta['first_task'])]

    m2 = meta.groupby('cell')['acc_second'].mean()
    m1 = meta.groupby('cell')['acc_first'].mean()
    n_cell = meta.groupby('cell').size()

    def delta(m, order):
        return float(m['%s|similar|blocked' % order] - m['%s|distinct|blocked' % order]
                     - m['%s|similar|interleaved' % order]
                     + m['%s|distinct|interleaved' % order])

    dc = delta(m2, 'conceptual_first')
    ds = delta(m2, 'spatial_first')

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

    supported = bool(abs(dc) >= 0.10 and p_c < 0.01 and abs(ds) < 0.5 * abs(dc))

    return {
        'n_participants_retained': int(len(retained)),
        'n_participants_analyzed': int(meta.shape[0]),
        'n_trials_analyzed': int(d.shape[0]),
        'n_participants_per_cell': {k: int(n_cell.get(k, 0)) for k in CELL_KEYS},
        'second_task_posttest_accuracy': {k: float(m2[k]) for k in CELL_KEYS},
        'first_task_posttest_accuracy': {k: float(m1[k]) for k in CELL_KEYS},
        'interaction_conceptual_first': dc,
        'interaction_spatial_first': ds,
        'interaction_three_way_difference': float(dc - ds),
        'p_interaction_conceptual_first': p_c,
        'p_interaction_spatial_first': p_s,
        'asymmetric_interaction_supported': supported,
    }


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_float(res, key):
    if not isinstance(res, dict) or key not in res:
        raise KeyError(key)
    v = res[key]
    if not _num(v):
        raise TypeError(key)
    return float(v)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    checks = []

    def add(name, ok, detail=''):
        checks.append({'name': name, 'passed': bool(ok), 'detail': str(detail)})

    gold_df = gold_gen.generate()
    gold = reference_analysis(gold_df)

    # --- data integrity -------------------------------------------------
    integrity = False
    try:
        amb = pd.read_csv(AMBIENT_CSV)
        cols = ['participant_id', 'block', 'task', 'trial_in_block', 'correct', 'rt_ms']
        a = amb[cols].sort_values(cols).reset_index(drop=True)
        b = gold_df[cols].sort_values(cols).reset_index(drop=True)
        integrity = bool(
            a.shape == b.shape
            and a['participant_id'].equals(b['participant_id'])
            and a['block'].equals(b['block'])
            and a['task'].equals(b['task'])
            and a['trial_in_block'].equals(b['trial_in_block'])
            and (a['correct'].to_numpy() == b['correct'].to_numpy()).all()
            and np.allclose(a['rt_ms'].to_numpy(), b['rt_ms'].to_numpy(), atol=1e-6)
        )
    except Exception:
        integrity = False
    add('data_integrity', integrity, 'input CSV identical to the reference generator output')

    # --- agent results --------------------------------------------------
    try:
        with open(RESULTS_JSON) as fh:
            res = json.load(fh)
    except Exception:
        res = None

    schema_ok = False
    if isinstance(res, dict):
        required = ['n_participants_retained', 'n_participants_analyzed', 'n_trials_analyzed',
                    'n_participants_per_cell', 'second_task_posttest_accuracy',
                    'first_task_posttest_accuracy', 'interaction_conceptual_first',
                    'interaction_spatial_first', 'interaction_three_way_difference',
                    'p_interaction_conceptual_first', 'p_interaction_spatial_first',
                    'asymmetric_interaction_supported']
        schema_ok = all(k in res for k in required)
        if schema_ok:
            for k in ['n_participants_per_cell', 'second_task_posttest_accuracy',
                      'first_task_posttest_accuracy']:
                v = res[k]
                schema_ok = (schema_ok and isinstance(v, dict)
                             and set(v.keys()) == set(CELL_KEYS)
                             and all(_num(x) for x in v.values()))
            for k in ['n_participants_retained', 'n_participants_analyzed', 'n_trials_analyzed',
                      'interaction_conceptual_first', 'interaction_spatial_first',
                      'interaction_three_way_difference', 'p_interaction_conceptual_first',
                      'p_interaction_spatial_first']:
                schema_ok = schema_ok and _num(res.get(k))
            schema_ok = schema_ok and isinstance(res.get('asymmetric_interaction_supported'), bool)
    add('results_schema', schema_ok, 'results.json present, all keys, correct types')

    # --- counts ---------------------------------------------------------
    counts_ok = False
    try:
        counts_ok = (int(res['n_participants_retained']) == gold['n_participants_retained']
                     and int(res['n_participants_analyzed']) == gold['n_participants_analyzed']
                     and int(res['n_trials_analyzed']) == gold['n_trials_analyzed'])
    except Exception:
        counts_ok = False
    add('exclusion_counts', counts_ok,
        'reference retained=%d analyzed=%d trials=%d' % (
            gold['n_participants_retained'], gold['n_participants_analyzed'],
            gold['n_trials_analyzed']))

    ncell_ok = False
    try:
        ncell_ok = all(int(res['n_participants_per_cell'][k]) == gold['n_participants_per_cell'][k]
                       for k in CELL_KEYS)
    except Exception:
        ncell_ok = False
    add('cell_counts', ncell_ok, 'per-cell participant counts identical to the reference')

    # --- cell means -----------------------------------------------------
    def means_ok(key):
        try:
            return all(abs(float(res[key][k]) - gold[key][k]) <= TOL_MEAN for k in CELL_KEYS)
        except Exception:
            return False
    add('second_task_means', means_ok('second_task_posttest_accuracy'),
        'eight second-task cell means within %.3f' % TOL_MEAN)
    add('first_task_means', means_ok('first_task_posttest_accuracy'),
        'eight first-task cell means within %.3f' % TOL_MEAN)

    # --- interaction contrasts -----------------------------------------
    inter_ok = False
    try:
        inter_ok = (abs(_as_float(res, 'interaction_conceptual_first')
                        - gold['interaction_conceptual_first']) <= TOL_DELTA
                    and abs(_as_float(res, 'interaction_spatial_first')
                            - gold['interaction_spatial_first']) <= TOL_DELTA
                    and abs(_as_float(res, 'interaction_three_way_difference')
                            - gold['interaction_three_way_difference']) <= 2 * TOL_DELTA)
    except Exception:
        inter_ok = False
    add('interactions', inter_ok,
        'reference delta_c=%.4f delta_s=%.4f delta_diff=%.4f' % (
            gold['interaction_conceptual_first'], gold['interaction_spatial_first'],
            gold['interaction_three_way_difference']))

    # --- p-values --------------------------------------------------------
    p_ok = False
    try:
        pc = _as_float(res, 'p_interaction_conceptual_first')
        ps = _as_float(res, 'p_interaction_spatial_first')
        p_ok = (0.0 <= pc <= 1.0) and (0.0 <= ps <= 1.0) and pc < 0.01 and pc <= ps
    except Exception:
        p_ok = False
    add('p_values', p_ok,
        'reference p_c=%.3g p_s=%.3f' % (gold['p_interaction_conceptual_first'],
                                         gold['p_interaction_spatial_first']))

    # --- decision flag ---------------------------------------------------
    flag_ok = False
    try:
        dc_r = _as_float(res, 'interaction_conceptual_first')
        ds_r = _as_float(res, 'interaction_spatial_first')
        pc_r = _as_float(res, 'p_interaction_conceptual_first')
        expected = bool(abs(dc_r) >= 0.10 and pc_r < 0.01 and abs(ds_r) < 0.5 * abs(dc_r))
        reported = res['asymmetric_interaction_supported']
        flag_ok = (isinstance(reported, bool) and reported == expected
                   and reported == gold['asymmetric_interaction_supported'])
    except Exception:
        flag_ok = False
    add('decision_flag', flag_ok, 'reference flag=%s' % gold['asymmetric_interaction_supported'])

    # --- internal consistency of the reported numbers --------------------
    cons_ok = False
    try:
        m = res['second_task_posttest_accuracy']

        def dlt(o):
            return (float(m['%s|similar|blocked' % o]) - float(m['%s|distinct|blocked' % o])
                    - float(m['%s|similar|interleaved' % o])
                    + float(m['%s|distinct|interleaved' % o]))
        dc_self = dlt('conceptual_first')
        ds_self = dlt('spatial_first')
        dc_r = _as_float(res, 'interaction_conceptual_first')
        ds_r = _as_float(res, 'interaction_spatial_first')
        dd_r = _as_float(res, 'interaction_three_way_difference')
        cons_ok = (abs(dc_self - dc_r) <= TOL_SELF and abs(ds_self - ds_r) <= TOL_SELF
                   and abs((dc_r - ds_r) - dd_r) <= TOL_SELF)
    except Exception:
        cons_ok = False
    add('internal_consistency', cons_ok,
        'reported contrasts reproduce the reported cell means')

    # --- score -----------------------------------------------------------
    n_total = len(checks)
    n_pass = sum(1 for c in checks if c['passed'])
    score = round(float(n_pass) / float(n_total), 4)

    with open(os.path.join(OUT_DIR, 'reward.json'), 'w') as fh:
        json.dump({'score': score, 'reward': score}, fh)
    with open(os.path.join(OUT_DIR, 'report.json'), 'w') as fh:
        json.dump({'score': score, 'checks': checks, 'gold': gold}, fh, indent=2, default=str)

    print('=== multi-task regime x similarity analysis : %d/%d checks passed ===' % (n_pass, n_total))
    for c in checks:
        print('[%s] %-22s %s' % ('PASS' if c['passed'] else 'FAIL', c['name'], c['detail']))
    print('score = %.4f' % score)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(os.path.join(OUT_DIR, 'reward.json'), 'w') as fh:
                json.dump({'score': 0.0, 'reward': 0.0}, fh)
        except Exception:
            pass
        sys.exit(0)
