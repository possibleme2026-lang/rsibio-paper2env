#!/usr/bin/env python3
'''Deep, partial-credit grader for the HNSCC 18F-FDG PET/CT chemoradiation task.'''

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gold_gen

OUT_PATH = Path('/app/output/results.json')
REWARD_PATH = Path('/logs/verifier/reward.json')
REPORT_PATH = Path('/logs/verifier/report.json')
Z95 = 1.959963984540054

CHECK_WEIGHTS = [
    ('schema', 0.05),
    ('cohort_counts', 0.08),
    ('response_medians', 0.07),
    ('growth_model_lrt', 0.10),
    ('response_growth_lrt', 0.08),
    ('logrank_models', 0.08),
    ('logrank_response_groups', 0.07),
    ('median_survival', 0.07),
    ('cox_univariable', 0.15),
    ('cox_multivariable', 0.15),
    ('concordance_index', 0.05),
    ('mannwhitney', 0.05),
]


def is_num(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(float(x)))


def tol_score(value, target, rel=0.05, abs_tol=1e-09):
    if not is_num(value) or not is_num(target):
        return 0.0
    v = float(value)
    t = float(target)
    tol = max(rel * abs(t), abs_tol)
    err = abs(v - t)
    if err <= tol:
        return 1.0
    hard = 4.0 * tol
    if err >= hard:
        return 0.0
    return 1.0 - (err - tol) / (hard - tol)


def p_score(value, target):
    if not is_num(value) or not is_num(target):
        return 0.0
    v = float(value)
    if v < 0.0 or v > 1.0:
        return 0.0
    t = max(float(target), 1e-12)
    if t >= 0.01:
        return tol_score(v, t, rel=0.15, abs_tol=0.005)
    d = abs(math.log10(max(v, 1e-12)) - math.log10(t))
    if d <= 0.2:
        return 1.0
    if d >= 0.8:
        return 0.0
    return 1.0 - (d - 0.2) / 0.6


def get(obj, *path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def check_schema(raw):
    required = [
        ('n_mice',), ('n_responders',), ('n_non_responders',),
        ('n_by_model', 'MOC1'), ('n_by_model', 'MOC2'), ('n_by_model', 'MLM1'),
        ('median_relative_volume_day11', 'responder'),
        ('median_relative_volume_day11', 'non_responder'),
        ('growth_model', 'model_by_day', 'lr_stat'),
        ('growth_model', 'model_by_day', 'df'),
        ('growth_model', 'model_by_day', 'p'),
        ('growth_model', 'response_by_day', 'lr_stat'),
        ('growth_model', 'response_by_day', 'df'),
        ('growth_model', 'response_by_day', 'p'),
        ('survival', 'logrank_p_across_models'),
        ('survival', 'logrank_p_response_groups'),
        ('survival', 'median_survival_days', 'MOC1'),
        ('survival', 'median_survival_days', 'MOC2'),
        ('survival', 'median_survival_days', 'MLM1'),
        ('cox_multivariable', 'll_null'),
        ('cox_multivariable', 'll_full'),
        ('cox_multivariable', 'lr_stat'),
        ('cox_multivariable', 'lr_df'),
        ('cox_multivariable', 'lr_p'),
        ('cox_multivariable', 'concordance_index'),
        ('mannwhitney_p', 'suv_max'),
        ('mannwhitney_p', 'suv_mean'),
    ]
    for cov in ('suv_max', 'suv_mean', 'roi_volume_mm3'):
        for f in ('coef', 'se', 'hr', 'hr_ci_low', 'hr_ci_high', 'z', 'p'):
            required.append(('cox_univariable', cov, f))
    for cov in ('suv_max', 'roi_volume_mm3'):
        for f in ('coef', 'se', 'hr', 'hr_ci_low', 'hr_ci_high', 'z', 'p'):
            required.append(('cox_multivariable', cov, f))
    if not isinstance(raw, dict):
        return 0.0, 'results.json missing or not a JSON object'
    missing = [p for p in required if not is_num(get(raw, *p))]
    score = (len(required) - len(missing)) / float(len(required))
    if missing:
        detail = 'missing or non-numeric: ' + ', '.join('.'.join(p) for p in missing[:8])
    else:
        detail = 'all {} numeric fields present'.format(len(required))
    return score, detail


def check_cohort_counts(raw, gold):
    parts = []
    n = get(raw, 'n_mice')
    parts.append(1.0 if is_num(n) and int(n) == gold['n_mice'] else 0.0)
    for m in gold_gen.MODELS:
        v = get(raw, 'n_by_model', m)
        parts.append(1.0 if is_num(v) and int(v) == gold['n_by_model'][m] else 0.0)
    nr = get(raw, 'n_responders')
    nn = get(raw, 'n_non_responders')
    parts.append(1.0 if is_num(nr) and int(nr) == gold['n_responders'] else 0.0)
    parts.append(1.0 if is_num(nn) and int(nn) == gold['n_non_responders'] else 0.0)
    consistent = (is_num(nr) and is_num(nn) and is_num(n)
                  and int(nr) + int(nn) == int(n))
    parts.append(1.0 if consistent else 0.0)
    detail = 'n={}, responders={}, non_responders={}, by_model={}'.format(
        n, nr, nn, get(raw, 'n_by_model'))
    return float(np.mean(parts)), detail


def check_response_medians(raw, gold):
    parts = []
    for k in ('responder', 'non_responder'):
        parts.append(tol_score(get(raw, 'median_relative_volume_day11', k),
                               gold['median_relative_volume_day11'][k],
                               rel=0.02, abs_tol=0.005))
    detail = 'gold medians: {}'.format(gold['median_relative_volume_day11'])
    return float(np.mean(parts)), detail


def check_lrt(raw, path, target, df_expected):
    block = get(raw, *path)
    if not isinstance(block, dict):
        return 0.0, 'missing block {}'.format('.'.join(path))
    lr = block.get('lr_stat')
    df = block.get('df')
    p = block.get('p')
    parts = [tol_score(lr, target['lr_stat'], rel=0.10, abs_tol=0.5)]
    parts.append(1.0 if is_num(df) and int(df) == df_expected else 0.0)
    if is_num(lr) and float(lr) >= 0.0:
        parts.append(p_score(p, float(stats.chi2.sf(float(lr), df_expected))))
    else:
        parts.append(0.0)
    parts.append(p_score(p, target['p']))
    detail = 'lr_stat={} (gold {:.3f}), df={}, p={} (gold {:.3g})'.format(
        lr, target['lr_stat'], df, p, target['p'])
    return float(np.mean(parts)), detail


def check_logrank(raw, gold, key):
    v = get(raw, 'survival', key)
    t = gold['survival'][key]
    return p_score(v, t), '{}={} (gold {:.4g})'.format(key, v, t)


def check_median_survival(raw, gold):
    parts, detail = [], []
    for m in gold_gen.MODELS:
        v = get(raw, 'survival', 'median_survival_days', m)
        t = gold['survival']['median_survival_days'][m]
        parts.append(tol_score(v, t, rel=0.05, abs_tol=3.0))
        detail.append('{}={} (gold {:.1f})'.format(m, v, t))
    return float(np.mean(parts)), '; '.join(detail)


def wald_block_score(block, target):
    if not isinstance(block, dict):
        return 0.0, 'missing cox block'
    coef = block.get('coef')
    se = block.get('se')
    z = block.get('z')
    hr = block.get('hr')
    lo = block.get('hr_ci_low')
    hi = block.get('hr_ci_high')
    p = block.get('p')
    parts = [tol_score(coef, target['coef'], rel=0.05, abs_tol=0.01),
             tol_score(se, target['se'], rel=0.10, abs_tol=0.005)]
    if is_num(coef) and is_num(se) and float(se) > 0.0:
        c = float(coef)
        s = float(se)
        parts.append(tol_score(hr, math.exp(c), rel=0.01, abs_tol=1e-04))
        parts.append(tol_score(lo, math.exp(c - Z95 * s), rel=0.02, abs_tol=1e-04))
        parts.append(tol_score(hi, math.exp(c + Z95 * s), rel=0.02, abs_tol=1e-04))
        parts.append(tol_score(z, c / s, rel=0.02, abs_tol=0.05))
        parts.append(p_score(p, 2.0 * float(stats.norm.sf(abs(c / s)))))
    else:
        parts.extend([0.0] * 5)
    ordered = (is_num(hr) and is_num(lo) and is_num(hi) and float(hr) > 0.0
               and float(lo) < float(hr) < float(hi))
    parts.append(1.0 if ordered else 0.0)
    if is_num(hr):
        parts.append(1.0 if (float(hr) > 1.0) == (target['hr'] > 1.0) else 0.0)
    else:
        parts.append(0.0)
    detail = 'coef={} vs {:.4f}; hr={} vs {:.4f}'.format(
        coef, target['coef'], hr, target['hr'])
    return float(np.mean(parts)), detail


def check_cox_univariable(raw, gold):
    parts, detail = [], []
    for cov in ('suv_max', 'suv_mean', 'roi_volume_mm3'):
        s, d = wald_block_score(get(raw, 'cox_univariable', cov),
                                gold['cox_univariable'][cov])
        parts.append(s)
        detail.append('{}: {}'.format(cov, d))
    return float(np.mean(parts)), ' | '.join(detail)


def check_cox_multivariable(raw, gold):
    g = gold['cox_multivariable']
    parts, detail = [], []
    for cov in ('suv_max', 'roi_volume_mm3'):
        s, d = wald_block_score(get(raw, 'cox_multivariable', cov), g[cov])
        parts.append(s)
        detail.append('{}: {}'.format(cov, d))
    ll_full = get(raw, 'cox_multivariable', 'll_full')
    ll_null = get(raw, 'cox_multivariable', 'll_null')
    lr = get(raw, 'cox_multivariable', 'lr_stat')
    lr_df = get(raw, 'cox_multivariable', 'lr_df')
    lr_p = get(raw, 'cox_multivariable', 'lr_p')
    parts.append(tol_score(ll_full, g['ll_full'], rel=0.0, abs_tol=2.0))
    parts.append(tol_score(ll_null, g['ll_null'], rel=0.0, abs_tol=2.0))
    parts.append(tol_score(lr, g['lr_stat'], rel=0.10, abs_tol=0.5))
    parts.append(1.0 if is_num(lr_df) and int(lr_df) == 2 else 0.0)
    if is_num(lr) and float(lr) >= 0.0:
        parts.append(p_score(lr_p, float(stats.chi2.sf(float(lr), 2))))
    else:
        parts.append(0.0)
    parts.append(p_score(lr_p, g['lr_p']))
    detail.append('lr_stat={} (gold {:.2f}); lr_p={} (gold {:.3g})'.format(
        lr, g['lr_stat'], lr_p, g['lr_p']))
    return float(np.mean(parts)), ' | '.join(detail)


def check_concordance(raw, gold):
    v = get(raw, 'cox_multivariable', 'concordance_index')
    t = gold['cox_multivariable']['concordance_index']
    return tol_score(v, t, rel=0.03, abs_tol=0.02), 'c={} (gold {:.4f})'.format(v, t)


def check_mannwhitney(raw, gold):
    parts, detail = [], []
    for k in ('suv_max', 'suv_mean'):
        v = get(raw, 'mannwhitney_p', k)
        t = gold['mannwhitney_p'][k]
        parts.append(p_score(v, t))
        detail.append('{}={} (gold {:.4g})'.format(k, v, t))
    return float(np.mean(parts)), '; '.join(detail)


def write_reward(score):
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {'score': float(score), 'reward': float(score)}
    REWARD_PATH.write_text(json.dumps(payload))


def run():
    gold = gold_gen.compute_gold()

    raw = None
    parse_error = ''
    if OUT_PATH.exists():
        try:
            raw = json.loads(OUT_PATH.read_text())
        except Exception as exc:
            parse_error = 'parse error: {}'.format(exc)
    else:
        parse_error = '{} not found'.format(OUT_PATH)

    results = {}
    results['schema'] = check_schema(raw)
    if isinstance(raw, dict):
        results['cohort_counts'] = check_cohort_counts(raw, gold)
        results['response_medians'] = check_response_medians(raw, gold)
        results['growth_model_lrt'] = check_lrt(
            raw, ('growth_model', 'model_by_day'), gold['growth_model']['model_by_day'], 2)
        results['response_growth_lrt'] = check_lrt(
            raw, ('growth_model', 'response_by_day'), gold['growth_model']['response_by_day'], 1)
        results['logrank_models'] = check_logrank(raw, gold, 'logrank_p_across_models')
        results['logrank_response_groups'] = check_logrank(raw, gold, 'logrank_p_response_groups')
        results['median_survival'] = check_median_survival(raw, gold)
        results['cox_univariable'] = check_cox_univariable(raw, gold)
        results['cox_multivariable'] = check_cox_multivariable(raw, gold)
        results['concordance_index'] = check_concordance(raw, gold)
        results['mannwhitney'] = check_mannwhitney(raw, gold)
    else:
        for name, _w in CHECK_WEIGHTS:
            results.setdefault(name, (0.0, 'no usable results.json'))

    lines = []
    total = 0.0
    report_checks = {}
    for name, weight in CHECK_WEIGHTS:
        score, detail = results.get(name, (0.0, 'not evaluated'))
        score = float(score)
        total += weight * score
        report_checks[name] = {'weight': weight, 'score': round(score, 4),
                               'detail': str(detail)}
        lines.append('{:<24} w={:.2f} score={:.3f}  {}'.format(name, weight, score, detail))
    lines.append('TOTAL SCORE = {:.4f}'.format(total))
    if parse_error:
        lines.append(parse_error)
    sys.stdout.write('\n'.join(lines) + '\n')

    report = {'score': round(total, 4), 'parse_error': parse_error,
              'checks': report_checks,
              'gold_summary': {
                  'n_mice': gold['n_mice'],
                  'n_responders': gold['n_responders'],
                  'logrank_p_across_models': gold['survival']['logrank_p_across_models'],
                  'concordance_index': gold['cox_multivariable']['concordance_index']}}
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2))
    except Exception:
        pass
    write_reward(total)


if __name__ == '__main__':
    try:
        run()
    except Exception:
        traceback.print_exc()
        try:
            write_reward(0.0)
        except Exception:
            pass
        sys.exit(0)
