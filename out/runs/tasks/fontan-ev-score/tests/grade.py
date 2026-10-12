#!/usr/bin/env python3
'''Deep verifier for the Fontan EV Score (FES) reproduction task.

The grader re-implements the reference pipeline from the raw cohort in
/app/data and compares every agent artefact in /app/outputs against it.
Partial credit per constraint; the reward file holds numbers only.
'''

import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.miscmodels.ordinal_model import OrderedModel

DATA = Path('/app/data')
OUT = Path('/app/outputs')
LOGDIR = Path('/logs/verifier')

FILTER_MIN_COUNT = 10
FILTER_MIN_FRACTION = 0.30
N_PANEL = 11
C_GRID = [0.5, 0.05, 0.005, 0.0005]
TOL_INDEX = 1e-4
TOL_EXPR = 1e-5
TOL_SELF = 1e-3
TOL_COEF = 0.05
TOL_COEF_LOOSE = 0.20
TOL_METRIC = 0.02
TOL_THRESHOLD = 0.40

WEIGHTS = {
    'apri': 0.07, 'fib4': 0.07, 'expression': 0.12, 'panel': 0.12,
    'model': 0.12, 'predictions': 0.08, 'self_consistency': 0.12,
    'auc_moderate': 0.10, 'auc_severe': 0.10, 'metrics': 0.10,
}


def _read_csv_safe(path, **kw):
    try:
        return pd.read_csv(path, **kw), None
    except Exception as exc:
        return None, '%s: %s' % (type(exc).__name__, exc)


def build_reference():
    clin = pd.read_csv(DATA / 'clinical.csv')
    counts = pd.read_csv(DATA / 'mirna_counts.csv', index_col=0)
    samples = [str(s) for s in clin['sample_id'].tolist()]
    counts = counts.loc[clin['sample_id']]
    n = len(samples)

    ref = {'clin': clin, 'samples': samples, 'n': n}
    ref['apri'] = ((clin['ast_u_l'] / 40.0) / clin['platelets_10e9_l'] * 100.0).values
    ref['fib4'] = ((clin['age_years'] * clin['ast_u_l']) /
                   (clin['platelets_10e9_l'] * np.sqrt(clin['alt_u_l']))).values

    keep = (counts >= FILTER_MIN_COUNT).sum(axis=0) >= FILTER_MIN_FRACTION * n
    kept = counts.loc[:, keep]
    cpm = kept.div(kept.sum(axis=1), axis=0) * 1e6
    logcpm = np.log2(cpm + 1.0)
    ref['logcpm'] = logcpm
    ref['kept_mirnas'] = sorted([str(c) for c in kept.columns])

    train = (clin['split'] == 'train').values
    test = (clin['split'] == 'test').values
    grade = clin['fibrosis_grade'].values.astype(int)
    ref['train'] = train
    ref['test'] = test
    ref['grade'] = grade

    mu = logcpm.loc[train].mean(axis=0)
    sd = logcpm.loc[train].std(axis=0, ddof=1)
    Z = (logcpm.loc[train] - mu) / sd
    ybin = (grade[train] >= 2).astype(int)

    panel, c_used, ranked, nz = None, None, None, 0
    for C in C_GRID:
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=C,
                                 max_iter=10000, random_state=0)
        clf.fit(Z.values, ybin)
        cm = {str(k): float(v) for k, v in zip(list(Z.columns), clf.coef_[0])}
        ranked = sorted(cm.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
        nz = sum(1 for v in cm.values() if v != 0.0)
        if nz >= N_PANEL:
            panel = sorted([k for k, _ in ranked[:N_PANEL]])
            c_used = C
            break
    if panel is None:
        panel = sorted([k for k, _ in ranked[:N_PANEL]])
        c_used = C_GRID[-1]
    ref['panel'] = panel
    ref['c_used'] = c_used
    ref['n_nonzero'] = nz

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
    ref['feats'] = feats
    ref['coefs'] = coefs
    ref['thresholds'] = thr
    ref['feat_matrix'] = X

    eta = X[feats].values @ np.array([coefs[f] for f in feats])
    ref['p_mod'] = 1.0 - expit(thr[1] - eta)
    ref['p_sev'] = 1.0 - expit(thr[2] - eta)
    return ref


def check_index(ref, name, filename, tags):
    df, err = _read_csv_safe(OUT / filename)
    if df is None:
        return 0.0, '%s unreadable (%s)' % (filename, err)
    cols = {str(c) for c in df.columns}
    if 'sample_id' not in cols or name not in cols:
        return 0.0, '%s missing column %s' % (filename, name)
    ids = [str(s) for s in df['sample_id'].tolist()]
    order_ok = ids == ref['samples']
    d = df.assign(sample_id=ids).set_index('sample_id')
    vals = pd.to_numeric(d.reindex(ref['samples'])[name], errors='coerce').values
    refv = ref[name]
    ok = np.isfinite(vals) & (np.abs(vals - refv) <= TOL_INDEX * np.maximum(1.0, np.abs(refv)))
    score = float(ok.mean())
    if not order_ok:
        score *= 0.75
    return score, '%d/%d rows within tolerance (row order %s)' % (
        int(ok.sum()), len(ok), 'ok' if order_ok else 'wrong')


def check_expression(ref):
    df, err = _read_csv_safe(OUT / 'normalized_expression.csv', index_col=0)
    if df is None:
        return 0.0, 'normalized_expression.csv unreadable (%s)' % err
    df = df.rename(columns=str, index=str)
    refm = ref['logcpm'].rename(columns=str)
    agent_cols = set(df.columns)
    ref_cols = set(refm.columns)
    inter = agent_cols & ref_cols
    union = agent_cols | ref_cols
    jac = len(inter) / max(1, len(union))
    rows = [s for s in ref['samples'] if s in set(df.index)]
    if not rows or not inter:
        return 0.0, 'no overlapping rows/columns with the reference matrix'
    cols = sorted(inter)
    a = df.loc[rows, cols].apply(pd.to_numeric, errors='coerce').values
    b = refm.loc[rows, cols].values
    cell = float((np.abs(a - b) <= TOL_EXPR).mean())
    rowfrac = len(rows) / len(ref['samples'])
    score = 0.45 * jac * rowfrac + 0.55 * cell
    return score, 'miRNA-set Jaccard=%.3f, %d/%d rows, cells within tol=%.3f' % (
        jac, len(rows), len(ref['samples']), cell)


def check_panel(ref):
    try:
        obj = json.loads((OUT / 'panel.json').read_text())
    except Exception as exc:
        return 0.0, 'panel.json unreadable (%s)' % exc
    panel = obj.get('panel')
    if not isinstance(panel, list) or not panel:
        return 0.0, 'panel.json has no panel list'
    panel = [str(p) for p in panel]
    uniq = set(panel)
    refset = set(ref['panel'])
    overlap = len(uniq & refset)
    base = overlap / len(refset)
    shape_ok = (len(panel) == N_PANEL and len(uniq) == N_PANEL)
    if not shape_ok:
        base *= 0.5
    c_ok = False
    try:
        c_ok = abs(float(obj.get('c_used')) - ref['c_used']) <= 1e-12
    except Exception:
        c_ok = False
    score = 0.9 * base + 0.1 * (1.0 if c_ok else 0.0)
    return score, '%d/%d reference miRNAs reproduced, exactly-11=%s, c_used %s' % (
        overlap, len(refset), shape_ok, 'ok' if c_ok else 'missing/wrong')


def check_model(ref):
    try:
        obj = json.loads((OUT / 'model.json').read_text())
        coefs = {str(k): float(v) for k, v in obj['coefficients'].items()}
        thr = [float(v) for v in obj['thresholds']]
    except Exception as exc:
        return 0.0, 'model.json unreadable (%s)' % exc
    per = []
    matched = 0
    for f, v in ref['coefs'].items():
        if f not in coefs:
            per.append(0.0)
            continue
        d = abs(coefs[f] - v)
        if d <= TOL_COEF:
            per.append(1.0)
            matched += 1
        elif d <= TOL_COEF_LOOSE:
            per.append(0.5)
            matched += 1
        else:
            per.append(0.0)
    coef_score = float(np.mean(per))
    extra = len(set(coefs.keys()) - set(ref['feats']))
    if extra:
        coef_score *= max(0.5, 1.0 - 0.1 * extra)
    thr_ok = (len(thr) == 3 and all(abs(a - b) <= TOL_THRESHOLD
                                    for a, b in zip(sorted(thr), ref['thresholds'])))
    score = 0.85 * coef_score + 0.15 * (1.0 if thr_ok else 0.0)
    return score, '%d/%d coefficients close, %d extra features, thresholds %s' % (
        matched, len(ref['feats']), extra, 'ok' if thr_ok else 'off')


def _agent_predictions(ref):
    df, err = _read_csv_safe(OUT / 'predictions.csv')
    if df is None:
        return None, None, None, 'predictions.csv unreadable (%s)' % err
    cols = {str(c) for c in df.columns}
    if not {'sample_id', 'fes_moderate', 'fes_severe'}.issubset(cols):
        return None, None, None, 'predictions.csv missing required columns'
    ids = [str(s) for s in df['sample_id'].tolist()]
    d = df.assign(sample_id=ids).set_index('sample_id').reindex(ref['samples'])
    pm = pd.to_numeric(d['fes_moderate'], errors='coerce').values.astype(float)
    ps = pd.to_numeric(d['fes_severe'], errors='coerce').values.astype(float)
    return pm, ps, ids, None


def check_predictions(ref):
    pm, ps, ids, err = _agent_predictions(ref)
    if pm is None:
        return 0.0, err
    present = float(np.isin(ref['samples'], ids).mean())
    finite = np.isfinite(pm) & np.isfinite(ps)
    rng_ok = finite & (pm >= 0) & (pm <= 1) & (ps >= 0) & (ps <= 1)
    mono = finite & (ps <= pm + 1e-9)
    score = 0.35 * present + 0.35 * float(rng_ok.mean()) + 0.30 * float(mono.mean())
    return score, 'coverage %.2f, in [0,1] %.2f, monotone %.2f' % (
        present, float(rng_ok.mean()), float(mono.mean()))


def check_self_consistency(ref):
    pm, ps, ids, err = _agent_predictions(ref)
    if pm is None:
        return 0.0, err
    dfe, e1 = _read_csv_safe(OUT / 'normalized_expression.csv', index_col=0)
    try:
        obj = json.loads((OUT / 'model.json').read_text())
        coefs = {str(k): float(v) for k, v in obj['coefficients'].items()}
        thr = sorted([float(v) for v in obj['thresholds']])
    except Exception as exc:
        return 0.0, 'model.json unreadable (%s)' % exc
    if dfe is None:
        return 0.0, 'normalized_expression.csv unreadable (%s)' % e1
    dfe = dfe.rename(columns=str, index=str)
    if len(thr) != 3 or 'log1p_months' not in coefs:
        return 0.0, 'model.json lacks thresholds or log1p_months coefficient'
    feats = [f for f in coefs if f != 'log1p_months']
    missing = [f for f in feats if f not in dfe.columns]
    if missing:
        return 0.0, 'model features absent from expression matrix: %s' % missing[:3]
    months = dict(zip([str(s) for s in ref['clin']['sample_id']],
                      ref['clin']['months_post_surgery']))
    rows = [s for s in ref['samples'] if s in set(dfe.index)]
    if not rows:
        return 0.0, 'no shared samples between predictions and expression matrix'
    eta = np.array([coefs['log1p_months'] * math.log1p(float(months[s])) for s in rows])
    for f in feats:
        v = pd.to_numeric(dfe.loc[rows, f], errors='coerce').values.astype(float)
        if not np.isfinite(v).all():
            return 0.0, 'non-numeric expression values for %s' % f
        eta = eta + coefs[f] * v
    ref_mod = 1.0 - expit(thr[1] - eta)
    ref_sev = 1.0 - expit(thr[2] - eta)
    pos = {s: i for i, s in enumerate(ref['samples'])}
    idx = [pos[s] for s in rows]
    am, ass = pm[idx], ps[idx]
    if not np.isfinite(am).all() or not np.isfinite(ass).all():
        return 0.0, 'non-numeric predictions for shared samples'
    dev = np.maximum(np.abs(am - ref_mod), np.abs(ass - ref_sev))
    ok = dev <= TOL_SELF
    return float(ok.mean()), '%d/%d samples reproduce their own model within %g' % (
        int(ok.sum()), len(ok), TOL_SELF)


def _band(a, full, mid, low):
    if not np.isfinite(a):
        return 0.0
    if a >= full:
        return 1.0
    if a >= mid:
        return 0.7
    if a >= low:
        return 0.4
    return 0.0


def check_auc(ref):
    pm, ps, ids, err = _agent_predictions(ref)
    if pm is None:
        return 0.0, 0.0, err
    test = ref['test']
    y = ref['grade']
    try:
        a_mod = float(roc_auc_score((y[test] >= 2).astype(int), pm[test]))
        a_sev = float(roc_auc_score((y[test] >= 3).astype(int), ps[test]))
    except Exception as exc:
        return 0.0, 0.0, 'AUC not computable (%s)' % exc
    return (_band(a_mod, 0.80, 0.72, 0.62),
            _band(a_sev, 0.85, 0.75, 0.65),
            'FES AUC moderate=%.3f severe=%.3f' % (a_mod, a_sev))


def check_metrics(ref):
    try:
        obj = json.loads((OUT / 'metrics.json').read_text())
    except Exception as exc:
        return 0.0, 'metrics.json unreadable (%s)' % exc
    test = ref['test']
    y = ref['grade']
    expected = {}
    pm, ps, ids, err = _agent_predictions(ref)
    if pm is not None:
        try:
            expected['auc_moderate_fes'] = float(roc_auc_score((y[test] >= 2).astype(int), pm[test]))
            expected['auc_severe_fes'] = float(roc_auc_score((y[test] >= 3).astype(int), ps[test]))
        except Exception:
            pass
    for tag, arr in (('apri', ref['apri']), ('fib4', ref['fib4'])):
        expected['auc_moderate_%s' % tag] = float(roc_auc_score((y[test] >= 2).astype(int), arr[test]))
        expected['auc_severe_%s' % tag] = float(roc_auc_score((y[test] >= 3).astype(int), arr[test]))
    hits = 0
    for k, v in expected.items():
        try:
            hits += int(abs(float(obj[k]) - v) <= TOL_METRIC)
        except Exception:
            pass
    score = hits / float(len(expected))
    return score, '%d/%d reported metrics match recomputation' % (hits, len(expected))


def main():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    (LOGDIR / 'reward.json').write_text(json.dumps({'score': 0.0, 'reward': 0.0}))
    report = {'constraints': {}, 'notes': []}
    scores = {k: 0.0 for k in WEIGHTS}
    details = {k: 'not evaluated' for k in WEIGHTS}

    ref = None
    try:
        ref = build_reference()
    except Exception:
        report['notes'].append('reference pipeline failed: ' + traceback.format_exc())

    if ref is not None:
        s, d = check_index(ref, 'apri', 'clinical_indices.csv', None)
        scores['apri'], details['apri'] = s, d
        s, d = check_index(ref, 'fib4', 'clinical_indices.csv', None)
        scores['fib4'], details['fib4'] = s, d
        s, d = check_expression(ref)
        scores['expression'], details['expression'] = s, d
        s, d = check_panel(ref)
        scores['panel'], details['panel'] = s, d
        s, d = check_model(ref)
        scores['model'], details['model'] = s, d
        s, d = check_predictions(ref)
        scores['predictions'], details['predictions'] = s, d
        s, d = check_self_consistency(ref)
        scores['self_consistency'], details['self_consistency'] = s, d
        am, asv, d = check_auc(ref)
        scores['auc_moderate'], details['auc_moderate'] = am, d
        scores['auc_severe'], details['auc_severe'] = asv, d
        s, d = check_metrics(ref)
        scores['metrics'], details['metrics'] = s, d
        report['reference_summary'] = {
            'n_samples': ref['n'],
            'n_retained_mirnas': len(ref['kept_mirnas']),
            'n_train': int(ref['train'].sum()),
            'n_test': int(ref['test'].sum()),
            'c_used': ref['c_used'],
            'n_nonzero_coef': ref['n_nonzero'],
            'panel': ref['panel'],
        }

    total = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    total = float(max(0.0, min(1.0, total)))
    for k in WEIGHTS:
        report['constraints'][k] = {
            'score': round(float(scores[k]), 4),
            'weight': WEIGHTS[k],
            'detail': details[k],
        }
    report['score'] = round(total, 4)
    report['reward'] = round(total, 4)
    print(json.dumps(report, indent=2, sort_keys=True))
    (LOGDIR / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True))
    (LOGDIR / 'reward.json').write_text(json.dumps({'score': round(total, 4),
                                                    'reward': round(total, 4)}))


if __name__ == '__main__':
    main()
