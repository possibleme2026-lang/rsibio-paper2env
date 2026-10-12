#!/usr/bin/env python3
"""Grader for the scGEM construction-strategy benchmark task.

Recomputes the whole pipeline from /app/data with the reference implementation in
/tests/gold_gen.py and scores the agent output on seven independently checked
constraints. Partial credit is granted per constraint.
"""
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, '/tests')
import gold_gen  # noqa: E402

DATA = '/app/data'
OUT = '/app/output'
LOGDIR = '/logs/verifier'

WEIGHTS = {'C1': 0.10, 'C2': 0.28, 'C3': 0.10, 'C4': 0.10,
           'C5': 0.08, 'C6': 0.22, 'C7': 0.12}
TOL_METRIC = 0.02
TOL_SIZE = 0.5
JACC_PASS = 0.98


def jacc_rows(A, B):
    inter = np.logical_and(A, B).sum(axis=1)
    union = np.logical_or(A, B).sum(axis=1)
    out = np.ones(len(inter))
    nz = union > 0
    out[nz] = inter[nz] / union[nz]
    return out


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    report = {'constraints': {}, 'details': {}}

    gold = gold_gen.compute_gold(DATA)
    cells = gold['cells']
    rids = gold['reactions']
    const_mask = gold['const_mask']
    sids = [str(s) for s in gold['metrics']['strategy_id']]
    gmet = gold['metrics'].set_index('strategy_id')

    # ---------------- C1: output format ----------------
    agent_models = {}
    fmt_errors = {}
    for sid in sids:
        path = os.path.join(OUT, 'models', sid + '.tsv')
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            df = pd.read_csv(path, sep='\t', index_col=0)
            if [str(c) for c in df.index] != cells:
                raise ValueError('cell ids / row order mismatch')
            if set(str(c) for c in df.columns) != set(rids):
                raise ValueError('reaction id set mismatch')
            df = df[[r for r in rids]]
            vals = df.values.astype(float)
            if not np.isin(vals, [0.0, 1.0]).all():
                raise ValueError('matrix is not binary')
            agent_models[sid] = vals.astype(int)
        except Exception as exc:
            fmt_errors[sid] = type(exc).__name__ + ': ' + str(exc)
    c1 = len(agent_models) / float(len(sids))
    report['details']['model_file_errors'] = fmt_errors

    # ---------------- C2: per-cell agreement with gold models ----------------
    per_strategy = {}
    hits = 0
    for sid in sids:
        if sid not in agent_models:
            per_strategy[sid] = 0.0
            continue
        a = agent_models[sid].astype(bool)
        g = gold['models'][sid].astype(bool)
        mean_j = float(np.mean(jacc_rows(a, g)))
        per_strategy[sid] = round(mean_j, 6)
        if mean_j >= JACC_PASS:
            hits += 1
    c2 = hits / float(len(sids))
    report['details']['mean_jaccard_vs_gold'] = per_strategy

    # ---------------- metrics table ----------------
    metrics = None
    metrics_err = None
    try:
        metrics = pd.read_csv(os.path.join(OUT, 'strategy_metrics.tsv'), sep='\t')
        metrics['strategy_id'] = metrics['strategy_id'].astype(str)
        metrics = metrics.set_index('strategy_id')
        for col in ['preprocessing', 'mem', 'param', 'mean_model_size', 'accuracy',
                    'robustness', 'specificity', 'composite']:
            if col not in metrics.columns:
                raise ValueError('missing column ' + col)
        missing = [s for s in sids if s not in metrics.index]
        if missing:
            raise ValueError('missing strategies ' + ','.join(missing))
    except Exception as exc:
        metrics_err = type(exc).__name__ + ': ' + str(exc)
        metrics = None
    report['details']['metrics_table_error'] = metrics_err

    def cmp_col(col, tol):
        if metrics is None:
            return 0.0
        good = 0
        for sid in sids:
            try:
                v = float(metrics.loc[sid, col])
                gv = float(gmet.loc[sid, col])
                if abs(v - gv) <= tol:
                    good += 1
            except Exception:
                pass
        return good / float(len(sids))

    c3 = 0.5 * cmp_col('accuracy', TOL_METRIC) + 0.5 * cmp_col('mean_model_size', TOL_SIZE)
    c4 = cmp_col('robustness', TOL_METRIC)
    c5 = cmp_col('specificity', TOL_METRIC)

    # ---------------- C6: strategy selection ----------------
    best = None
    comp = None
    sel_err = None
    try:
        with open(os.path.join(OUT, 'selection.json')) as fh:
            sel = json.load(fh)
        best = str(sel['best_strategy'])
        comp = float(sel['composite'])
    except Exception as exc:
        sel_err = type(exc).__name__ + ': ' + str(exc)
    report['details']['selection_error'] = sel_err
    id_ok = 1.0 if best == gold['best'] else 0.0
    gold_comp = float(gmet.loc[gold['best'], 'composite'])
    comp_ok = 1.0 if (comp is not None and abs(comp - gold_comp) <= 3 * TOL_METRIC) else 0.0
    c6 = 0.70 * id_ok + 0.30 * comp_ok

    # ---------------- C7: mandatory reactions / non-degenerate models ----------------
    ok = 0
    total = 0
    for sid, arr in agent_models.items():
        for i in range(arr.shape[0]):
            total += 1
            row = arr[i]
            if not row[const_mask].all():
                continue
            if row.sum() < 5:
                continue
            ok += 1
    c7 = (ok / float(total)) if total else 0.0

    parts = {'C1': c1, 'C2': c2, 'C3': c3, 'C4': c4, 'C5': c5, 'C6': c6, 'C7': c7}
    score = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)
    score = round(float(min(max(score, 0.0), 1.0)), 4)

    report['constraints'] = {
        k: {'weight': WEIGHTS[k], 'fraction': round(parts[k], 4),
            'points': round(WEIGHTS[k] * parts[k], 4)} for k in WEIGHTS}
    report['score'] = score
    report['gold_best_strategy'] = gold['best']
    report['agent_best_strategy'] = best

    with open(os.path.join(LOGDIR, 'report.json'), 'w') as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(LOGDIR, 'reward.json'), 'w') as fh:
        json.dump({'score': score, 'reward': score}, fh)

    print('=== scGEM strategy benchmark - grader ===')
    for k in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        print('%s weight=%.2f fraction=%.4f points=%.4f'
              % (k, WEIGHTS[k], parts[k], WEIGHTS[k] * parts[k]))
    print('gold best strategy = %s ; agent best strategy = %s' % (gold['best'], best))
    print('FINAL score = %.4f' % score)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        os.makedirs(LOGDIR, exist_ok=True)
        with open(os.path.join(LOGDIR, 'reward.json'), 'w') as fh:
            json.dump({'score': 0.0, 'reward': 0.0}, fh)
    sys.exit(0)
