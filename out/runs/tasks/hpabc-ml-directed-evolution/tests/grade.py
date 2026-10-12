#!/usr/bin/env python3
'''Deep verifier for the HpaBC ML + directed-evolution design task.'''

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gold_gen  # noqa: E402

DATA_DIR = '/app/data'
OUT_DIR = '/app/output'
REWARD_PATH = '/logs/verifier/reward.json'
CONSERVED = {101, 104, 106, 109, 205, 397}
MUT_RE = re.compile(r'^([ACDEFGHIKLMNPQRSTVWY])(\d{1,4})([ACDEFGHIKLMNPQRSTVWY])$')
CHECKS = []


def add(name, score, weight, detail=''):
    CHECKS.append({'name': name, 'score': round(min(max(float(score), 0.0), 1.0), 4),
                   'weight': float(weight), 'detail': str(detail)[:400]})


def spearman(x, y):
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size < 3 or a.size != b.size:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def rmse_log(pred, gold):
    p = np.asarray([max(float(v), 1e-4) for v in pred], dtype=float)
    g = np.asarray([max(float(v), 1e-4) for v in gold], dtype=float)
    return float(math.sqrt(((np.log(p) - np.log(g)) ** 2).mean()))


def parse_muts(text, seq):
    out = []
    for tok in re.split(r'[;,]', text or ''):
        tok = tok.strip()
        if not tok:
            continue
        m = MUT_RE.match(tok)
        if not m:
            return None
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        if not 1 <= pos <= len(seq) or seq[pos - 1] != wt or wt == mut:
            return None
        out.append((pos, mut))
    return out if out else None


def read_rows(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline='') as fh:
        return [r for r in csv.reader(fh)]


def main():
    ds = gold_gen.build_dataset()
    seq = ds['seq']
    zero = ds['zeroshot']
    lib = ds['library']
    test_ids = [r['variant_id'] for r in ds['test']]
    test_muts = {r['variant_id']: parse_muts(r['mutations'], seq) for r in ds['test']}
    test_text = {r['variant_id']: r['mutations'] for r in ds['test']}
    gold = gold_gen.generate_gold(ds)

    xs = np.array([zero[(r['position'], r['mut_aa'])] for r in lib])
    ys = np.array([math.log(float(r['relative_activity'])) for r in lib])
    slope = float((xs @ ys) / (xs @ xs)) if float(xs @ xs) > 0 else 0.5
    baseline = {}
    for vid in test_ids:
        s = sum(slope * zero[m] for m in test_muts[vid])
        baseline[vid] = math.exp(min(max(s, -8.0), 8.0))
    base_rho = spearman([baseline[v] for v in test_ids], [gold[v] for v in test_ids])
    base_rmse = rmse_log([baseline[v] for v in test_ids], [gold[v] for v in test_ids])

    rows = read_rows(os.path.join(OUT_DIR, 'predictions.csv'))
    header_ok = bool(rows) and len(rows[0]) >= 2 and [c.strip().lower() for c in rows[0][:2]] == ['variant_id', 'predicted_activity']
    seen, preds = [], {}
    for row in (rows[1:] if rows else []):
        if len(row) < 2:
            continue
        vid = row[0].strip()
        seen.append(vid)
        try:
            preds[vid] = float(row[1])
        except ValueError:
            pass

    id_hits = sum(1 for a, b in zip(seen, test_ids) if a == b)
    c1 = 0.34 * float(header_ok) + 0.33 * id_hits / len(test_ids) + 0.33 * len(preds) / len(test_ids)
    add('predictions_format', c1, 0.08, 'header=%s id_order_hits=%d/%d parsed=%d' % (header_ok, id_hits, len(test_ids), len(preds)))

    in_range = sum(1 for v in test_ids if v in preds and math.isfinite(preds[v]) and 0.005 <= preds[v] <= 10.0)
    add('prediction_domain', in_range / len(test_ids), 0.05, '%d/%d inside [0.005, 10]' % (in_range, len(test_ids)))

    ranked = sorted([v for v in test_ids if v in preds and math.isfinite(preds[v])],
                    key=lambda v: (-preds[v], v))
    expected_top = ranked[:10]
    trows = read_rows(os.path.join(OUT_DIR, 'top10.csv'))
    top_header = bool(trows) and len(trows[0]) >= 2 and [c.strip().lower() for c in trows[0][:2]] == ['rank', 'variant_id']
    got, ranks = [], []
    for row in (trows[1:] if trows else []):
        if len(row) < 2:
            continue
        ranks.append(row[0].strip())
        got.append(row[1].strip())
    rank_ok = ranks == [str(i + 1) for i in range(len(ranks))] and len(ranks) == 10
    match = sum(1 for a, b in zip(got, expected_top) if a == b)
    c3 = 0.2 * float(trows is not None) + 0.2 * float(top_header) + 0.2 * float(rank_ok) + 0.4 * match / 10.0
    add('top10_ranking', c3, 0.08, 'header=%s ten_ranks=%s overlap=%d/10' % (top_header, rank_ok, match))

    eligible = [v for v in test_ids if len(test_muts[v]) <= 4 and not any(p in CONSERVED for p, _ in test_muts[v])]
    ranked_elig = sorted([v for v in eligible if v in preds], key=lambda v: (-preds[v], v))
    expected_panel = ranked_elig[:5]
    prows = read_rows(os.path.join(OUT_DIR, 'panel.csv'))
    pbody = prows[1:] if prows else []
    pids = [r[0].strip() for r in pbody if r]
    subs = []
    subs.append(1.0 if prows is not None else 0.0)
    subs.append(1.0 if (prows and len(prows[0]) >= 3 and [c.strip().lower() for c in prows[0][:3]] == ['variant_id', 'mutations', 'predicted_activity']) else 0.0)
    subs.append(1.0 if len(pids) == 5 and len(set(pids)) == 5 else 0.0)
    subs.append(sum(1 for v in pids if v in test_ids) / 5.0)
    subs.append(sum(1 for v in pids if v in eligible) / 5.0)
    mut_ok = 0
    act_ok = 0
    for row in pbody:
        if len(row) < 3:
            continue
        vid = row[0].strip()
        parsed = parse_muts(row[1], seq)
        if parsed is not None and vid in test_muts and set(parsed) == set(test_muts[vid]):
            mut_ok += 1
        try:
            val = float(row[2])
        except ValueError:
            continue
        if vid in preds and abs(val - preds[vid]) <= 1e-4 and val >= 1.2:
            act_ok += 1
    subs.append(mut_ok / 5.0)
    subs.append(act_ok / 5.0)
    subs.append(1.0 if len(pids) == 5 and set(pids) == set(expected_panel) else 0.0)
    add('panel_selection', sum(subs) / len(subs), 0.17,
        'subs=%s eligible=%d expected=%s' % ([round(s, 2) for s in subs], len(eligible), expected_panel))

    cv = None
    cv_path = os.path.join(OUT_DIR, 'cv_results.json')
    if os.path.isfile(cv_path):
        try:
            with open(cv_path) as fh:
                cv = json.load(fh)
        except Exception:
            cv = None
    csubs = []
    csubs.append(1.0 if isinstance(cv, dict) else 0.0)
    configs = cv.get('configs') if isinstance(cv, dict) else None
    configs = configs if isinstance(configs, list) else []
    csubs.append(1.0 if len(configs) >= 3 else (0.5 if len(configs) == 2 else 0.0))
    good, means = 0, []
    for c in configs:
        if not isinstance(c, dict):
            continue
        folds = c.get('folds')
        if not isinstance(folds, list) or len(folds) != 5:
            continue
        try:
            fv = [float(x) for x in folds]
            mean = float(c.get('mean'))
        except (TypeError, ValueError):
            continue
        if any(x < -1.0 or x > 1.0 for x in fv):
            continue
        if abs(sum(fv) / 5.0 - mean) > 1e-4:
            continue
        good += 1
        means.append(mean)
    csubs.append(good / max(len(configs), 1))
    csubs.append(1.0 if sum(1 for m in means if m >= 0.30) >= 3 else 0.0)
    consistent = 0.0
    if isinstance(cv, dict):
        bc, bs = cv.get('best_config'), cv.get('best_cv_spearman')
        try:
            bs = float(bs)
        except (TypeError, ValueError):
            bs = None
        for c in configs:
            if isinstance(c, dict) and c.get('name') == bc and bs is not None:
                try:
                    if abs(float(c.get('mean')) - bs) <= 0.01 and 0.25 <= bs <= 0.99:
                        consistent = 1.0
                except (TypeError, ValueError):
                    pass
    csubs.append(consistent)
    add('cv_report', sum(csubs) / len(csubs), 0.10, 'subs=%s configs=%d' % ([round(s, 2) for s in csubs], len(configs)))

    common = [v for v in test_ids if v in preds and math.isfinite(preds[v])]
    if len(common) < 5:
        rho, delta, ratio = 0.0, -9.0, 9.0
        rmse_agent = 9.0
    else:
        rho = spearman([preds[v] for v in common], [gold[v] for v in common])
        delta = rho - base_rho
        rmse_agent = rmse_log([preds[v] for v in common], [gold[v] for v in common])
        ratio = rmse_agent / base_rmse if base_rmse > 0 else 9.0
    c6 = 1.0 if delta >= 0.06 else 0.75 if delta >= 0.04 else 0.5 if delta >= 0.025 else 0.25 if delta >= 0.01 else 0.0
    add('spearman_vs_baseline', c6, 0.22, 'rho=%.3f baseline=%.3f delta=%.3f' % (rho, base_rho, delta))
    c7 = 1.0 if ratio <= 0.85 else 0.75 if ratio <= 0.90 else 0.5 if ratio <= 0.95 else 0.25 if ratio <= 1.0 else 0.0
    add('log_rmse_ratio', c7, 0.12, 'rmse=%.3f baseline=%.3f ratio=%.3f' % (rmse_agent, base_rmse, ratio))

    gold_top = set(sorted(test_ids, key=lambda v: -gold[v])[:20])
    hits = sum(1 for v in expected_top if v in gold_top)
    c8 = 1.0 if hits >= 7 else 0.8 if hits == 6 else 0.6 if hits == 5 else 0.4 if hits == 4 else 0.2 if hits == 3 else 0.0
    add('top10_recovery', c8, 0.10, '%d/10 predicted top designs are in the true top-20' % hits)

    decoys = [v for v in test_ids if any(p in CONSERVED for p, _ in test_muts[v])]
    if decoys:
        worst = max([preds.get(v, 999.0) for v in decoys])
        c9 = 1.0 if worst <= 0.4 else 0.6 if worst <= 0.8 else 0.3 if worst <= 1.5 else 0.0
    else:
        worst = float('nan')
        c9 = 0.0
    add('essential_position_decoy', c9, 0.08, 'n=%d max_prediction=%.3f' % (len(decoys), worst if worst == worst else -1.0))

    total_w = sum(c['weight'] for c in CHECKS) or 1.0
    score = sum(c['score'] * c['weight'] for c in CHECKS) / total_w
    report = {'score': round(score, 4), 'reward': round(score, 4),
              'checks': CHECKS, 'baseline_spearman': round(base_rho, 4),
              'baseline_log_rmse': round(base_rmse, 4)}
    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(REWARD_PATH, 'w') as fh:
        json.dump(report, fh, indent=2)
    for c in CHECKS:
        print('%-26s %.4f  (w=%.2f)  %s' % (c['name'], c['score'], c['weight'], c['detail']))
    print('TOTAL %.4f' % score)
    return 0


if __name__ == '__main__':
    sys.exit(main())
