#!/usr/bin/env python3
'''Deep verifier for the MPN plasma-proteome diagnostic task.

Reward = fraction of independent constraints passed. Only JSON numbers are
written to /logs/verifier/reward.json; the breakdown goes to stdout and to
/logs/verifier/report.json.
'''
import csv
import json
import math
import os
import sys
import traceback

sys.path.insert(0, '/tests')
import gold_gen

OUT_DIR = '/app/output'
LOG_DIR = '/logs/verifier'
CHECKS = []
CONSTRAINTS = ['C1_schema_and_coverage', 'C2_prediction_validity', 'C3_test_auc',
               'C4_balanced_accuracy', 'C5_de_table_validity', 'C6_de_sensitivity',
               'C7_de_direction', 'C8_il6_negative_control', 'C9_pv_tfrc',
               'C10_panel', 'C11_report_consistency']
GOLD_DE = dict(gold_gen.GOLD_DE_SET)
GOLD_INFORMATIVE = set(gold_gen.GOLD_DE_UP) | set(gold_gen.GOLD_DE_DOWN)
ALL_PROTEINS = set(gold_gen.PROTEINS)


def record(name, ok, detail=''):
    CHECKS.append({'constraint': name, 'passed': bool(ok), 'detail': str(detail)[:400]})


def read_csv_rows(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def to_float(value):
    if value is None:
        return None
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def roc_auc(y_true, scores):
    pairs = sorted(zip(scores, y_true), key=lambda t: t[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    pos = sum(1 for _, yy in pairs if yy == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None
    rank_sum = sum(r for r, (_, yy) in zip(ranks, pairs) if yy == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / float(pos * neg)


def imputed_column(values, ids, protein):
    col = [values[sid].get(protein) for sid in ids]
    obs = sorted(v for v in col if v is not None)
    med = obs[len(obs) // 2] if obs else 0.0
    return [med if v is None else v for v in col]


def run():
    gold = gold_gen.generate_dataset()
    samples = gold['samples']
    values = gold['values']
    labels = {s['sample_id']: s['group'] for s in samples}
    subtypes = {s['sample_id']: s['subtype'] for s in samples}
    cohorts = {s['sample_id']: s['cohort'] for s in samples}
    order = [s['sample_id'] for s in samples]
    test_ids = [sid for sid in order if cohorts[sid] == 'test']
    train_ids = [sid for sid in order if cohorts[sid] == 'train']
    y_test = [labels[sid] for sid in test_ids]

    pred_rows, pred_err = [], None
    try:
        pred_rows = read_csv_rows(os.path.join(OUT_DIR, 'predictions.csv'))
    except Exception as exc:
        pred_err = '%s: %s' % (type(exc).__name__, exc)
    de_rows, de_err = [], None
    try:
        de_rows = read_csv_rows(os.path.join(OUT_DIR, 'differential_expression.csv'))
    except Exception as exc:
        de_err = '%s: %s' % (type(exc).__name__, exc)
    pv_rows, pv_err = [], None
    try:
        pv_rows = read_csv_rows(os.path.join(OUT_DIR, 'pv_markers.csv'))
    except Exception as exc:
        pv_err = '%s: %s' % (type(exc).__name__, exc)
    panel, panel_err = None, None
    try:
        with open(os.path.join(OUT_DIR, 'panel.json')) as fh:
            panel = json.load(fh)
    except Exception as exc:
        panel_err = '%s: %s' % (type(exc).__name__, exc)
    report, report_err = None, None
    try:
        with open(os.path.join(OUT_DIR, 'report.json')) as fh:
            report = json.load(fh)
    except Exception as exc:
        report_err = '%s: %s' % (type(exc).__name__, exc)

    pred_ids = [str(r.get('sample_id', '')).strip() for r in pred_rows]
    de_cols = set(de_rows[0].keys()) if de_rows else set()
    pv_cols = set(pv_rows[0].keys()) if pv_rows else set()
    schema_ok = (pred_err is None and de_err is None and pv_err is None
                 and panel_err is None and report_err is None
                 and bool(pred_rows) and bool(de_rows) and bool(pv_rows)
                 and {'sample_id', 'mpn_probability'} <= set(pred_rows[0].keys())
                 and {'protein', 'log2_fc', 'p_value', 'q_value'} <= de_cols
                 and {'protein', 'log2_fc', 'q_value'} <= pv_cols
                 and isinstance(panel, dict) and isinstance(report, dict)
                 and pred_ids == test_ids)
    record('C1_schema_and_coverage', schema_ok,
           'pred_err=%s de_err=%s pv_err=%s panel_err=%s report_err=%s pred_ids_match=%s' %
           (pred_err, de_err, pv_err, panel_err, report_err, pred_ids == test_ids))

    probs = [to_float(r.get('mpn_probability')) for r in pred_rows]
    valid_probs = bool(probs) and all(p is not None and 0.0 <= p <= 1.0 for p in probs)
    above = sum(1 for p in probs if p is not None and p > 0.5)
    below = sum(1 for p in probs if p is not None and p <= 0.5)
    distinct = len(set(round(p, 4) for p in probs if p is not None))
    c2 = (valid_probs and len(pred_rows) == len(test_ids)
          and above >= 5 and below >= 5 and distinct >= 8)
    record('C2_prediction_validity', c2,
           'n=%d above=%d below=%d distinct=%d valid=%s' %
           (len(pred_rows), above, below, distinct, valid_probs))

    score_by_id = {}
    for r in pred_rows:
        score_by_id[str(r.get('sample_id', '')).strip()] = to_float(r.get('mpn_probability'))
    s_test = [score_by_id.get(sid) for sid in test_ids]
    auc = None
    if len(s_test) == len(test_ids) and all(v is not None for v in s_test):
        auc = roc_auc(y_test, s_test)
    record('C3_test_auc', auc is not None and auc >= 0.75, 'auc=%s' % auc)

    bal_acc = None
    if auc is not None:
        calls = [1 if v >= 0.5 else 0 for v in s_test]
        tp = sum(1 for c, y in zip(calls, y_test) if c == 1 and y == 1)
        tn = sum(1 for c, y in zip(calls, y_test) if c == 0 and y == 0)
        npos = sum(1 for y in y_test if y == 1)
        nneg = len(y_test) - npos
        if npos and nneg:
            bal_acc = 0.5 * (tp / float(npos) + tn / float(nneg))
    record('C4_balanced_accuracy', bal_acc is not None and bal_acc >= 0.70,
           'balanced_accuracy=%s' % bal_acc)

    de_map = {}
    de_valid = True
    for r in de_rows:
        p = str(r.get('protein', '')).strip()
        lfc = to_float(r.get('log2_fc'))
        pv = to_float(r.get('p_value'))
        q = to_float(r.get('q_value'))
        if (not p or lfc is None or pv is None or q is None
                or not (0.0 <= pv <= 1.0) or not (0.0 <= q <= 1.0)
                or q < pv - 1e-9):
            de_valid = False
            continue
        de_map[p] = (lfc, pv, q)
    n_sig = sum(1 for v in de_map.values() if v[2] < 0.05)
    c5 = len(de_map) >= 40 and de_valid and n_sig >= 12
    record('C5_de_table_validity', c5,
           'n_rows=%d valid=%s n_sig=%d' % (len(de_map), de_valid, n_sig))

    true_pos = [p for p in GOLD_DE if p in de_map and de_map[p][2] < 0.05]
    recall = len(true_pos) / float(len(GOLD_DE))
    fps = [p for p in de_map if p not in GOLD_DE and de_map[p][2] < 0.05]
    c6 = recall >= 0.55 and len(fps) <= 8
    record('C6_de_sensitivity', c6,
           'recall=%.3f n_true_pos=%d n_false_pos=%d' % (recall, len(true_pos), len(fps)))

    agree = 0
    for p in true_pos:
        sign = 1 if de_map[p][0] > 0 else -1
        if sign == GOLD_DE[p]:
            agree += 1
    c7 = len(true_pos) >= 5 and (agree / float(len(true_pos))) >= 0.80
    record('C7_de_direction', c7,
           'n_evaluated=%d n_agree=%d' % (len(true_pos), agree))

    il6 = de_map.get('IL6')
    c8 = il6 is not None and il6[2] >= 0.05 and abs(il6[0]) <= 0.30
    record('C8_il6_negative_control', c8, 'IL6=%s' % (il6,))

    pv_map = {}
    for r in pv_rows:
        p = str(r.get('protein', '')).strip()
        lfc = to_float(r.get('log2_fc'))
        q = to_float(r.get('q_value'))
        if p and lfc is not None and q is not None:
            pv_map[p] = (lfc, q)
    tfrc_marker_ok = ('TFRC' in pv_map and pv_map['TFRC'][1] < 0.05
                      and pv_map['TFRC'][0] > 0)
    mpn_train = [sid for sid in train_ids if labels[sid] == 1]
    pv_set = set(sid for sid in mpn_train if subtypes[sid] == 'PV')
    sub_y = [1 if sid in pv_set else 0 for sid in mpn_train]
    tfrc_vals = imputed_column(values, mpn_train, 'TFRC')
    gold_tfrc_auc = roc_auc(sub_y, tfrc_vals)
    reported_auc = to_float(report.get('tfrc_pv_auc')) if isinstance(report, dict) else None
    auc_ok = (reported_auc is not None and gold_tfrc_auc is not None
              and abs(reported_auc - gold_tfrc_auc) <= 0.10)
    c9 = tfrc_marker_ok and auc_ok
    record('C9_pv_tfrc', c9,
           'tfrc_sig=%s reported=%s gold=%s' % (tfrc_marker_ok, reported_auc, gold_tfrc_auc))

    sel = panel.get('selected_proteins') if isinstance(panel, dict) else None
    sel = sel if isinstance(sel, list) else []
    sel_clean = [str(x).strip() for x in sel]
    uniq = set(sel_clean)
    valid_names = all(p in ALL_PROTEINS for p in sel_clean)
    informative = len(uniq & GOLD_INFORMATIVE)
    c10 = (3 <= len(uniq) <= 12 and len(uniq) == len(sel_clean)
           and valid_names and informative >= 4)
    record('C10_panel', c10,
           'n=%d unique=%d valid_names=%s informative=%d' %
           (len(sel_clean), len(uniq), valid_names, informative))

    c11 = False
    detail = 'report missing'
    if isinstance(report, dict):
        ok = True
        try:
            ok = ok and int(report.get('n_de_significant')) == n_sig
        except (TypeError, ValueError):
            ok = False
        try:
            ok = ok and int(report.get('n_train_samples')) == len(train_ids)
            ok = ok and int(report.get('n_test_samples')) == len(test_ids)
        except (TypeError, ValueError):
            ok = False
        rep_il6_q = to_float(report.get('il6_q_mpn_vs_nonmpn'))
        if il6 is not None and rep_il6_q is not None:
            ok = ok and abs(rep_il6_q - il6[2]) <= 1e-3
        else:
            ok = False
        c11 = ok
        detail = 'n_de_reported=%s n_de_table=%d n_train=%s n_test=%s il6_q=%s' % (
            report.get('n_de_significant'), n_sig, report.get('n_train_samples'),
            report.get('n_test_samples'), rep_il6_q)
    record('C11_report_consistency', c11, detail)

    try:
        shipped = read_csv_rows('/app/data/sample_metadata.csv')
        if len(shipped) != len(samples):
            print('WARNING: shipped metadata row count differs from regenerated data')
    except Exception:
        pass


def main():
    try:
        run()
    except Exception:
        record('grader_internal_error', False, traceback.format_exc()[-400:])
    seen = set(c['constraint'] for c in CHECKS)
    for name in CONSTRAINTS:
        if name not in seen:
            record(name, False, 'not evaluated')
    ordered = []
    for name in CONSTRAINTS:
        for c in CHECKS:
            if c['constraint'] == name:
                ordered.append(c)
                break
    n_pass = sum(1 for c in ordered if c['passed'])
    score = round(n_pass / float(len(CONSTRAINTS)), 4)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, 'reward.json'), 'w') as fh:
        json.dump({'score': score, 'reward': score}, fh)
    with open(os.path.join(LOG_DIR, 'report.json'), 'w') as fh:
        json.dump({'score': score, 'n_passed': n_pass, 'n_constraints': len(CONSTRAINTS),
                   'constraints': ordered}, fh, indent=2)
    print(json.dumps({'score': score, 'constraints': ordered}, indent=2))


if __name__ == '__main__':
    main()
