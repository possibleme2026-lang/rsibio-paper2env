#!/usr/bin/env python3
'''Verifier for the confidence-aware OXPHOS prioritisation task.

Seven independent constraint groups are checked against the submitted candidate
ranking and combined into a weighted score in [0, 1].  The hidden OXPHOS ground
truth is regenerated deterministically from the protected generator seed
(gold_gen.py) and is never present inside the task container.
'''
import csv
import json
import math
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gold_gen

REWARD_PATH = '/logs/verifier/reward.json'
REPORT_PATH = '/logs/verifier/report.json'
RESULT_DIRS = ['/app/results', '/results', '/app/output', '/output', './results']
REQUIRED = ['gene_id', 'score_supervised', 'score_coexpression',
            'n_modules_supporting', 'final_score', 'rank']
GROUPS = [('format', 0.12), ('bagging', 0.06), ('shortlist', 0.07),
          ('report', 0.10), ('integration', 0.15), ('auroc_main', 0.35),
          ('auroc_low_confidence', 0.15)]


def write_json(path, payload):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    with open(path, 'w') as fh:
        json.dump(payload, fh)


def safe_float(x):
    try:
        v = float(str(x).strip())
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def rankdata_avg(values):
    values = np.asarray(values, dtype=float)
    n = values.size
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    ordered = values[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1] == ordered[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata_avg(scores)
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3:
        return None
    ra = rankdata_avg(a)
    rb = rankdata_avg(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    if denom <= 0:
        return None
    return float((ra * rb).sum() / denom)


def ramp(value, lo, hi):
    if value is None:
        return 0.0
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


def read_table(path):
    with open(path, newline='') as fh:
        raw = list(csv.reader(fh, delimiter='\t'))
    rows = [r for r in raw if any(str(c).strip() for c in r)]
    if not rows:
        raise ValueError('empty file')
    header = [str(c).strip() for c in rows[0]]
    return header, rows[1:]


def main():
    values = {name: 0.0 for name, _ in GROUPS}
    details = {name: 'not evaluated' for name, _ in GROUPS}
    metrics = {}

    gold = gold_gen.build()
    cand_ids = [str(g) for g in gold['candidate_ids']]
    cand_set = set(cand_ids)
    cand_labels = np.asarray(gold['candidate_labels'], dtype=int)
    low_conf = np.asarray(gold['candidate_low_conf']).astype(bool)
    n_hc = int(len(gold['idx_hc']))
    n_lc = int(low_conf.sum())

    res_dir = None
    for cand in RESULT_DIRS:
        if os.path.isfile(os.path.join(cand, 'candidate_scores.tsv')):
            res_dir = cand
            break

    gene_col = []
    sup = co = fin = bag = None

    # ---------------- constraint 1/2: format and completeness ----------------
    try:
        if res_dir is None:
            raise ValueError('candidate_scores.tsv not found in ' + ', '.join(RESULT_DIRS))
        header, rows = read_table(os.path.join(res_dir, 'candidate_scores.tsv'))
        missing = [c for c in REQUIRED if c not in header]
        if missing:
            raise ValueError('missing columns: ' + ','.join(missing))
        col = {c: header.index(c) for c in REQUIRED}
        gene_col = [r[col['gene_id']].strip() for r in rows]
        sup_l = [safe_float(r[col['score_supervised']]) for r in rows]
        co_l = [safe_float(r[col['score_coexpression']]) for r in rows]
        fin_l = [safe_float(r[col['final_score']]) for r in rows]
        bag_l = [safe_float(r[col['n_modules_supporting']]) for r in rows]
        rank_l = [safe_float(r[col['rank']]) for r in rows]
        numeric_ok = all(v is not None for v in sup_l + co_l + fin_l + bag_l + rank_l)
        ids_ok = (len(set(gene_col)) == len(gene_col)) and (set(gene_col) == cand_set)
        subs = [1.0 if len(rows) == len(cand_ids) else 0.0,
                1.0 if ids_ok else 0.0,
                1.0 if numeric_ok else 0.0]
        if numeric_ok:
            ranks_int = [int(round(v)) for v in rank_l]
            int_ranks = all(abs(v - round(v)) < 1e-6 for v in rank_l)
            subs.append(1.0 if (ranks_int == list(range(1, len(rows) + 1)) and int_ranks) else 0.0)
            subs.append(1.0 if all(fin_l[i] >= fin_l[i + 1] - 1e-12 for i in range(len(fin_l) - 1)) else 0.0)
        else:
            subs.extend([0.0, 0.0])
        values['format'] = float(np.mean(subs))
        details['format'] = 'rows=%d (expected %d); ids_match=%s; numeric=%s; rank_seq=%.1f; sorted=%.1f' % (
            len(rows), len(cand_ids), ids_ok, numeric_ok, subs[3], subs[4])
        if numeric_ok and ids_ok:
            m_fin = dict(zip(gene_col, fin_l))
            m_sup = dict(zip(gene_col, sup_l))
            m_co = dict(zip(gene_col, co_l))
            m_bag = dict(zip(gene_col, bag_l))
            fin = np.array([m_fin[g] for g in cand_ids], dtype=float)
            sup = np.array([m_sup[g] for g in cand_ids], dtype=float)
            co = np.array([m_co[g] for g in cand_ids], dtype=float)
            bag = np.array([m_bag[g] for g in cand_ids], dtype=float)
            metrics['n_rows'] = len(rows)
    except Exception as exc:
        details['format'] = 'error: %s' % exc

    # ---------------- constraint 3: consensus column ----------------
    try:
        if bag is None:
            raise ValueError('candidate table unusable')
        ints_ok = all((0.0 <= v <= 5.0) and abs(v - round(v)) < 1e-6 for v in bag)
        sp = spearman(bag, fin)
        vals = 0.6 * (1.0 if ints_ok else 0.0)
        if sp is not None and sp >= 0.20:
            vals += 0.4
        elif sp is not None and sp >= 0.05:
            vals += 0.2
        values['bagging'] = vals
        details['bagging'] = 'integer_range_ok=%s; spearman(consensus, final)=%s' % (
            ints_ok, 'n/a' if sp is None else '%.4f' % sp)
    except Exception as exc:
        details['bagging'] = 'error: %s' % exc

    # ---------------- constraint 4: shortlist ----------------
    try:
        if res_dir is None:
            raise ValueError('no result directory')
        header_s, rows_s = read_table(os.path.join(res_dir, 'shortlist.tsv'))
        if 'gene_id' not in header_s:
            raise ValueError('shortlist.tsv has no gene_id column')
        sid = [r[header_s.index('gene_id')].strip() for r in rows_s]
        expected = gene_col[:25] if gene_col else []
        vals = 0.4
        if len(sid) == 25:
            vals += 0.3
        if expected and sid == expected:
            vals += 0.3
        values['shortlist'] = vals
        details['shortlist'] = 'n=%d; matches_top25=%s' % (len(sid), bool(expected) and sid == expected)
    except Exception as exc:
        details['shortlist'] = 'error: %s' % exc

    # ---------------- constraint 5: report diagnostics ----------------
    try:
        if res_dir is None:
            raise ValueError('no result directory')
        with open(os.path.join(res_dir, 'report.json')) as fh:
            rep = json.load(fh)
        if not isinstance(rep, dict):
            raise ValueError('report.json is not a JSON object')

        def num(x):
            return isinstance(x, (int, float)) and not isinstance(x, bool)

        subs = [1.0]
        n1 = rep.get('n_round1_positives')
        subs.append(1.0 if num(n1) and int(n1) == n_hc else 0.0)
        n2 = rep.get('n_round2_added')
        subs.append(1.0 if num(n2) and 1 <= int(n2) <= n_lc else 0.0)
        nm = rep.get('n_module_models')
        subs.append(1.0 if num(nm) and int(nm) == 5 else 0.0)
        pma = rep.get('per_module_auroc')
        pma_ok = False
        if isinstance(pma, dict) and set(pma.keys()) == set(gold_gen.COMPLEXES):
            pma_ok = all(num(v) and 0.4 <= float(v) <= 1.0 for v in pma.values())
        subs.append(1.0 if pma_ok else 0.0)
        ocv = rep.get('overall_cv_auroc')
        subs.append(1.0 if num(ocv) and 0.5 <= float(ocv) <= 1.0 else 0.0)
        values['report'] = float(np.mean(subs))
        details['report'] = 'subchecks=' + ','.join('%.0f' % s for s in subs)
    except Exception as exc:
        details['report'] = 'error: %s' % exc

    # ---------------- constraint 6: two-stream integration ----------------
    try:
        if sup is None or co is None or fin is None:
            raise ValueError('candidate table unusable')
        a_f = auroc(fin, cand_labels)
        a_s = auroc(sup, cand_labels)
        a_c = auroc(co, cand_labels)
        sp_s = spearman(fin, sup)
        sp_c = spearman(fin, co)
        metrics['auroc_final'] = a_f
        metrics['auroc_supervised'] = a_s
        metrics['auroc_coexpression'] = a_c
        metrics['spearman_final_supervised'] = sp_s
        metrics['spearman_final_coexpression'] = sp_c
        vals = 0.0
        if sp_s is not None and sp_s >= 0.15:
            vals += 0.25
        if sp_c is not None and sp_c >= 0.15:
            vals += 0.25
        delta = None
        if a_f is not None and a_s is not None and a_c is not None:
            delta = a_f - max(a_s, a_c)
            metrics['auroc_gain'] = delta
            if delta >= 0.02 or a_f >= 0.85:
                vals += 0.5
            elif delta >= 0.0:
                vals += 0.25
        values['integration'] = vals
        details['integration'] = 'spearman(sup)=%s spearman(coexp)=%s auroc_gain=%s' % (
            'n/a' if sp_s is None else '%.3f' % sp_s,
            'n/a' if sp_c is None else '%.3f' % sp_c,
            'n/a' if delta is None else '%.3f' % delta)
    except Exception as exc:
        details['integration'] = 'error: %s' % exc

    # ---------------- constraint 7: ranking quality ----------------
    values['auroc_main'] = ramp(metrics.get('auroc_final'), 0.62, 0.86)
    details['auroc_main'] = 'auroc(final)=%s (ramp 0.62 -> 0.86)' % (
        'n/a' if metrics.get('auroc_final') is None else '%.4f' % metrics['auroc_final'])
    try:
        if fin is None:
            raise ValueError('candidate table unusable')
        a_lc = auroc(fin[low_conf], cand_labels[low_conf])
        metrics['auroc_low_confidence'] = a_lc
        values['auroc_low_confidence'] = ramp(a_lc, 0.55, 0.75)
        details['auroc_low_confidence'] = 'auroc over %d low-confidence genes = %s (ramp 0.55 -> 0.75)' % (
            int(low_conf.sum()), 'n/a' if a_lc is None else '%.4f' % a_lc)
    except Exception as exc:
        details['auroc_low_confidence'] = 'error: %s' % exc

    total = 0.0
    for name, w in GROUPS:
        total += w * float(values.get(name, 0.0))
    total = float(min(1.0, max(0.0, total)))

    write_json(REWARD_PATH, {'score': round(total, 6), 'reward': round(total, 6)})
    write_json(REPORT_PATH, {'score': round(total, 6),
                             'constraints': values,
                             'weights': {n: w for n, w in GROUPS},
                             'details': details,
                             'metrics': metrics,
                             'results_dir': res_dir})
    print('=== grader breakdown ===')
    for name, w in GROUPS:
        print('%-22s weight=%.2f value=%.3f contribution=%.4f | %s' % (
            name, w, values.get(name, 0.0), w * values.get(name, 0.0), details.get(name, '')))
    print('total score = %.4f' % total)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            write_json(REWARD_PATH, {'score': 0.0, 'reward': 0.0})
            write_json(REPORT_PATH, {'score': 0.0, 'error': 'grader crashed'})
        except Exception:
            pass
