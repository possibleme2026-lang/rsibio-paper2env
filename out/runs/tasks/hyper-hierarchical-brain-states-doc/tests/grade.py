#!/usr/bin/env python3
"""Deep grader for the hyper-hierarchical brain state task (partial credit)."""
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gold_gen as G

OUT = '/app/outputs'
CHECKS = []


def add(name, ok, detail=''):
    CHECKS.append({'name': name, 'ok': bool(ok), 'detail': str(detail)[:400]})


def fin(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def close(a, b, rel=0.10, absol=0.0):
    fa, fb = fin(a), fin(b)
    if fa is None or fb is None:
        return False
    return abs(fa - fb) <= max(absol, rel * abs(fb))


def dig(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def main():
    gold, grows = G.analyse(G.generate_dataset())
    gsub = {r['subject_id']: r for r in grows}

    res = None
    try:
        with open(os.path.join(OUT, 'results.json')) as fh:
            res = json.load(fh)
        add('results_json_parses', isinstance(res, dict), 'loaded top-level json')
    except Exception as exc:  # noqa: BLE001
        add('results_json_parses', False, repr(exc))

    rows = []
    try:
        with open(os.path.join(OUT, 'subject_states.csv')) as fh:
            rd = list(csv.DictReader(fh))
            rows = rd
        cols = set(rd[0].keys()) if rd else set()
        need = {'subject_id', 'group', 'crsr_score', 'mean_trophic_level', 'f0',
                'hyper_occupancy', 'hyper_dwell_time_s', 'hyper_n_entries',
                'metastability', 'synchrony'}
        ok = (len(rd) == 45) and need.issubset(cols)
        detail = 'rows=%d missing=%s' % (len(rd), sorted(need - cols))
        if ok:
            for r in rd:
                occ = fin(r['hyper_occupancy'])
                if occ is None or occ < -1e-9 or occ > 1.0 + 1e-9:
                    ok = False
                    detail = 'bad occupancy for %s' % r['subject_id']
                    break
                if r['group'] == 'CNT':
                    if str(r['crsr_score']).strip() not in ('', 'NA', 'nan', 'None'):
                        ok = False
                        detail = 'CNT row carries CRS-R'
                        break
                else:
                    cs = fin(r['crsr_score'])
                    if cs is None or cs < 0 or cs > 23:
                        ok = False
                        detail = 'bad CRS-R for %s' % r['subject_id']
                        break
        add('subject_csv_schema', ok, detail)
    except Exception as exc:  # noqa: BLE001
        add('subject_csv_schema', False, repr(exc))

    add('window_params', dig(res, 'n_windows') == 1845 and dig(res, 'n_subjects') == 45
        and dig(res, 'window_params', 'n_windows_per_subject') == 41
        and dig(res, 'window_params', 'window_length_tr') == 60
        and dig(res, 'window_params', 'step_tr') == 6 and dig(res, 'n_states') == 4,
        'n_windows=%s params=%s' % (dig(res, 'n_windows'), dig(res, 'window_params')))

    gs = gold['static_trophic']
    ok = all(close(dig(res, 'static_trophic', g, 'mean_f0'), gs[g]['mean_f0'], 0.10, 0.02)
             for g in G.GROUPS)
    add('static_f0_by_group', ok,
        'agent=%s gold=%s' % ([dig(res, 'static_trophic', g, 'mean_f0') for g in G.GROUPS],
                              [round(gs[g]['mean_f0'], 4) for g in G.GROUPS]))

    ok = all(close(dig(res, 'static_trophic', g, 'mean_trophic_level'),
                   gs[g]['mean_trophic_level'], 0.10, 0.02) for g in G.GROUPS)
    add('static_trophic_level_by_group', ok,
        'agent=%s gold=%s' % ([dig(res, 'static_trophic', g, 'mean_trophic_level') for g in G.GROUPS],
                              [round(gs[g]['mean_trophic_level'], 4) for g in G.GROUPS]))

    gd = gold['dynamic_states']
    vals = [dig(res, 'dynamic_states', g, 'occupancy') for g in G.GROUPS]
    ok = all(close(v, gd[g]['occupancy'], 0.0, 0.06) for v, g in zip(vals, G.GROUPS))
    add('hyper_occupancy_by_group', ok,
        'agent=%s gold=%s' % ([fin(v) for v in vals], [round(gd[g]['occupancy'], 4) for g in G.GROUPS]))

    av = [fin(v) for v in vals]
    gv = [gd[g]['occupancy'] for g in G.GROUPS]
    ok = all(v is not None for v in av)
    if ok:
        ok = (av[0] < av[1] < av[2]) == (gv[0] < gv[1] < gv[2])
        if gv[0] < gv[2]:
            ok = ok and (av[2] - av[0]) > 0.5 * (gv[2] - gv[0])
    add('hyper_occupancy_group_ordering', ok, 'agent=%s gold=%s' % (av, [round(x, 4) for x in gv]))

    vals = [dig(res, 'dynamic_states', g, 'dwell_time_s') for g in G.GROUPS]
    ok = all(close(v, gd[g]['dwell_time_s'], 0.15, 6.0) for v, g in zip(vals, G.GROUPS))
    add('hyper_dwell_time_by_group', ok,
        'agent=%s gold=%s' % ([fin(v) for v in vals], [round(gd[g]['dwell_time_s'], 3) for g in G.GROUPS]))

    av = [fin(v) for v in vals]
    gv = [gd[g]['dwell_time_s'] for g in G.GROUPS]
    ok = all(v is not None for v in av)
    if ok:
        ok = (av[0] < av[1] < av[2]) == (gv[0] < gv[1] < gv[2])
    add('hyper_dwell_group_ordering', ok, 'agent=%s gold=%s' % (av, [round(x, 3) for x in gv]))

    mvals, svals = [], []
    ok = True
    for g in G.GROUPS:
        m = dig(res, 'metastability', g)
        s = dig(res, 'synchrony', g)
        mvals.append(m)
        svals.append(s)
        if not close(m, gold['metastability'][g], 0.10, 0.01):
            ok = False
        if not close(s, gold['synchrony'][g], 0.10, 0.01):
            ok = False
    add('metastability_synchrony_by_group', ok,
        'agent_m=%s gold_m=%s agent_s=%s gold_s=%s' % (
            [fin(v) for v in mvals], [round(gold['metastability'][g], 4) for g in G.GROUPS],
            [fin(v) for v in svals], [round(gold['synchrony'][g], 4) for g in G.GROUPS]))

    am = [fin(v) for v in mvals]
    asy = [fin(v) for v in svals]
    gm = [gold['metastability'][g] for g in G.GROUPS]
    gsy = [gold['synchrony'][g] for g in G.GROUPS]
    ok = all(v is not None for v in am + asy)
    if ok:
        ok = (np.argsort(am).tolist() == np.argsort(gm).tolist()) and \
             (np.argsort(asy).tolist() == np.argsort(gsy).tolist())
    add('metastability_synchrony_ordering', ok, 'agent_m=%s gold_m=%s' % (am, [round(x, 4) for x in gm]))

    ga = gold['crsr_association']
    det = []
    ok = True
    for net in ('FRO', 'THA', 'INS'):
        a = dig(res, 'crsr_association', net, 'rho')
        grho = ga[net]['rho']
        det.append('%s agent=%s gold=%.3f' % (net, fin(a), grho))
        if not close(a, grho, 0.0, 0.12):
            ok = False
        fa = fin(a)
        if fa is not None and abs(grho) > 0.15 and (fa * grho) <= 0:
            ok = False
    add('crsr_regional_associations', ok, ' | '.join(det))

    a = dig(res, 'crsr_association', 'hyper_occupancy', 'rho')
    add('crsr_hyper_occupancy_association', close(a, ga['hyper_occupancy']['rho'], 0.0, 0.15),
        'agent=%s gold=%.3f' % (fin(a), ga['hyper_occupancy']['rho']))

    top = dig(res, 'hyper_state_top_rois')
    ok = isinstance(top, list) and len(top) == 20
    ov = 0
    if ok:
        gset = set(gold['hyper_state_top_rois'])
        ov = len(set(top) & gset)
        ok = ov >= 15
        detail = 'overlap=%d/20' % ov
    else:
        detail = 'not a 20-element list'
    add('top_roi_overlap', ok, detail)

    if rows:
        n_ok = 0
        for r in rows:
            g = gsub.get(r['subject_id'])
            if g is None:
                continue
            if close(r['mean_trophic_level'], g['mean_trophic_level'], 0.10, 0.02) and \
               close(r['f0'], g['f0'], 0.10, 0.02):
                n_ok += 1
        add('per_subject_static', n_ok >= 40, '%d/45 subjects within tolerance' % n_ok)

        n_ok = 0
        for r in rows:
            g = gsub.get(r['subject_id'])
            if g is None:
                continue
            if close(r['hyper_occupancy'], g['hyper_occupancy'], 0.0, 0.12) and \
               close(r['metastability'], g['metastability'], 0.15, 0.01) and \
               close(r['synchrony'], g['synchrony'], 0.15, 0.01):
                n_ok += 1
        add('per_subject_dynamic', n_ok >= 35, '%d/45 subjects within tolerance' % n_ok)
    else:
        add('per_subject_static', False, 'no csv rows')
        add('per_subject_dynamic', False, 'no csv rows')

    passed = sum(1 for c in CHECKS if c['ok'])
    score = passed / float(max(1, len(CHECKS)))
    for c in CHECKS:
        print(('PASS ' if c['ok'] else 'FAIL ') + c['name'] + ' :: ' + c['detail'])
    print('SCORE %.4f (%d/%d)' % (score, passed, len(CHECKS)))
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as fh:
        json.dump({'score': round(score, 6), 'reward': round(score, 6)}, fh)


if __name__ == '__main__':
    main()
