#!/usr/bin/env python3
"""Verifier for the neutrophil NBT / TBARS-MDA cohort task (8 constraints, partial credit)."""
import json
import os
import traceback

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

DATA = '/app/data'
OUT = '/app/output/results.json'
REWARD_PATH = '/logs/verifier/reward.json'
REPORT_PATH = '/logs/verifier/report.json'

MIN_FOCUS = 0.60
MIN_CELL_AREA_PX = 200.0
AREA_THRESHOLD_UM2 = 25.0
SAT_A532 = 2.0


def build_gold():
    with open(os.path.join(DATA, 'assay_parameters.json')) as fh:
        params = json.load(fh)
    px = float(params['pixel_size_um'])
    dil = float(params['tbars_dilution_factor'])

    meta = pd.read_csv(os.path.join(DATA, 'subject_metadata.csv'), dtype={'subject_id': str})
    cells = pd.read_csv(os.path.join(DATA, 'nbt_cell_measurements.csv'),
                        dtype={'subject_id': str, 'field_id': str})
    fields = pd.read_csv(os.path.join(DATA, 'field_qc.csv'),
                         dtype={'subject_id': str, 'field_id': str})
    readings = pd.read_csv(os.path.join(DATA, 'tbars_plate_readings.csv'),
                           dtype={'plate_id': str, 'well': str})
    wmap = pd.read_csv(os.path.join(DATA, 'tbars_well_map.csv'),
                       dtype={'plate_id': str, 'well': str, 'sample_id': str})

    # ---------- single-cell NBT QC ----------
    merged = cells.merge(fields[['subject_id', 'field_id', 'focus_score']],
                         on=['subject_id', 'field_id'], how='left')
    if merged['focus_score'].isna().any():
        raise RuntimeError('cell rows without a matching field-QC row')
    n_cells_measured = int(len(merged))
    n_fields_measured = int(len(fields))

    focus_ok = merged['focus_score'] >= MIN_FOCUS
    n_exc_focus = int((~focus_ok).sum())
    s1 = merged[focus_ok]
    class_ok = s1['object_class'] == 'neutrophil'
    n_exc_class = int((~class_ok).sum())
    s2 = s1[class_ok]
    area_ok = s2['cell_area_px'] >= MIN_CELL_AREA_PX
    n_exc_area = int((~area_ok).sum())
    kept = s2[area_ok].copy()
    kept['area_um2'] = kept['formazan_area_px'] * (px ** 2)
    kept['positive'] = kept['area_um2'] >= AREA_THRESHOLD_UM2

    grp = kept.groupby('subject_id')
    per = pd.DataFrame({'n_cells_retained': grp.size(),
                        'nbt_positive_pct': grp['positive'].mean() * 100.0,
                        'nbt_reactivity': grp['area_um2'].mean()})
    n_cells_retained = int(len(kept))
    n_fields_retained = int(kept.groupby(['subject_id', 'field_id']).ngroups)

    # ---------- TBARS / MDA ----------
    r = readings.merge(wmap, on=['plate_id', 'well'], how='left')
    r['a_corr'] = r['absorbance_532'] - r['absorbance_600']
    curves = {}
    for plate, sub in r[r['well_type'] == 'standard'].groupby('plate_id'):
        x = sub['mda_nmol_per_ml'].astype(float).to_numpy()
        y = sub['a_corr'].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        curves[str(plate)] = {'slope': float(slope), 'intercept': float(intercept),
                              'r_squared': 1.0 - ss_res / ss_tot}

    samp = r[r['well_type'] == 'sample'].copy()
    n_sat = int((samp['absorbance_532'] > SAT_A532).sum())
    samp = samp[samp['absorbance_532'] <= SAT_A532]
    vals = []
    for row in samp.itertuples():
        cv = curves[str(row.plate_id)]
        vals.append((row.a_corr - cv['intercept']) / cv['slope'] * dil)
    samp = samp.assign(mda=vals)
    mda_subj = samp.groupby('sample_id')['mda'].mean()

    meta_idx = meta.set_index('subject_id')
    per = per.reindex(meta_idx.index)
    per['mda_nmol_per_ml'] = mda_subj.reindex(meta_idx.index)
    per['group'] = meta_idx['group']
    per['specimen_type'] = meta_idx['specimen_type']

    # ---------- statistics ----------
    group_stats = {}
    for col in ('nbt_positive_pct', 'nbt_reactivity', 'mda_nmol_per_ml'):
        a = per.loc[per['group'] == 'fertile_control', col].astype(float)
        b = per.loc[per['group'] == 'infertile', col].astype(float)
        _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        group_stats[col] = {'control_median': float(a.median()),
                            'infertile_median': float(b.median()),
                            'p_value': float(p)}

    corr = {}
    for c1, c2 in (('nbt_positive_pct', 'nbt_reactivity'),
                   ('nbt_reactivity', 'mda_nmol_per_ml'),
                   ('nbt_positive_pct', 'mda_nmol_per_ml')):
        rho, p = stats.spearmanr(per[c1].astype(float), per[c2].astype(float))
        corr[c1 + '_vs_' + c2] = {'rho': float(rho), 'p_value': float(p)}

    y = (per['group'] == 'infertile').astype(int).to_numpy()
    diag = {}
    for col in ('nbt_positive_pct', 'nbt_reactivity'):
        v = per[col].astype(float).to_numpy()
        auc = float(roc_auc_score(y, v))
        best_j = -1.0
        best = (float(np.unique(v)[0]), 0.0, 0.0)
        for c in np.unique(v):
            pred = v >= c
            tp = int((pred & (y == 1)).sum())
            fn = int(((~pred) & (y == 1)).sum())
            tn = int(((~pred) & (y == 0)).sum())
            fp = int((pred & (y == 0)).sum())
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            j = sens + spec - 1.0
            if j > best_j + 1e-12:
                best_j = j
                best = (float(c), sens, spec)
        diag[col] = {'auc': auc, 'youden_cutoff': best[0],
                     'sensitivity': best[1], 'specificity': best[2]}

    return {'qc': {'n_cells_measured': n_cells_measured,
                   'n_fields_measured': n_fields_measured,
                   'n_cells_retained': n_cells_retained,
                   'n_fields_retained': n_fields_retained,
                   'n_excluded_focus': n_exc_focus,
                   'n_excluded_object_class': n_exc_class,
                   'n_excluded_cell_area': n_exc_area,
                   'n_wells_saturated_excluded': n_sat},
            'standard_curves': curves,
            'per_subject': per,
            'group_comparison': group_stats,
            'correlations': corr,
            'diagnostic_performance': diag}


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def evaluate(agent, gold):
    res = []

    def add(name, ok, detail):
        res.append({'constraint': name, 'passed': bool(ok), 'detail': str(detail)})

    per = gold['per_subject']
    expected_ids = list(per.index)
    gqc = gold['qc']

    # ---- C1: QC funnel counts ----
    qc_keys = ['n_cells_measured', 'n_fields_measured', 'n_cells_retained',
               'n_fields_retained', 'n_excluded_focus', 'n_excluded_object_class',
               'n_excluded_cell_area', 'n_wells_saturated_excluded']
    aqc = agent.get('qc')
    bad = []
    for k in qc_keys:
        av = aqc.get(k) if isinstance(aqc, dict) else None
        try:
            av = int(av)
        except (TypeError, ValueError):
            av = None
        if av != gqc[k]:
            bad.append('{}: got {} expected {}'.format(k, av, gqc[k]))
    add('C1_qc_counts', not bad, 'all 8 QC counts match' if not bad else '; '.join(bad))

    # ---- C2: per-subject table ----
    aps = agent.get('per_subject')
    problems = []
    rows_by_id = {}
    if not isinstance(aps, list):
        problems.append('per_subject is not a list')
        aps = []
    if len(aps) != len(expected_ids):
        problems.append('per_subject has {} rows, expected {}'.format(len(aps), len(expected_ids)))
    got_ids = [r.get('subject_id') for r in aps if isinstance(r, dict)]
    if got_ids != expected_ids:
        problems.append('subject_id coverage/order mismatch (ascending order required)')
    for r in aps:
        if not isinstance(r, dict):
            problems.append('non-object entry in per_subject')
            break
        sid = r.get('subject_id')
        rows_by_id[sid] = r
        if sid not in per.index:
            problems.append('unknown subject_id {}'.format(sid))
            continue
        exp = per.loc[sid]
        if r.get('group') != exp['group']:
            problems.append('{}: group mismatch'.format(sid))
        if r.get('specimen_type') != exp['specimen_type']:
            problems.append('{}: specimen_type mismatch'.format(sid))
        if fnum(r.get('n_cells_retained')) != int(exp['n_cells_retained']):
            problems.append('{}: n_cells_retained {} != {}'.format(
                sid, r.get('n_cells_retained'), int(exp['n_cells_retained'])))
    add('C2_per_subject_table', not problems,
        'rows/ids/order/group/specimen/retained-cell counts correct'
        if not problems else '; '.join(problems[:6]))

    # ---- C3: NBT metric values ----
    bad = []
    for sid in expected_ids:
        r = rows_by_id.get(sid)
        if not isinstance(r, dict):
            bad.append('{}: missing row'.format(sid))
            continue
        exp = per.loc[sid]
        v = fnum(r.get('nbt_positive_pct'))
        if v is None or abs(v - exp['nbt_positive_pct']) > 0.01:
            bad.append('{}: nbt_positive_pct {} vs {}'.format(
                sid, r.get('nbt_positive_pct'), round(float(exp['nbt_positive_pct']), 4)))
            continue
        v2 = fnum(r.get('nbt_reactivity'))
        if v2 is None or abs(v2 - exp['nbt_reactivity']) > 0.05:
            bad.append('{}: nbt_reactivity {} vs {}'.format(
                sid, r.get('nbt_reactivity'), round(float(exp['nbt_reactivity']), 4)))
    add('C3_nbt_metrics', not bad,
        'nbt_positive_pct within 0.01 pp and nbt_reactivity within 0.05 um2 for all subjects'
        if not bad else '{} mismatching subjects: '.format(len(bad)) + '; '.join(bad[:6]))

    # ---- C4: MDA values and plate standard curves ----
    bad = []
    for sid in expected_ids:
        r = rows_by_id.get(sid)
        v = fnum(r.get('mda_nmol_per_ml')) if isinstance(r, dict) else None
        exp = float(per.loc[sid, 'mda_nmol_per_ml'])
        if v is None or abs(v - exp) > 0.10:
            bad.append('{}: mda {} vs {}'.format(sid, v, round(exp, 4)))
    sc = agent.get('standard_curves')
    for plate, gcv in gold['standard_curves'].items():
        acv = sc.get(plate) if isinstance(sc, dict) else None
        if not isinstance(acv, dict):
            bad.append('missing standard curve {}'.format(plate))
            continue
        r2 = fnum(acv.get('r_squared'))
        if r2 is None or r2 < 0.99 or abs(r2 - gcv['r_squared']) > 0.01:
            bad.append('{}: r_squared {} (gold {})'.format(plate, r2, round(gcv['r_squared'], 5)))
        for key in ('slope', 'intercept'):
            v = fnum(acv.get(key))
            gv = gcv[key]
            if v is None or abs(v - gv) > 0.05 * abs(gv):
                bad.append('{}: {} {} vs gold {}'.format(plate, key, v, round(gv, 6)))
    add('C4_mda_and_curves', not bad,
        '100 subject MDA values and 3 plate standard curves within tolerance'
        if not bad else '; '.join(bad[:6]))

    # ---- C5: group comparison ----
    bad = []
    gc = agent.get('group_comparison')
    tolmap = {'nbt_positive_pct': 0.2, 'nbt_reactivity': 0.5, 'mda_nmol_per_ml': 0.05}
    for col, gs in gold['group_comparison'].items():
        ac = gc.get(col) if isinstance(gc, dict) else None
        if not isinstance(ac, dict):
            bad.append('missing group_comparison.{}'.format(col))
            continue
        p = fnum(ac.get('p_value'))
        if p is None or not (0.0 < p < 0.001):
            bad.append('{}: p_value {} is not < 0.001'.format(col, ac.get('p_value')))
        cm = fnum(ac.get('control_median'))
        im = fnum(ac.get('infertile_median'))
        tol = tolmap[col]
        if cm is None or abs(cm - gs['control_median']) > tol:
            bad.append('{}: control median {} vs {}'.format(col, cm, round(gs['control_median'], 3)))
        if im is None or abs(im - gs['infertile_median']) > tol:
            bad.append('{}: infertile median {} vs {}'.format(col, im, round(gs['infertile_median'], 3)))
        if cm is not None and im is not None and im <= cm:
            bad.append('{}: infertile median not above control median'.format(col))
    add('C5_group_comparison', not bad,
        '3 biomarkers: direction, medians and p < 0.001 verified'
        if not bad else '; '.join(bad[:6]))

    # ---- C6: Spearman correlations ----
    bad = []
    acorr = agent.get('correlations')
    for key, gv in gold['correlations'].items():
        av = acorr.get(key) if isinstance(acorr, dict) else None
        rho = fnum(av.get('rho')) if isinstance(av, dict) else None
        if rho is None or abs(rho - gv['rho']) > 0.03:
            bad.append('{}: rho {} vs gold {}'.format(key, rho, round(gv['rho'], 4)))
    add('C6_spearman_correlations', not bad,
        '3 Spearman rho within +/-0.03 of the reference'
        if not bad else '; '.join(bad))

    # ---- C7: ROC AUC ----
    bad = []
    adiag = agent.get('diagnostic_performance')
    for col, gv in gold['diagnostic_performance'].items():
        av = adiag.get(col) if isinstance(adiag, dict) else None
        auc = fnum(av.get('auc')) if isinstance(av, dict) else None
        if auc is None or abs(auc - gv['auc']) > 0.015 or auc <= 0.70:
            bad.append('{}: auc {} vs gold {} (must be > 0.70)'.format(
                col, auc, round(gv['auc'], 4)))
    add('C7_roc_auc', not bad,
        'both AUCs within 0.015 of reference and > 0.70'
        if not bad else '; '.join(bad))

    # ---- C8: Youden cutoff and self-consistency ----
    bad = []
    for col, gv in gold['diagnostic_performance'].items():
        av = adiag.get(col) if isinstance(adiag, dict) else None
        cut = fnum(av.get('youden_cutoff')) if isinstance(av, dict) else None
        sens = fnum(av.get('sensitivity')) if isinstance(av, dict) else None
        spec = fnum(av.get('specificity')) if isinstance(av, dict) else None
        if cut is None or sens is None or spec is None:
            bad.append('{}: youden_cutoff/sensitivity/specificity missing'.format(col))
            continue
        if abs(cut - gv['youden_cutoff']) > 1.0:
            bad.append('{}: cutoff {} vs reference {}'.format(col, cut, gv['youden_cutoff']))
        xs, ys = [], []
        for sid in expected_ids:
            r = rows_by_id.get(sid)
            v = fnum(r.get(col)) if isinstance(r, dict) else None
            if v is None:
                continue
            xs.append(v)
            ys.append(1 if per.loc[sid, 'group'] == 'infertile' else 0)
        if len(xs) != len(expected_ids):
            bad.append('{}: cannot re-derive sens/spec (per_subject incomplete)'.format(col))
            continue
        xs = np.array(xs)
        ys = np.array(ys)
        pred = xs >= cut
        tp = int((pred & (ys == 1)).sum())
        fn = int(((~pred) & (ys == 1)).sum())
        tn = int(((~pred) & (ys == 0)).sum())
        fp = int((pred & (ys == 0)).sum())
        s1 = tp / max(tp + fn, 1)
        s2 = tn / max(tn + fp, 1)
        if abs(s1 - sens) > 0.035 or abs(s2 - spec) > 0.035:
            bad.append('{}: reported sens/spec {}/{} inconsistent with own table ({}/{})'.format(
                col, round(sens, 3), round(spec, 3), round(s1, 3), round(s2, 3)))
    add('C8_youden_cutoff', not bad,
        'both cutoffs within 1.0 of reference and sensitive/specificity self-consistent'
        if not bad else '; '.join(bad[:4]))

    return res


def write_outputs(results, note=''):
    n = len(results)
    passed = sum(1 for r in results if r['passed'])
    score = passed / n if n else 0.0
    os.makedirs('/logs/verifier', exist_ok=True)
    with open(REWARD_PATH, 'w') as fh:
        json.dump({'score': round(score, 4), 'reward': round(score, 4)}, fh)
    with open(REPORT_PATH, 'w') as fh:
        json.dump({'score': score, 'n_constraints': n, 'note': note,
                   'constraints': results}, fh, indent=2)
    if note:
        print('NOTE: ' + note)
    for r in results:
        print(('PASS ' if r['passed'] else 'FAIL ') + r['constraint'] + ' :: ' + r['detail'])
    print('SCORE {:.4f} ({}/{})'.format(score, passed, n))


def main():
    try:
        gold = build_gold()
    except Exception:
        traceback.print_exc()
        write_outputs([{'constraint': 'setup', 'passed': False,
                        'detail': 'gold-standard recomputation failed'}])
        return
    note = ''
    agent = {}
    try:
        with open(OUT) as fh:
            agent = json.load(fh)
        if not isinstance(agent, dict):
            agent = {}
            note = 'agent output is not a JSON object'
    except Exception as exc:
        note = 'could not read {}: {}'.format(OUT, exc)
        agent = {}
    results = evaluate(agent, gold)
    write_outputs(results, note)


if __name__ == '__main__':
    main()
