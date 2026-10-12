#!/usr/bin/env python3
'''Deep grader for the cross-domain white-matter tract segmentation benchmark.

Recomputes the whole benchmark from /app/data (the same inputs the agent has)
with the canonical tools (nibabel nearest-neighbour resampling, scipy connected
components) and compares the agent artefacts against it. Partial credit per
constraint group.
'''
import csv
import json
import os

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import label as cc_label
from scipy.ndimage import generate_binary_structure

DATA = '/app/data'
OUT = '/app/output'
LOGDIR = '/logs/verifier'
TPL_PATH = DATA + '/atlas/template.nii.gz'
ATLAS_PATH = DATA + '/atlas/tract_labels.csv'
SUBJ_PATH = DATA + '/subjects.csv'

PRESENT_MM3 = 250.0
CC_MIN_MM3 = 150.0
MODELS = ['cnn_a', 'cnn_b', 'transformer', 'unet3d']
ENS = 'EWMTS'
ALLMAPS = MODELS + [ENS]
MODEL_ORDER = dict((m, i) for i, m in enumerate(ALLMAPS))
DOMAINS = ['adult_ref', 'adult_alt', 'pediatric']
HEADER = ['subject_id', 'dataset', 'model', 'tract_id', 'tract_name', 'dice',
          'ref_volume_mm3', 'pred_volume_mm3', 'present_in_ref', 'present_in_pred']
WEIGHTS = {'structure': 0.10, 'dice': 0.22, 'volume': 0.10, 'presence': 0.08,
           'detection_f1': 0.12, 'summary': 0.13, 'ensemble': 0.15,
           'recommendation': 0.10}
REPORT = {}


def load_atlas():
    rows = list(csv.DictReader(open(ATLAS_PATH)))
    core = [r for r in rows if str(r['benchmark_core']).strip().lower() in ('yes', 'true', '1')]
    core.sort(key=lambda r: int(r['label_value']))
    name_by_label = dict((int(r['label_value']), r['tract_name']) for r in rows)
    non_core = set(int(r['label_value']) for r in rows
                   if str(r['benchmark_core']).strip().lower() not in ('yes', 'true', '1'))
    return rows, core, name_by_label, non_core


def build_ensemble(stack, vox_vol):
    a, b, c, d = stack[0], stack[1], stack[2], stack[3]
    cand = np.zeros(stack.shape[1:], np.int16)
    for x, y, z in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
        eq = (x == y) & (y == z)
        cand[eq] = x[eq]
    st = generate_binary_structure(3, 1)
    ens = np.zeros(stack.shape[1:], np.int16)
    for lb in np.unique(cand):
        if int(lb) == 0:
            continue
        m = (cand == lb)
        lab, n = cc_label(m, structure=st)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        keep = (sizes * vox_vol) >= CC_MIN_MM3
        keep[0] = False
        sel = keep[lab]
        if sel.any():
            ens[sel] = lb
    return ens


def aggregate(rows, n_subjects):
    out = {'n_subjects': n_subjects,
           'n_scored_tract_pairs': len(set((k[0], k[2]) for k, v in rows.items()
                                           if v['present_in_ref'])),
           'models': {}}
    for m in ALLMAPS:
        md = {}
        f1s = {}
        for d in DOMAINS:
            dice_vals = [v['dice'] for k, v in rows.items()
                         if k[1] == m and v['dataset'] == d and v['present_in_ref']]
            md[d] = float(np.mean(dice_vals)) if dice_vals else 0.0
            tp = fp = fn = 0
            for k, v in rows.items():
                if k[1] != m or v['dataset'] != d:
                    continue
                if v['present_in_ref'] and v['present_in_pred']:
                    tp += 1
                elif v['present_in_pred'] and not v['present_in_ref']:
                    fp += 1
                elif v['present_in_ref'] and not v['present_in_pred']:
                    fn += 1
            den = 2 * tp + fp + fn
            f1s[d] = (2.0 * tp / den) if den > 0 else 0.0
        within = [v['dice'] for k, v in rows.items()
                  if k[1] == m and v['dataset'] == 'adult_ref' and v['present_in_ref']]
        cross = [v['dice'] for k, v in rows.items()
                 if k[1] == m and v['dataset'] in ('adult_alt', 'pediatric') and v['present_in_ref']]
        w = float(np.mean(within)) if within else 0.0
        c = float(np.mean(cross)) if cross else 0.0
        out['models'][m] = {'mean_dice_by_domain': md,
                            'detection_f1_by_domain': f1s,
                            'within_domain_mean_dice': w,
                            'cross_domain_mean_dice': c,
                            'degradation_delta': w - c}
    return out


def make_recommendation(summary):
    ms = summary['models']

    def pick(field):
        return sorted(ALLMAPS, key=lambda m: (-ms[m][field], m))[0]

    def pick_dom(d):
        return sorted(ALLMAPS, key=lambda m: (-ms[m]['mean_dice_by_domain'][d], m))[0]

    rec = {'best_within_domain_model': pick('within_domain_mean_dice'),
           'best_cross_domain_model': pick('cross_domain_mean_dice'),
           'worst_degradation_model': pick('degradation_delta'),
           'best_pediatric_model': pick_dom('pediatric')}
    rec['ewmts_recovers_accuracy'] = bool(
        all(ms[ENS]['cross_domain_mean_dice'] > ms[m]['cross_domain_mean_dice'] for m in MODELS))
    return rec


def compute_gold():
    tpl = nib.load(TPL_PATH)
    tpl_aff = np.asarray(tpl.affine, float)
    tpl_shape = tuple(int(v) for v in tpl.shape[:3])
    vox_vol = float(abs(np.linalg.det(tpl_aff[:3, :3])))
    _, core, name_by_label, _ = load_atlas()
    subjects = list(csv.DictReader(open(SUBJ_PATH)))
    core_labels = [int(r['label_value']) for r in core]
    gold_rows = {}
    gold_ens = {}
    for s in subjects:
        sid = s['subject_id']
        ref = np.rint(nib.load(DATA + '/refs/' + sid + '_ref.nii.gz').get_fdata()).astype(np.int16)
        preds = {}
        for m in MODELS:
            img = nib.load(DATA + '/native/' + sid + '/pred_' + m + '.nii.gz')
            preds[m] = np.rint(resample_from_to(img, tpl, order=0).get_fdata()).astype(np.int16)
        ens = build_ensemble(np.stack([preds[m] for m in MODELS], axis=0), vox_vol)
        gold_ens[sid] = ens
        maps = dict(preds)
        maps[ENS] = ens
        for m in ALLMAPS:
            pm = maps[m]
            for lb in core_labels:
                A = (ref == lb)
                B = (pm == lb)
                na = int(np.count_nonzero(A))
                nb = int(np.count_nonzero(B))
                ni = int(np.count_nonzero(A & B))
                dv = (2.0 * ni / (na + nb)) if (na + nb) > 0 else 0.0
                va = na * vox_vol
                vb = nb * vox_vol
                gold_rows[(sid, m, lb)] = dict(dice=dv, ref_volume_mm3=va,
                                               pred_volume_mm3=vb,
                                               present_in_ref=bool(va >= PRESENT_MM3),
                                               present_in_pred=bool(vb >= PRESENT_MM3),
                                               dataset=s['dataset'],
                                               tract_name=name_by_label[lb])
    summary = aggregate(gold_rows, len(subjects))
    return dict(rows=gold_rows, ens=gold_ens, summary=summary,
                rec=make_recommendation(summary), vox_vol=vox_vol,
                tpl_shape=tpl_shape, tpl_aff=tpl_aff, subjects=subjects)


def as_bool(v):
    return str(v).strip().lower() in ('true', '1', 'yes')


def read_agent_rows():
    path = OUT + '/per_tract_metrics.csv'
    if not os.path.exists(path):
        return None, False
    with open(path, newline='') as fh:
        raw = list(csv.reader(fh))
    if not raw:
        return [], False
    hdr = [h.strip() for h in raw[0]]
    rows = []
    for r in raw[1:]:
        if not r or all(not c.strip() for c in r):
            continue
        if len(r) != len(hdr):
            continue
        rows.append(dict(zip(hdr, r)))
    return rows, (hdr == HEADER)


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def main():
    gold = compute_gold()
    gold_rows = gold['rows']
    gold_sum = gold['summary']
    gold_rec = gold['rec']
    tpl_aff = gold['tpl_aff']
    tpl_shape = gold['tpl_shape']
    subjects = gold['subjects']
    n_g = len(gold_rows)
    _, _, _, non_core = load_atlas()

    agent_rows, header_ok = read_agent_rows()
    agent_sum = load_json(OUT + '/benchmark_summary.json')
    agent_rec = load_json(OUT + '/recommendation.json')

    sub = {}
    detail = {}

    # ---------------- structure ---------------- #
    keys = []
    key_err = 0
    ds_ok = True
    if agent_rows:
        for r in agent_rows:
            try:
                k = (str(r['subject_id']).strip(), str(r['model']).strip(), int(str(r['tract_id']).strip()))
            except Exception:
                key_err += 1
                continue
            keys.append(k)
            g = gold_rows.get(k)
            if g is not None and str(r.get('dataset', '')).strip() != g['dataset']:
                ds_ok = False
    kset = set(keys)
    miss = len([k for k in gold_rows if k not in kset])
    extra = len([k for k in kset if k not in gold_rows])
    part_keys = max(0.0, 1.0 - miss / max(n_g, 1) - 0.5 * extra / max(n_g, 1))
    seq = [(k[0], MODEL_ORDER.get(k[1], 99), k[2]) for k in keys]
    sorted_ok = (seq == sorted(seq))
    leak = any(k[2] in non_core for k in kset)
    sum_schema = bool(isinstance(agent_sum, dict) and isinstance(agent_sum.get('models'), dict)
                      and all(isinstance(agent_sum['models'].get(m), dict) for m in ALLMAPS))
    rec_schema = bool(isinstance(agent_rec, dict))
    s_struct = (0.20 * (1.0 if header_ok else 0.0) + 0.30 * part_keys +
                0.15 * (1.0 if (sorted_ok and ds_ok and not leak and key_err == 0) else 0.0) +
                0.20 * (1.0 if sum_schema else 0.0) + 0.15 * (1.0 if rec_schema else 0.0))
    sub['structure'] = round(min(1.0, s_struct), 4)
    detail['structure'] = {'header_ok': header_ok, 'rows': len(keys), 'gold_rows': n_g,
                           'missing': miss, 'extra': extra, 'sorted': sorted_ok,
                           'non_core_leak': leak, 'dataset_ok': ds_ok,
                           'summary_schema': sum_schema, 'recommendation_schema': rec_schema}

    # ---------------- dice ---------------- #
    errs = []
    for k, r in zip(keys, agent_rows or []):
        g = gold_rows.get(k)
        if g is None:
            continue
        try:
            d = float(r['dice'])
        except Exception:
            d = 1e6
        errs.append(abs(d - g['dice']))
    cov = len(errs) / float(n_g) if n_g else 0.0
    if errs:
        arr = np.array(errs, float)
        frac_ok = float(np.mean(arr <= 0.02))
        mae = float(np.mean(np.minimum(arr, 1.0)))
        s_dice = (0.35 * min(1.0, cov) + 0.35 * frac_ok +
                  0.30 * max(0.0, min(1.0, 1.0 - max(0.0, mae - 0.01) / 0.04)))
    else:
        frac_ok, mae, s_dice = 0.0, 1.0, 0.0
    sub['dice'] = round(max(0.0, min(1.0, s_dice)), 4)
    detail['dice'] = {'matched_rows': len(errs), 'coverage': round(cov, 4),
                      'frac_within_0.02': round(frac_ok, 4), 'mae': round(mae, 6)}

    # ---------------- volumes ---------------- #
    ok = []
    for k, r in zip(keys, agent_rows or []):
        g = gold_rows.get(k)
        if g is None:
            continue
        try:
            va = float(r['ref_volume_mm3'])
            vb = float(r['pred_volume_mm3'])
        except Exception:
            ok.append(0.0)
            continue
        t1 = abs(va - g['ref_volume_mm3']) <= max(5.0, 0.02 * g['ref_volume_mm3'])
        t2 = abs(vb - g['pred_volume_mm3']) <= max(5.0, 0.02 * g['pred_volume_mm3'])
        ok.append(1.0 if (t1 and t2) else 0.0)
    sub['volume'] = round((float(np.mean(ok)) * cov) if ok else 0.0, 4)
    detail['volume'] = {'rows_checked': len(ok),
                        'frac_ok': round(float(np.mean(ok)), 4) if ok else 0.0}

    # ---------------- presence ---------------- #
    pv = []
    for k, r in zip(keys, agent_rows or []):
        g = gold_rows.get(k)
        if g is None:
            continue
        a = as_bool(r.get('present_in_ref', ''))
        b = as_bool(r.get('present_in_pred', ''))
        pv.append(1.0 if (a == g['present_in_ref'] and b == g['present_in_pred']) else 0.0)
    sub['presence'] = round((float(np.mean(pv)) * cov) if pv else 0.0, 4)
    detail['presence'] = {'rows_checked': len(pv),
                          'frac_ok': round(float(np.mean(pv)), 4) if pv else 0.0}

    # ---------------- detection F1 ---------------- #
    hits = 0
    cells = 0
    for m in ALLMAPS:
        am = None
        if isinstance(agent_sum, dict) and isinstance(agent_sum.get('models'), dict):
            am = agent_sum['models'].get(m)
        for d in DOMAINS:
            cells += 1
            if not isinstance(am, dict):
                continue
            af = am.get('detection_f1_by_domain')
            if not isinstance(af, dict):
                continue
            try:
                v = float(af.get(d))
            except Exception:
                continue
            if abs(v - gold_sum['models'][m]['detection_f1_by_domain'][d]) <= 0.01:
                hits += 1
    sub['detection_f1'] = round(hits / float(len(ALLMAPS) * len(DOMAINS)), 4)
    detail['detection_f1'] = {'cells_matched': hits, 'cells_total': len(ALLMAPS) * len(DOMAINS)}

    # ---------------- summary aggregation ---------------- #
    checks = 0
    passed = 0
    if isinstance(agent_sum, dict):
        checks += 2
        try:
            if int(agent_sum.get('n_subjects')) == gold_sum['n_subjects']:
                passed += 1
        except Exception:
            pass
        try:
            if int(agent_sum.get('n_scored_tract_pairs')) == gold_sum['n_scored_tract_pairs']:
                passed += 1
        except Exception:
            pass
        amodels = agent_sum.get('models') or {}
        for m in ALLMAPS:
            am = amodels.get(m) if isinstance(amodels, dict) else None
            gm = gold_sum['models'][m]
            for d in DOMAINS:
                checks += 1
                try:
                    if abs(float(am['mean_dice_by_domain'][d]) - gm['mean_dice_by_domain'][d]) <= 0.01:
                        passed += 1
                except Exception:
                    pass
            for key, tol in (('within_domain_mean_dice', 0.01),
                             ('cross_domain_mean_dice', 0.01),
                             ('degradation_delta', 0.015)):
                checks += 1
                try:
                    if abs(float(am[key]) - gm[key]) <= tol:
                        passed += 1
                except Exception:
                    pass
    sub['summary'] = round(passed / float(checks), 4) if checks else 0.0
    detail['summary'] = {'checks_passed': passed, 'checks_total': checks}

    # ---------------- ensemble maps ---------------- #
    ens_scores = []
    ens_detail = {}
    for s in subjects:
        sid = s['subject_id']
        p = OUT + '/ensemble/' + sid + '_ewmts.nii.gz'
        if not os.path.exists(p):
            ens_scores.append(0.0)
            ens_detail[sid] = {'present': False}
            continue
        try:
            img = nib.load(p)
        except Exception:
            ens_scores.append(0.0)
            ens_detail[sid] = {'loadable': False}
            continue
        shp = tuple(int(v) for v in img.shape[:3])
        aff = np.asarray(img.affine, float)
        geom_ok = (shp == tpl_shape) and bool(np.allclose(aff, tpl_aff, atol=1e-4))
        if not geom_ok:
            ens_scores.append(0.0)
            ens_detail[sid] = {'geometry_ok': False, 'shape': shp}
            continue
        arr = np.rint(img.get_fdata()).astype(np.int16)
        g = gold['ens'][sid]
        labels = sorted(set(int(v) for v in np.unique(g)) | set(int(v) for v in np.unique(arr)))
        labels = [lb for lb in labels if lb != 0]
        if not labels:
            ens_scores.append(0.0)
            ens_detail[sid] = {'labels': 0}
            continue
        ds = []
        for lb in labels:
            A = (g == lb)
            B = (arr == lb)
            na = int(np.count_nonzero(A))
            nb = int(np.count_nonzero(B))
            ni = int(np.count_nonzero(A & B))
            ds.append((2.0 * ni / (na + nb)) if (na + nb) > 0 else 0.0)
        macro = float(np.mean(ds))
        sc = max(0.0, min(1.0, (macro - 0.80) / 0.17))
        ens_scores.append(sc)
        ens_detail[sid] = {'macro_dice': round(macro, 4), 'score': round(sc, 4),
                           'n_labels': len(labels)}
    sub['ensemble'] = round(float(np.mean(ens_scores)) if ens_scores else 0.0, 4)
    detail['ensemble'] = ens_detail

    # ---------------- recommendation ---------------- #
    fields = ['best_within_domain_model', 'best_cross_domain_model',
              'worst_degradation_model', 'best_pediatric_model', 'ewmts_recovers_accuracy']
    rec_ok = 0
    for f in fields:
        if not isinstance(agent_rec, dict):
            continue
        v = agent_rec.get(f)
        if f == 'ewmts_recovers_accuracy':
            if isinstance(v, bool) and v == gold_rec[f]:
                rec_ok += 1
            elif isinstance(v, str) and v.strip().lower() in ('true', 'false'):
                if (v.strip().lower() == 'true') == gold_rec[f]:
                    rec_ok += 1
        else:
            if isinstance(v, str) and v.strip() == gold_rec[f]:
                rec_ok += 1
    sub['recommendation'] = round(rec_ok / float(len(fields)), 4)
    detail['recommendation'] = {'fields_ok': rec_ok, 'fields_total': len(fields)}

    score = sum(WEIGHTS[k] * sub[k] for k in WEIGHTS)
    score = max(0.0, min(1.0, score))

    print('================ WMT benchmark grader ================')
    for k in WEIGHTS:
        print('%-16s weight=%.2f score=%.4f' % (k, WEIGHTS[k], sub[k]))
    print('FINAL SCORE %.4f' % score)
    print('gold: within/cross (EWMTS) %.4f / %.4f ; n_scored_tract_pairs=%d' % (
        gold_sum['models'][ENS]['within_domain_mean_dice'],
        gold_sum['models'][ENS]['cross_domain_mean_dice'],
        gold_sum['n_scored_tract_pairs']))
    print('gold recommendation: ' + json.dumps(gold_rec))
    print('details: ' + json.dumps(detail)[:6000])

    REPORT.update({'score': score, 'sub_scores': sub, 'weights': WEIGHTS,
                   'gold_summary': gold_sum, 'gold_recommendation': gold_rec,
                   'details': detail})
    return score


if __name__ == '__main__':
    try:
        final = main()
    except Exception:
        import traceback
        traceback.print_exc()
        final = 0.0
        REPORT.update({'score': 0.0, 'error': 'grader exception'})
    try:
        os.makedirs(LOGDIR, exist_ok=True)
        with open(os.path.join(LOGDIR, 'reward.json'), 'w') as fh:
            json.dump({'score': float(round(final, 4)), 'reward': float(round(final, 4))}, fh)
        with open(os.path.join(LOGDIR, 'report.json'), 'w') as fh:
            json.dump(REPORT, fh, default=str)
    except Exception:
        import traceback
        traceback.print_exc()
