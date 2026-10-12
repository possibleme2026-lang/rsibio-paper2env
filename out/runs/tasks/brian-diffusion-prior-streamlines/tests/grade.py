#!/usr/bin/env python3
"""Grader for the BRIAN diffusion-prior streamline task.

Recomputes every metric from the artefacts written under /app/output and awards
partial credit as a weighted sum of eleven independently checked constraint
groups. Only {"score": float, "reward": float} is written to reward.json; the
per-constraint breakdown goes to stdout and to report.json.
"""
import json
import os
import sys
import traceback

import numpy as np

OUT = '/app/output'
DATA = '/app/data'
LOGD = '/logs/verifier'
os.makedirs(LOGD, exist_ok=True)

WEIGHTS = {
    'files': 0.06,
    'occupancy_valid': 0.08,
    'dice_per_pairing': 0.16,
    'median_dice': 0.10,
    'metrics_match': 0.10,
    'anti_shortcut': 0.10,
    'streamlines': 0.14,
    'curriculum': 0.08,
    'params_bounds': 0.06,
    'isotropic': 0.07,
    'dodf': 0.05,
}

METRIC_KEYS = ['dice', 'hausdorff_agreement', 'surface_dice', 'precision', 'recall']
CONN6 = np.zeros((3, 3, 3), bool)
CONN6[1, 1, 1] = True
CONN6[0, 1, 1] = CONN6[2, 1, 1] = CONN6[1, 0, 1] = True
CONN6[1, 2, 1] = CONN6[1, 1, 0] = CONN6[1, 1, 2] = True


def emit(score, detail):
    score = float(max(0.0, min(1.0, score)))
    with open(os.path.join(LOGD, 'reward.json'), 'w') as fh:
        json.dump({'score': score, 'reward': score}, fh)
    with open(os.path.join(LOGD, 'report.json'), 'w') as fh:
        json.dump(detail, fh, indent=2, default=str)
    print(json.dumps(detail, indent=2, default=str))
    print('FINAL_SCORE %.4f' % score)


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    tot = a.sum() + b.sum()
    return float(2.0 * inter / tot) if tot > 0 else 0.0


def surface(mask):
    from scipy import ndimage
    return mask & ~ndimage.binary_erosion(mask, structure=CONN6)


def surface_dice(a, b, tol):
    from scipy import ndimage
    sa, sb = surface(a), surface(b)
    if sa.sum() == 0 or sb.sum() == 0:
        return 0.0
    da = ndimage.distance_transform_edt(~sb)
    db = ndimage.distance_transform_edt(~sa)
    return 0.5 * (float((da[sa] <= tol).mean()) + float((db[sb] <= tol).mean()))


def hd_agreement(a, b, cap):
    from scipy import ndimage
    sa, sb = surface(a), surface(b)
    if sa.sum() == 0 or sb.sum() == 0:
        return 0.0
    da = ndimage.distance_transform_edt(~sb)[sa]
    db = ndimage.distance_transform_edt(~sa)[sb]
    d95 = float(np.percentile(np.concatenate([da, db]), 95.0))
    return float(max(0.0, 1.0 - min(d95, cap) / cap))


def compute_metrics(a, b, tol, cap):
    inter = float(np.logical_and(a, b).sum())
    return {
        'dice': dice(a, b),
        'precision': float(inter / a.sum()) if a.sum() > 0 else 0.0,
        'recall': float(inter / b.sum()) if b.sum() > 0 else 0.0,
        'surface_dice': surface_dice(a, b, tol),
        'hausdorff_agreement': hd_agreement(a, b, cap),
    }


def load_mask(path):
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj) > 0.5


def rasterize(points, offsets, shape):
    out = np.zeros(shape, bool)
    shp = np.array(shape)
    for k in range(len(offsets) - 1):
        seg = points[offsets[k]:offsets[k + 1]]
        if seg.shape[0] == 0:
            continue
        idx = np.clip(np.round(seg).astype(int), 0, shp - 1)
        out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out


def check_streamline_geometry(points, offsets, step_size, ang_deg, brain):
    shp = np.array(brain.shape)
    n_ok = 0
    n_tot = 0
    issues = []
    for k in range(len(offsets) - 1):
        seg = points[offsets[k]:offsets[k + 1]].astype(np.float64)
        n_tot += 1
        if seg.shape[0] < 5:
            continue
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        if d.size == 0 or d.mean() <= 0:
            continue
        step_ok = abs(d.mean() - step_size) <= 0.2 * step_size
        v = np.diff(seg, axis=0)
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.maximum(nrm, 1e-12)
        cosang = np.clip((v[:-1] * v[1:]).sum(1), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang)) if cosang.size else np.array([0.0])
        turn_ok = float(ang.max()) <= ang_deg + 5.0
        idx = np.clip(np.round(seg).astype(int), 0, shp - 1)
        inside_ok = float(brain[idx[:, 0], idx[:, 1], idx[:, 2]].mean()) >= 0.98
        if step_ok and turn_ok and inside_ok:
            n_ok += 1
        elif len(issues) < 3:
            issues.append({'k': k, 'step_ok': bool(step_ok), 'turn_ok': bool(turn_ok),
                           'inside_ok': bool(inside_ok), 'mean_step': float(d.mean()),
                           'max_turn': float(ang.max())})
    frac = float(n_ok / n_tot) if n_tot > 0 else 0.0
    return frac, issues, n_tot


def main():
    import nibabel as nib
    from scipy import ndimage

    prot = json.load(open(os.path.join(DATA, 'protocol.json')))
    pairings = list(prot['pairings'])
    pbounds = prot['parameter_bounds']
    tol = float(prot['surface_tolerance_voxels'])
    cap = float(prot['hausdorff_cap_voxels'])

    detail = {'checks': {}, 'weights': WEIGHTS}
    passed = {}

    def mark(name, ok, info=None):
        passed[name] = bool(ok)
        detail['checks'][name] = {'passed': bool(ok), 'weight': WEIGHTS[name], 'info': info}

    # ---------------- load reference + host masks ----------------
    refs, brains = {}, {}
    for pid in pairings:
        refs[pid] = load_mask(os.path.join(DATA, 'tumours', pid + '_ref.nii.gz'))
    for host in sorted({pid.split('_')[0] for pid in pairings}):
        brains[host] = load_mask(os.path.join(DATA, 'hosts', host, 'brain_mask.nii.gz'))

    # ---------------- C1: files ----------------
    need = ['metrics.json', 'params.json', 'curriculum.json', 'comparison.json', 'comparison_dodf.json']
    file_info = {'missing': []}
    for f in need:
        if not os.path.isfile(os.path.join(OUT, f)):
            file_info['missing'].append(f)
    for pid in pairings:
        for sub, ext in (('occupancy', '.nii.gz'), ('occupancy_iso', '.nii.gz'),
                         ('occupancy_tensor', '.nii.gz'), ('streamlines', '.npz')):
            p = os.path.join(OUT, sub, pid + ext)
            if not os.path.isfile(p):
                file_info['missing'].append(sub + '/' + pid + ext)
    mark('files', not file_info['missing'], file_info)

    if file_info['missing']:
        detail['note'] = 'missing artefacts; remaining checks are scored as failures'
        score = sum(WEIGHTS[k] for k, v in passed.items() if v)
        detail['score'] = score
        emit(score, detail)
        return

    occ, occ_iso, occ_ten, ref_loaded = {}, {}, {}, {}
    valid_info = {'bad': []}
    for pid in pairings:
        host = pid.split('_')[0]
        brain = brains[host]
        for tag, store in (('occupancy', occ), ('occupancy_iso', occ_iso), ('occupancy_tensor', occ_ten)):
            m = load_mask(os.path.join(OUT, tag, pid + '.nii.gz'))
            store[pid] = m
            if m.shape != (40, 40, 40):
                valid_info['bad'].append({'pairing': pid, 'tag': tag, 'why': 'shape', 'shape': list(m.shape)})
                continue
            u = np.unique(m.astype(np.uint8))
            if not set(u.tolist()).issubset({0, 1}):
                valid_info['bad'].append({'pairing': pid, 'tag': tag, 'why': 'non-binary', 'values': u.tolist()[:6]})
                continue
            if m.sum() == 0:
                valid_info['bad'].append({'pairing': pid, 'tag': tag, 'why': 'empty'})
                continue
            frac_in = float((m & brain).sum()) / float(m.sum())
            if frac_in < 0.95:
                valid_info['bad'].append({'pairing': pid, 'tag': tag, 'why': 'outside brain', 'frac_in': frac_in})
        ref_loaded[pid] = refs[pid]
    mark('occupancy_valid', not valid_info['bad'], valid_info)

    # ---------------- recomputed metrics ----------------
    recomputed = {}
    for pid in pairings:
        recomputed[pid] = compute_metrics(occ[pid], ref_loaded[pid], tol, cap)
    rec_iso = {pid: compute_metrics(occ_iso[pid], ref_loaded[pid], tol, cap) for pid in pairings}
    rec_ten = {pid: compute_metrics(occ_ten[pid], ref_loaded[pid], tol, cap) for pid in pairings}

    try:
        reported = json.load(open(os.path.join(OUT, 'metrics.json')))
        rep_by = {r['pairing_id']: r for r in reported['pairings']}
    except Exception as exc:
        reported, rep_by = None, {}
        detail['metrics_json_error'] = repr(exc)

    # ---------------- C3: per-pairing dice ----------------
    dice_info = {'per_pairing': [], 'n_pass': 0}
    for pid in pairings:
        d = recomputed[pid]['dice']
        rep = rep_by.get(pid, {}).get('dice', None)
        match = (rep is not None) and abs(float(rep) - d) <= 0.03
        ok = (d >= 0.45) and match
        dice_info['per_pairing'].append({'pairing': pid, 'dice': d, 'reported': rep,
                                         'match': bool(match), 'ok': bool(ok)})
        dice_info['n_pass'] += int(ok)
    frac_ok = dice_info['n_pass'] / float(len(pairings))
    dice_info['fraction_ok'] = frac_ok
    mark('dice_per_pairing', frac_ok >= 0.8, dice_info)

    # ---------------- C4: median dice ----------------
    med_dir = float(np.median([recomputed[p]['dice'] for p in pairings]))
    pooled_dir = float(dice(np.stack([occ[p] for p in pairings]).any(0),
                            np.stack([ref_loaded[p] for p in pairings]).any(0)))
    mark('median_dice', med_dir >= 0.55, {'median_dice': med_dir, 'pooled_dice': pooled_dir,
                                          'threshold': 0.55})

    # ---------------- C5: remaining metrics match ----------------
    mm_info = {'per_pairing': [], 'n_pass': 0}
    for pid in pairings:
        rec = recomputed[pid]
        rep = rep_by.get(pid, {})
        worst = 0.0
        missing = []
        for k in METRIC_KEYS[1:]:
            if k not in rep:
                missing.append(k)
                continue
            worst = max(worst, abs(float(rep[k]) - rec[k]))
        ok = (not missing) and worst <= 0.03
        mm_info['per_pairing'].append({'pairing': pid, 'worst_abs_diff': worst,
                                       'missing': missing, 'ok': bool(ok)})
        mm_info['n_pass'] += int(ok)
    frac_mm = mm_info['n_pass'] / float(len(pairings))
    mm_info['fraction_ok'] = frac_mm
    mark('metrics_match', frac_mm >= 0.8, mm_info)

    # ---------------- C6: anti-shortcut ----------------
    anti = {'per_pairing': [], 'n_pass': 0}
    for pid in pairings:
        a, b = occ[pid], ref_loaded[pid]
        exact = bool(np.array_equal(a, b))
        diff_frac = float((a != b).mean())
        d = recomputed[pid]['dice']
        near_copy = d >= 0.99
        ok = (not exact) and diff_frac > 0.004 and (not near_copy)
        anti['per_pairing'].append({'pairing': pid, 'exact_copy': exact,
                                    'diff_fraction': diff_frac, 'dice': d, 'ok': bool(ok)})
        anti['n_pass'] += int(ok)
    mark('anti_shortcut', anti['n_pass'] == len(pairings), anti)

    # ---------------- C7: streamlines ----------------
    try:
        params = json.load(open(os.path.join(OUT, 'params.json')))
        par_by = {p['pairing_id']: p for p in params['pairings']}
    except Exception as exc:
        params, par_by = None, {}
        detail['params_json_error'] = repr(exc)
    sl_info = {'per_pairing': [], 'n_pass': 0}
    for pid in pairings:
        host = pid.split('_')[0]
        brain = brains[host]
        p = par_by.get(pid, {})
        step_size = float(p.get('step_size', 0.0))
        ang = float(p.get('angular_threshold', 0.0))
        entry = {'pairing': pid, 'step_size': step_size, 'angular_threshold': ang}
        try:
            z = np.load(os.path.join(OUT, 'streamlines', pid + '.npz'))
            pts = np.asarray(z['points'], dtype=np.float64)
            offs = np.asarray(z['offsets'], dtype=np.int64)
            if pts.ndim != 2 or pts.shape[1] != 3 or offs.ndim != 1 or offs.size < 2:
                entry['why'] = 'bad array shapes'
                entry['ok'] = False
                sl_info['per_pairing'].append(entry)
                continue
            frac, issues, ntot = check_streamline_geometry(pts, offs, step_size, ang, brain)
            ras = rasterize(pts, offs, occ[pid].shape)
            d_ras = dice(ras, occ[pid])
            ras1 = ndimage.binary_dilation(ras, iterations=1)
            d_ras1 = dice(ras1, occ[pid])
            entry.update({'n_streamlines': ntot, 'geometry_fraction': frac,
                          'raster_dice_r0': d_ras, 'raster_dice_r1': d_ras1,
                          'examples': issues})
            ok = (frac >= 0.8) and (d_ras >= 0.75) and (d_ras1 >= 0.90) and (ntot >= 50)
            entry['ok'] = bool(ok)
            sl_info['n_pass'] += int(ok)
        except Exception as exc:
            entry.update({'why': repr(exc), 'ok': False})
        sl_info['per_pairing'].append(entry)
    mark('streamlines', sl_info['n_pass'] >= max(1, int(0.8 * len(pairings))), sl_info)

    # ---------------- C8: curriculum tightening ----------------
    try:
        curr = json.load(open(os.path.join(OUT, 'curriculum.json')))
        cur_by = {c['pairing_id']: c for c in curr['pairings']}
    except Exception as exc:
        curr, cur_by = None, {}
        detail['curriculum_json_error'] = repr(exc)
    cur_info = {'per_pairing': [], 'n_pass': 0}
    for pid in pairings:
        c = cur_by.get(pid)
        entry = {'pairing': pid}
        if not c or 'stages' not in c or len(c['stages']) != 3:
            entry['ok'] = False
            entry['why'] = 'missing three stages'
            cur_info['per_pairing'].append(entry)
            continue
        widths = []
        try:
            for st in c['stages']:
                b = st['bounds']
                widths.append(float(sum(float(b[k][1]) - float(b[k][0]) for k in b)))
        except Exception as exc:
            entry['ok'] = False
            entry['why'] = repr(exc)
            cur_info['per_pairing'].append(entry)
            continue
        ok = widths[0] > widths[1] > widths[2]
        entry.update({'widths': widths, 'ok': bool(ok)})
        cur_info['n_pass'] += int(ok)
        cur_info['per_pairing'].append(entry)
    mark('curriculum', cur_info['n_pass'] == len(pairings), cur_info)

    # ---------------- C9: parameter bounds ----------------
    pb_info = {'per_pairing': [], 'n_pass': 0}
    for pid in pairings:
        p = par_by.get(pid)
        entry = {'pairing': pid}
        if not p:
            entry['ok'] = False
            entry['why'] = 'missing params'
            pb_info['per_pairing'].append(entry)
            continue
        bad = []
        for k, (lo, hi) in pbounds.items():
            if k not in p:
                bad.append(k + ':missing')
                continue
            v = float(p[k])
            if v < lo * 0.99 or v > hi * 1.01:
                bad.append('%s=%s' % (k, v))
        n = p.get('n_seeds')
        if n is None or abs(float(n) - round(float(n))) > 1e-6 or not (300 <= float(n) <= 900):
            bad.append('n_seeds=%s' % n)
        ok = not bad
        entry.update({'bad': bad, 'ok': bool(ok)})
        pb_info['n_pass'] += int(ok)
        pb_info['per_pairing'].append(entry)
    mark('params_bounds', pb_info['n_pass'] == len(pairings), pb_info)

    # ---------------- C10: isotropic null ----------------
    med_iso = {k: float(np.median([rec_iso[p][k] for p in pairings])) for k in METRIC_KEYS}
    med_dir_all = {k: float(np.median([recomputed[p][k] for p in pairings])) for k in METRIC_KEYS}
    improved = [k for k in METRIC_KEYS if med_dir_all[k] >= med_iso[k] - 0.005]
    iso_ok = (med_dir_all['dice'] - med_iso['dice'] >= 0.05) and (len(improved) >= 3)
    try:
        cmpj = json.load(open(os.path.join(OUT, 'comparison.json')))
        cmp_ok = all(k in cmpj.get('directional', {}) for k in METRIC_KEYS) and \
                 all(k in cmpj.get('isotropic', {}) for k in METRIC_KEYS)
    except Exception:
        cmp_ok = False
    mark('isotropic', iso_ok and cmp_ok,
         {'median_directional': med_dir_all, 'median_isotropic': med_iso,
          'improved_metrics': improved, 'dice_gain': med_dir_all['dice'] - med_iso['dice'],
          'comparison_json_ok': bool(cmp_ok)})

    # ---------------- C11: single-tensor dODF ----------------
    med_ten = float(np.median([rec_ten[p]['dice'] for p in pairings]))
    diffs = {p: float(dice(occ[p], occ_ten[p])) for p in pairings}
    n_diff = sum(1 for p in pairings if diffs[p] < 0.995)
    try:
        dj = json.load(open(os.path.join(OUT, 'comparison_dodf.json')))
        dj_ok = all(k in dj.get('shore', {}) for k in METRIC_KEYS) and \
                all(k in dj.get('tensor', {}) for k in METRIC_KEYS)
    except Exception:
        dj_ok = False
    mark('dodf', (med_ten >= 0.45) and (n_diff >= 3) and dj_ok,
         {'median_tensor_dice': med_ten, 'pairs_with_distinct_maps': n_diff,
          'shore_vs_tensor_dice': diffs, 'comparison_dodf_ok': bool(dj_ok)})

    score = float(sum(WEIGHTS[k] for k, v in passed.items() if v))
    detail['passed'] = passed
    detail['score'] = score
    emit(score, detail)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        emit(0.0, {'error': repr(exc), 'trace': traceback.format_exc()})
