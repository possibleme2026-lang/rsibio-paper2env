#!/bin/bash
set -euo pipefail
mkdir -p /app/output/ensemble
cat > /tmp/run_wmt_benchmark.py <<'PY'
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
MODELS = ['cnn_a', 'cnn_b', 'transformer', 'unet3d']
ENS = 'EWMTS'
ALLMAPS = MODELS + [ENS]
MODEL_ORDER = dict((m, i) for i, m in enumerate(ALLMAPS))
DOMAINS = ['adult_ref', 'adult_alt', 'pediatric']
HEADER = ['subject_id', 'dataset', 'model', 'tract_id', 'tract_name', 'dice',
          'ref_volume_mm3', 'pred_volume_mm3', 'present_in_ref', 'present_in_pred']

os.makedirs(OUT + '/ensemble', exist_ok=True)
tpl = nib.load(DATA + '/atlas/template.nii.gz')
tpl_aff = np.asarray(tpl.affine, float)
vox_vol = float(abs(np.linalg.det(tpl_aff[:3, :3])))

atlas = list(csv.DictReader(open(DATA + '/atlas/tract_labels.csv')))
core = [r for r in atlas if r['benchmark_core'].strip().lower() in ('yes', 'true', '1')]
core.sort(key=lambda r: int(r['label_value']))
name_by_label = dict((int(r['label_value']), r['tract_name']) for r in atlas)
subjects = list(csv.DictReader(open(DATA + '/subjects.csv')))

st = generate_binary_structure(3, 1)
rows = []

for s in subjects:
    sid = s['subject_id']
    ds = s['dataset']
    ref = np.rint(nib.load(DATA + '/refs/' + sid + '_ref.nii.gz').get_fdata()).astype(np.int16)
    preds = {}
    for m in MODELS:
        img = nib.load(DATA + '/native/' + sid + '/pred_' + m + '.nii.gz')
        preds[m] = np.rint(resample_from_to(img, tpl, order=0).get_fdata()).astype(np.int16)

    a, b, c, d = preds[MODELS[0]], preds[MODELS[1]], preds[MODELS[2]], preds[MODELS[3]]
    cand = np.zeros(ref.shape, np.int16)
    for x, y, z in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
        eq = (x == y) & (y == z)
        cand[eq] = x[eq]
    ens = np.zeros(ref.shape, np.int16)
    for lb in np.unique(cand):
        if int(lb) == 0:
            continue
        m = (cand == lb)
        lab, n = cc_label(m, structure=st)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        keep = (sizes * vox_vol) >= 150.0
        keep[0] = False
        sel = keep[lab]
        if sel.any():
            ens[sel] = lb
    nib.save(nib.Nifti1Image(ens, tpl_aff), OUT + '/ensemble/' + sid + '_ewmts.nii.gz')

    maps = dict(preds)
    maps[ENS] = ens
    for m in ALLMAPS:
        pm = maps[m]
        for r in core:
            lb = int(r['label_value'])
            A = (ref == lb)
            B = (pm == lb)
            na = int(np.count_nonzero(A))
            nb = int(np.count_nonzero(B))
            ni = int(np.count_nonzero(A & B))
            dv = (2.0 * ni / (na + nb)) if (na + nb) > 0 else 0.0
            rows.append(dict(subject_id=sid, dataset=ds, model=m, tract_id=lb,
                             tract_name=name_by_label[lb], dice=dv,
                             ref_volume_mm3=na * vox_vol, pred_volume_mm3=nb * vox_vol,
                             present_in_ref=bool(na * vox_vol >= 250.0),
                             present_in_pred=bool(nb * vox_vol >= 250.0)))

rows.sort(key=lambda r: (r['subject_id'], MODEL_ORDER[r['model']], r['tract_id']))
with open(OUT + '/per_tract_metrics.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(HEADER)
    for r in rows:
        w.writerow([r['subject_id'], r['dataset'], r['model'], r['tract_id'], r['tract_name'],
                    '%.6f' % r['dice'], '%.2f' % r['ref_volume_mm3'],
                    '%.2f' % r['pred_volume_mm3'],
                    'true' if r['present_in_ref'] else 'false',
                    'true' if r['present_in_pred'] else 'false'])

summary = {'n_subjects': len(subjects),
           'n_scored_tract_pairs': len(set((r['subject_id'], r['tract_id']) for r in rows
                                           if r['present_in_ref'])),
           'models': {}}
for m in ALLMAPS:
    md = {}
    f1s = {}
    for dom in DOMAINS:
        vals = [r['dice'] for r in rows if r['model'] == m and r['dataset'] == dom and r['present_in_ref']]
        md[dom] = round(float(np.mean(vals)) if vals else 0.0, 4)
        tp = fp = fn = 0
        for r in rows:
            if r['model'] != m or r['dataset'] != dom:
                continue
            if r['present_in_ref'] and r['present_in_pred']:
                tp += 1
            elif r['present_in_pred'] and not r['present_in_ref']:
                fp += 1
            elif r['present_in_ref'] and not r['present_in_pred']:
                fn += 1
        den = 2 * tp + fp + fn
        f1s[dom] = round((2.0 * tp / den) if den > 0 else 0.0, 4)
    within = [r['dice'] for r in rows if r['model'] == m and r['dataset'] == 'adult_ref' and r['present_in_ref']]
    cross = [r['dice'] for r in rows if r['model'] == m and r['dataset'] in ('adult_alt', 'pediatric') and r['present_in_ref']]
    wval = round(float(np.mean(within)) if within else 0.0, 4)
    cval = round(float(np.mean(cross)) if cross else 0.0, 4)
    summary['models'][m] = {'mean_dice_by_domain': md, 'detection_f1_by_domain': f1s,
                            'within_domain_mean_dice': wval,
                            'cross_domain_mean_dice': cval,
                            'degradation_delta': round(wval - cval, 4)}

with open(OUT + '/benchmark_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2)

ms = summary['models']
rec = {'best_within_domain_model': sorted(ALLMAPS, key=lambda m: (-ms[m]['within_domain_mean_dice'], m))[0],
       'best_cross_domain_model': sorted(ALLMAPS, key=lambda m: (-ms[m]['cross_domain_mean_dice'], m))[0],
       'worst_degradation_model': sorted(ALLMAPS, key=lambda m: (-ms[m]['degradation_delta'], m))[0],
       'best_pediatric_model': sorted(ALLMAPS, key=lambda m: (-ms[m]['mean_dice_by_domain']['pediatric'], m))[0],
       'ewmts_recovers_accuracy': bool(all(ms[ENS]['cross_domain_mean_dice'] > ms[m]['cross_domain_mean_dice'] for m in MODELS))}
with open(OUT + '/recommendation.json', 'w') as fh:
    json.dump(rec, fh, indent=2)
print('wrote', len(rows), 'rows')
print(json.dumps(rec))
PY
python /tmp/run_wmt_benchmark.py
ls -l /app/output
