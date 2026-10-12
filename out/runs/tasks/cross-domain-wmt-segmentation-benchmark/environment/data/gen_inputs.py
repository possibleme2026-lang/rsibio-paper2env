#!/usr/bin/env python3
'''Deterministic generator of the synthetic cross-domain WMT segmentation benchmark.

Creates, relative to this script:
  atlas/tract_labels.csv                  24-tract atlas (20 benchmark-core tracts)
  atlas/template.nii.gz                   template-space grid (112^3, 1.6 mm)
  refs/<subject>_ref.nii.gz               reference label maps (template space)
  native/<subject>/pred_<model>.nii.gz    four model predictions (native space)
  subjects.csv                            cohort metadata
Seed 20260730; all geometry is parametric tubes in world (mm) coordinates that
are rasterised onto each subject-specific affine, so different cohorts really
have different voxel grids.
'''
import csv
import math
import os

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260730

PAIR_GEOM = {
    'CST': ([(-22, -16, 60), (-20, -22, 40), (-15, -26, 12), (-9, -28, -18), (-7, -30, -44)], 3.6),
    'CA': ([(-11, -44, 30), (-13, -12, 42), (-15, 24, 28), (-17, 44, 6)], 3.0),
    'FX': ([(-22, -34, -10), (-11, -27, 0), (-3, -19, 9)], 2.8),
    'OR': ([(-30, -72, -6), (-38, -40, 14), (-32, -14, 30)], 4.2),
    'AF': ([(-45, 22, 10), (-50, -4, 20), (-48, -30, 24), (-40, -52, 14)], 3.4),
    'UF': ([(-30, 26, -12), (-36, 8, -4), (-32, -10, -22)], 3.0),
    'ILF': ([(-40, -64, -16), (-43, -30, -10), (-35, -6, -6)], 3.4),
    'SCP': ([(-13, -44, -32), (-8, -30, -16)], 3.4),
    'TPT': ([(-35, -42, -26), (-30, -16, -30)], 3.0),
    'STFO': ([(-38, 16, -14), (-30, -14, -26)], 3.0),
}
MID_GEOM = {
    'CC_1': ([(-30, 32, 6), (0, 32, 8), (30, 32, 6)], 5.0),
    'CC_2': ([(-34, 2, 26), (0, 2, 28), (34, 2, 26)], 4.6),
    'CC_3': ([(-28, -42, 16), (0, -42, 18), (28, -42, 16)], 5.0),
    'MCP': ([(-20, -44, -38), (0, -46, -38), (20, -44, -38)], 5.0),
}
DOMAINS = [
    ('adult_ref', 5, (96, 96, 96), (1.9, 1.9, 2.0), (-91.2, -91.2, -96.0), (0.0, 0.0, 0.0), 'ScannerA_3T'),
    ('adult_alt', 5, (104, 88, 80), (1.8, 1.9, 2.3), (-93.6, -83.6, -92.0), (0.0, 0.0, 4.0), 'ScannerB_3T'),
    ('pediatric', 5, (80, 80, 80), (2.2, 2.2, 2.5), (-88.0, -88.0, -100.0), (0.0, 3.0, 0.0), 'ScannerC_1_5T'),
]
TPL_SHAPE = (112, 112, 112)
TPL_AFFINE = np.array([[1.6, 0.0, 0.0, -89.6],
                       [0.0, 1.6, 0.0, -89.6],
                       [0.0, 0.0, 1.6, -89.6],
                       [0.0, 0.0, 0.0, 1.0]])
MODELS = ['cnn_a', 'cnn_b', 'transformer', 'unet3d']

QUALITY = {
    'cnn_a': {
        'adult_ref': dict(sigma=1.0, bias=-0.010, flip=0.015, noise_thr=2.6, miss=0.00, halluc=0.01),
        'adult_alt': dict(sigma=1.5, bias=0.040, flip=0.060, noise_thr=2.0, miss=0.05, halluc=0.05),
        'pediatric': dict(sigma=2.1, bias=0.090, flip=0.150, noise_thr=1.7, miss=0.13, halluc=0.10),
    },
    'cnn_b': {
        'adult_ref': dict(sigma=1.2, bias=0.020, flip=0.030, noise_thr=2.4, miss=0.01, halluc=0.01),
        'adult_alt': dict(sigma=1.5, bias=0.010, flip=0.070, noise_thr=1.9, miss=0.07, halluc=0.05),
        'pediatric': dict(sigma=1.9, bias=0.030, flip=0.160, noise_thr=1.7, miss=0.11, halluc=0.10),
    },
    'transformer': {
        'adult_ref': dict(sigma=1.1, bias=0.010, flip=0.025, noise_thr=2.4, miss=0.01, halluc=0.02),
        'adult_alt': dict(sigma=1.7, bias=-0.030, flip=0.080, noise_thr=1.9, miss=0.09, halluc=0.06),
        'pediatric': dict(sigma=2.3, bias=-0.060, flip=0.170, noise_thr=1.7, miss=0.15, halluc=0.12),
    },
    'unet3d': {
        'adult_ref': dict(sigma=1.3, bias=0.030, flip=0.035, noise_thr=2.2, miss=0.02, halluc=0.02),
        'adult_alt': dict(sigma=1.8, bias=0.050, flip=0.090, noise_thr=1.8, miss=0.10, halluc=0.07),
        'pediatric': dict(sigma=2.2, bias=0.060, flip=0.180, noise_thr=1.6, miss=0.16, halluc=0.12),
    },
}


def build_atlas():
    atlas = []
    lb = 1
    pairs = [('CST', 'motor'), ('CA', 'limbic'), ('FX', 'limbic'), ('OR', 'visual'),
             ('AF', 'language'), ('UF', 'language'), ('ILF', 'visual'), ('SCP', 'cerebellar')]
    for key, cat in pairs:
        for hemi, sign in (('left', 1), ('right', -1)):
            atlas.append(dict(label=lb, name=key + '_' + hemi, geom=(key, sign),
                              hemi=hemi, category=cat, core=True))
            lb += 1
    for key, cat in (('CC_1', 'commissural'), ('CC_2', 'commissural'),
                     ('CC_3', 'commissural'), ('MCP', 'cerebellar')):
        atlas.append(dict(label=lb, name=key, geom=(key, 1), hemi='midline',
                          category=cat, core=True))
        lb += 1
    for key, cat in (('TPT', 'extended'), ('STFO', 'extended')):
        for hemi, sign in (('left', 1), ('right', -1)):
            atlas.append(dict(label=lb, name=key + '_' + hemi, geom=(key, sign),
                              hemi=hemi, category=cat, core=False))
            lb += 1
    return atlas


ATLAS = build_atlas()


def sample_polyline(pts, step=1.2):
    pts = np.asarray(pts, float)
    seg = np.diff(pts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    if total <= 0.0:
        return pts
    n = max(int(total / step) + 2, 4)
    tvals = np.linspace(0.0, total, n)
    out = np.empty((n, 3), float)
    for i, tv in enumerate(tvals):
        k = int(np.searchsorted(cum, tv, side='right') - 1)
        k = min(max(k, 0), len(seg) - 1)
        f = (tv - cum[k]) / max(seglen[k], 1e-9)
        out[i] = pts[k] + f * seg[k]
    return out


def subject_tracts(rng, domain):
    tracts = {}
    for e in ATLAS:
        key, sign = e['geom']
        pts, rad = PAIR_GEOM[key] if key in PAIR_GEOM else MID_GEOM[key]
        p = np.array(pts, float)
        p[:, 0] = p[:, 0] * sign
        p = p + rng.normal(0.0, 2.6, p.shape)
        r = float(rad * (1.0 + rng.normal(0.0, 0.09)))
        present = True
        if not e['core']:
            present = rng.random() > 0.10
        elif key in ('FX', 'SCP'):
            present = rng.random() > (0.12 if domain == 'pediatric' else 0.05)
        if present:
            tracts[e['label']] = (sample_polyline(p), max(r, 1.2))
    return tracts


def make_affine(vox, origin, rot_deg, rng):
    rx, ry, rz = np.deg2rad(np.asarray(rot_deg, float) + rng.normal(0.0, 1.2, 3))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    aff = np.eye(4)
    aff[:3, :3] = (Rz @ Ry @ Rx) @ np.diag(np.asarray(vox, float))
    aff[:3, 3] = np.asarray(origin, float) + rng.normal(0.0, 1.5, 3)
    return aff


def rasterize(shape, affine, tracts):
    out = np.zeros(shape, np.int16)
    best = np.full(shape, np.inf, np.float32)
    inv = np.linalg.inv(affine)
    for lb, (path, r) in tracts.items():
        lo = path.min(0) - r - 1.5
        hi = path.max(0) + r + 1.5
        corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        vox = (inv @ corners.T).T[:, :3]
        i0 = max(int(np.floor(vox[:, 0].min())), 0)
        i1 = min(int(np.ceil(vox[:, 0].max())) + 1, shape[0])
        j0 = max(int(np.floor(vox[:, 1].min())), 0)
        j1 = min(int(np.ceil(vox[:, 1].max())) + 1, shape[1])
        k0 = max(int(np.floor(vox[:, 2].min())), 0)
        k1 = min(int(np.ceil(vox[:, 2].max())) + 1, shape[2])
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue
        ii, jj, kk = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1),
                                 np.arange(k0, k1), indexing='ij')
        idx = np.stack([ii.ravel().astype(float), jj.ravel().astype(float),
                        kk.ravel().astype(float), np.ones(ii.size)])
        world = (affine @ idx)[:3]
        d2 = np.full(world.shape[1], np.inf, np.float32)
        for p in path:
            d2 = np.minimum(d2, ((world.T - p) ** 2).sum(1))
        dist = np.sqrt(d2).reshape(i1 - i0, j1 - j0, k1 - k0)
        m = dist <= r
        sub_best = best[i0:i1, j0:j1, k0:k1]
        upd = m & (dist < sub_best)
        sub_best[upd] = dist[upd]
        sub_out = out[i0:i1, j0:j1, k0:k1]
        sub_out[upd] = lb
    return out


def perturb(mask, q, rng):
    idxs = np.argwhere(mask)
    if idxs.size == 0:
        return mask
    lo = np.maximum(idxs.min(0) - 9, 0)
    hi = np.minimum(idxs.max(0) + 10, np.array(mask.shape))
    sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    sub = mask[sl]
    f = gaussian_filter(sub.astype(np.float32), sigma=q['sigma'])
    m = f > (0.5 + q['bias'])
    if q['noise_thr'] < 5.0:
        n = gaussian_filter(rng.standard_normal(sub.shape).astype(np.float32), sigma=2.2)
        m = m | (n > q['noise_thr'])
    if q['flip'] > 0:
        m = np.logical_xor(m, rng.random(sub.shape) < q['flip'])
    out = np.zeros(mask.shape, bool)
    out[sl] = m
    return out


def make_prediction(nref, model, domain, rng, halluc):
    q = dict(QUALITY[model][domain])
    q['bias'] = q['bias'] + rng.normal(0.0, 0.03)
    pred = nref.copy()
    for e in ATLAS:
        lb = e['label']
        mask = (nref == lb)
        if mask.any():
            if rng.random() < q['miss']:
                pred[mask] = 0
                continue
            newm = perturb(mask, q, rng)
            lost = mask & (~newm) & (pred == lb)
            pred[lost] = 0
            gain = newm & (pred == 0)
            pred[gain] = lb
        elif lb in halluc:
            if rng.random() < q['halluc']:
                hm = halluc[lb] & (pred == 0)
                pred[hm] = lb
    return pred


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(HERE + '/atlas', exist_ok=True)
    os.makedirs(HERE + '/refs', exist_ok=True)
    os.makedirs(HERE + '/native', exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(TPL_SHAPE, np.int16), TPL_AFFINE),
             HERE + '/atlas/template.nii.gz')
    with open(HERE + '/atlas/tract_labels.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['label_value', 'tract_name', 'hemisphere', 'category', 'benchmark_core'])
        for e in ATLAS:
            w.writerow([e['label'], e['name'], e['hemi'], e['category'],
                        'yes' if e['core'] else 'no'])
    meta = []
    n = 0
    for domain, count, shape, vox, origin, rot, scanner in DOMAINS:
        for i in range(count):
            n += 1
            sid = 'sub_%02d' % n
            aff = make_affine(vox, origin, rot, rng)
            tracts = subject_tracts(rng, domain)
            ref = rasterize(TPL_SHAPE, TPL_AFFINE, tracts)
            nib.save(nib.Nifti1Image(ref, TPL_AFFINE), HERE + '/refs/' + sid + '_ref.nii.gz')
            nref = rasterize(shape, aff, tracts)
            os.makedirs(HERE + '/native/' + sid, exist_ok=True)
            present = set(int(v) for v in np.unique(nref) if int(v) != 0)
            halluc = {}
            for e in ATLAS:
                if e['label'] in present:
                    continue
                key, sign = e['geom']
                pts, rad = PAIR_GEOM[key] if key in PAIR_GEOM else MID_GEOM[key]
                p = np.array(pts, float)
                p[:, 0] = p[:, 0] * sign
                halluc[e['label']] = rasterize(shape, aff,
                                               {e['label']: (sample_polyline(p), rad)})
            for m in MODELS:
                pm = make_prediction(nref, m, domain, rng, halluc)
                nib.save(nib.Nifti1Image(pm, aff),
                         HERE + '/native/' + sid + '/pred_' + m + '.nii.gz')
            age = round(float(rng.uniform(6.0, 12.0) if domain == 'pediatric'
                              else rng.uniform(22.0, 46.0)), 1)
            meta.append([sid, domain, 'pediatric' if domain == 'pediatric' else 'adult',
                         age, 'F' if rng.random() < 0.5 else 'M', scanner,
                         '%gx%gx%g' % tuple(vox)])
    with open(HERE + '/subjects.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['subject_id', 'dataset', 'group', 'age_years', 'sex', 'scanner',
                    'voxel_size_mm'])
        for row in meta:
            w.writerow(row)


if __name__ == '__main__':
    main()
