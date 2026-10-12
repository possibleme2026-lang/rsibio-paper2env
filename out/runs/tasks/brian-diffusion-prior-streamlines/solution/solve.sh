#!/bin/bash
set -euo pipefail
mkdir -p /app/output/occupancy /app/output/occupancy_iso /app/output/occupancy_tensor /app/output/streamlines

cat > /app/solve.py <<'PYEOF'
import json, os, zlib
import numpy as np
import nibabel as nib
from scipy import ndimage
from dipy.core.gtabs import gradient_table
from dipy.core.sphere import Sphere
from dipy.reconst.shore import ShoreModel
from dipy.reconst.dti import TensorModel

DATA = '/app/data'
OUT = '/app/output'
for d in ['occupancy', 'occupancy_iso', 'occupancy_tensor', 'streamlines']:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

prot = json.load(open(os.path.join(DATA, 'protocol.json')))
PAIRINGS = prot['pairings']
PB = prot['parameter_bounds']
TOL = float(prot['surface_tolerance_voxels'])
CAP = float(prot['hausdorff_cap_voxels'])
W = prot['objective_weights']
SPHERE = np.load(os.path.join(DATA, 'sphere.npy')).astype(np.float64)
ND = SPHERE.shape[0]
CONN6 = np.zeros((3, 3, 3), bool)
CONN6[1, 1, 1] = True
CONN6[0, 1, 1] = CONN6[2, 1, 1] = CONN6[1, 0, 1] = True
CONN6[1, 2, 1] = CONN6[1, 1, 0] = CONN6[1, 1, 2] = True


def surf(m):
    return m & ~ndimage.binary_erosion(m, structure=CONN6)


def metrics(a, b):
    inter = float(np.logical_and(a, b).sum())
    sa, sb = surf(a), surf(b)
    out = {}
    tot = a.sum() + b.sum()
    out['dice'] = float(2.0 * inter / tot) if tot > 0 else 0.0
    out['precision'] = float(inter / a.sum()) if a.sum() > 0 else 0.0
    out['recall'] = float(inter / b.sum()) if b.sum() > 0 else 0.0
    if sa.sum() > 0 and sb.sum() > 0:
        da = ndimage.distance_transform_edt(~sb)
        db = ndimage.distance_transform_edt(~sa)
        out['surface_dice'] = 0.5 * (float((da[sa] <= TOL).mean()) + float((db[sb] <= TOL).mean()))
        d95 = float(np.percentile(np.concatenate([da[sa], db[sb]]), 95.0))
        out['hausdorff_agreement'] = float(max(0.0, 1.0 - min(d95, CAP) / CAP))
    else:
        out['surface_dice'] = 0.0
        out['hausdorff_agreement'] = 0.0
    return out


def objective(m):
    return float(sum(W[k] * m[k] for k in W))


def propagate(odf, wm, seeds, step, ang_deg, max_len, rng, isotropic=False):
    shp = np.array(wm.shape)
    nsteps = int(np.ceil(max_len / step))
    K = seeds.shape[0]
    pos = seeds.astype(np.float64).copy()
    idx0 = np.clip(np.round(pos).astype(int), 0, shp - 1)
    if isotropic or odf is None:
        v = rng.normal(size=(K, 3))
        dirs = v / np.linalg.norm(v, axis=1, keepdims=True)
    else:
        p0 = odf[idx0[:, 0], idx0[:, 1], idx0[:, 2], :]
        dirs = SPHERE[np.argmax(p0, axis=1)]
    pts = np.zeros((K, nsteps + 1, 3), np.float32)
    pts[:, 0, :] = pos
    occ = np.zeros(wm.shape, bool)
    occ[idx0[:, 0], idx0[:, 1], idx0[:, 2]] = True
    astep = np.zeros(K, np.int32)
    cosang = np.cos(np.deg2rad(ang_deg))
    alive = np.ones(K, bool)
    for t in range(nsteps):
        idx = np.clip(np.round(pos).astype(int), 0, shp - 1)
        ok = alive & wm[idx[:, 0], idx[:, 1], idx[:, 2]]
        if not ok.any():
            break
        if isotropic or odf is None:
            p = np.ones((K, ND))
        else:
            p = odf[idx[:, 0], idx[:, 1], idx[:, 2], :].astype(np.float64)
        good = (SPHERE @ dirs.T >= cosang).T
        p = p * good
        tot = p.sum(1, keepdims=True)
        bad = tot[:, 0] <= 1e-12
        if bad.any():
            p[bad] = good[bad]
            tot = p.sum(1, keepdims=True)
        p = p / np.maximum(tot, 1e-12)
        csum = np.cumsum(p, axis=1)
        u = rng.random((K, 1))
        sel = np.clip((csum < u).sum(1), 0, ND - 1)
        dirs = np.where(ok[:, None], SPHERE[sel], dirs)
        pos = pos + step * dirs * ok[:, None]
        pts[:, t + 1, :] = pos
        vidx = np.clip(np.round(pos).astype(int), 0, shp - 1)
        occ[vidx[ok, 0], vidx[ok, 1], vidx[ok, 2]] = True
        astep[ok] = t + 1
        alive = ok
    return occ, pts, astep


hosts = {}
for host in sorted({p.split('_')[0] for p in PAIRINGS}):
    base = os.path.join(DATA, 'hosts', host)
    dwi = np.asarray(nib.load(os.path.join(base, 'dwi.nii.gz')).dataobj, dtype=np.float64)
    bvals = np.loadtxt(os.path.join(base, 'bvals.txt'))
    bvecs = np.loadtxt(os.path.join(base, 'bvecs.txt'))
    wm = np.asarray(nib.load(os.path.join(base, 'wm_mask.nii.gz')).dataobj) > 0
    brain = np.asarray(nib.load(os.path.join(base, 'brain_mask.nii.gz')).dataobj) > 0
    gtab = gradient_table(bvals, bvecs)
    sph = Sphere(xyz=SPHERE.astype(np.float32))
    so = np.clip(np.asarray(ShoreModel(gtab, radial_order=4, zeta=700.0).fit(dwi).odf(sph), np.float32), 0, None)
    so = so / np.maximum(so.sum(-1, keepdims=True), 1e-8)
    to = np.clip(np.asarray(TensorModel(gtab).fit(dwi).odf(sph), np.float32), 0, None)
    to = to / np.maximum(to.sum(-1, keepdims=True), 1e-8)
    hosts[host] = {'wm': wm, 'brain': brain, 'shore': so, 'tensor': to, 'affine': nib.load(os.path.join(base, 'wm_mask.nii.gz')).affine}
    print('reconstructed dODFs for', host, flush=True)

params_out, curriculum_out, metrics_out = [], [], []
comp_dir, comp_iso, comp_ten = {}, {}, {}

for pid in PAIRINGS:
    host = pid.split('_')[0]
    H = hosts[host]
    wm, brain, odf = H['wm'], H['brain'], H['shore']
    ref = np.asarray(nib.load(os.path.join(DATA, 'tumours', pid + '_ref.nii.gz')).dataobj) > 0
    core = ndimage.binary_erosion(ref, structure=CONN6, iterations=int(prot['seed_erosion_iterations'])) & wm
    corevox = np.argwhere(core)
    if corevox.shape[0] == 0:
        corevox = np.argwhere(ref & wm)
    if corevox.shape[0] == 0:
        corevox = np.argwhere(wm)
    base_seed = zlib.crc32(pid.encode()) % (2 ** 31)

    def run(p, seed, odf_use, iso=False):
        r = np.random.default_rng(seed)
        n = int(round(float(p['n_seeds'])))
        sel = r.integers(0, corevox.shape[0], size=n)
        seeds = corevox[sel].astype(np.float64) + r.uniform(-0.45, 0.45, (n, 3))
        return propagate(odf_use, wm, seeds, float(p['step_size']), float(p['angular_threshold']),
                         float(p['max_length']), r, isotropic=iso)

    bounds = {k: [float(v[0]), float(v[1])] for k, v in PB.items()}
    cur = {k: list(v) for k, v in bounds.items()}
    rng = np.random.default_rng(base_seed)
    best, best_obj, stage_log = None, -1.0, []
    for st in prot['curriculum']:
        sh = st.get('shrink')
        if sh is not None and best is not None:
            for k in cur:
                lo = max(bounds[k][0], best[k] * (1.0 - sh))
                hi = min(bounds[k][1], best[k] * (1.0 + sh))
                if hi <= lo:
                    hi = min(bounds[k][1], lo + 1e-3)
                cur[k] = [lo, hi]
        for i in range(int(st['n_evals'])):
            cand = {k: float(rng.uniform(cur[k][0], cur[k][1])) for k in cur}
            occ_c, _, _ = run(cand, base_seed + 17 * st['stage'] + i, odf)
            o = objective(metrics(occ_c, ref))
            if o > best_obj:
                best_obj, best = o, dict(cand)
        stage_log.append({'stage': int(st['stage']), 'n_evals': int(st['n_evals']),
                          'bounds': {k: [float(cur[k][0]), float(cur[k][1])] for k in cur},
                          'best_objective': float(best_obj)})
        print(pid, 'stage', st['stage'], 'best', round(best_obj, 4), flush=True)

    best['n_seeds'] = int(round(best['n_seeds']))
    occ, pts, astep = run(best, base_seed + 9001, odf)
    mS = metrics(occ, ref)
    occ_i, _, _ = run(best, base_seed + 9002, odf, iso=True)
    mI = metrics(occ_i, ref)
    occ_t, _, _ = run(best, base_seed + 9003, H['tensor'])
    mT = metrics(occ_t, ref)

    affine = H['affine']
    nib.save(nib.Nifti1Image(occ.astype(np.uint8), affine), os.path.join(OUT, 'occupancy', pid + '.nii.gz'))
    nib.save(nib.Nifti1Image(occ_i.astype(np.uint8), affine), os.path.join(OUT, 'occupancy_iso', pid + '.nii.gz'))
    nib.save(nib.Nifti1Image(occ_t.astype(np.uint8), affine), os.path.join(OUT, 'occupancy_tensor', pid + '.nii.gz'))

    keep = np.where(astep >= 4)[0]
    segs = [pts[k, :astep[k] + 1] for k in keep]
    if segs:
        allpts = np.concatenate(segs, 0).astype(np.float32)
    else:
        allpts = np.zeros((0, 3), np.float32)
    offsets = np.zeros(len(segs) + 1, np.int32)
    offsets[1:] = np.cumsum([len(s) for s in segs])
    np.savez_compressed(os.path.join(OUT, 'streamlines', pid + '.npz'), points=allpts, offsets=offsets)

    params_out.append({'pairing_id': pid, 'host': host, 'step_size': float(best['step_size']),
                       'angular_threshold': float(best['angular_threshold']),
                       'max_length': float(best['max_length']), 'n_seeds': int(best['n_seeds']),
                       'objective': float(objective(mS))})
    curriculum_out.append({'pairing_id': pid, 'stages': stage_log})
    metrics_out.append({'pairing_id': pid, **mS})
    comp_dir[pid] = mS
    comp_iso[pid] = mI
    comp_ten[pid] = mT


def median_block(store):
    return {k: float(np.median([store[p][k] for p in PAIRINGS])) for k in
            ['dice', 'hausdorff_agreement', 'surface_dice', 'precision', 'recall']}


med_dir = median_block(comp_dir)
med_iso = median_block(comp_iso)
med_ten = median_block(comp_ten)
improved = [k for k in ['dice', 'hausdorff_agreement', 'surface_dice', 'precision', 'recall']
            if med_dir[k] >= med_iso[k]]

json.dump({'pairings': params_out}, open(os.path.join(OUT, 'params.json'), 'w'), indent=2)
json.dump({'pairings': curriculum_out}, open(os.path.join(OUT, 'curriculum.json'), 'w'), indent=2)
json.dump({'pairings': metrics_out,
           'median_dice': float(np.median([m['dice'] for m in metrics_out])),
           'pooled_dice': float(np.median([m['dice'] for m in metrics_out]))},
          open(os.path.join(OUT, 'metrics.json'), 'w'), indent=2)
json.dump({'directional': med_dir, 'isotropic': med_iso, 'improved_metrics': improved},
          open(os.path.join(OUT, 'comparison.json'), 'w'), indent=2)
json.dump({'shore': med_dir, 'tensor': med_ten},
          open(os.path.join(OUT, 'comparison_dodf.json'), 'w'), indent=2)
print('done', med_dir, flush=True)
PYEOF

cd /app && python /app/solve.py
