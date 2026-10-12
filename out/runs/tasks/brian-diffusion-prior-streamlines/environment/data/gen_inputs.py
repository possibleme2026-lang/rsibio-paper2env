#!/usr/bin/env python3
"""Deterministic generator of the synthetic BRIAN cohort.

Run once at image build time; the script is then deleted from the image so that
the held-out propagation parameters used to synthesise the reference masks are
not readable by the agent. Re-running it reproduces byte-identical inputs.
"""
import json
import os

import numpy as np
import nibabel as nib
from scipy import ndimage
from dipy.core.gtabs import gradient_table
from dipy.core.sphere import Sphere
from dipy.reconst.shore import ShoreModel

SEED = 20260730
NX = NY = NZ = 40
NDIR = 60
VOX_MM = 2.0
HOSTS = ['host01', 'host02']
TUMOURS = ['A', 'B', 'C']
HERE = os.path.dirname(os.path.abspath(__file__))
CONN6 = np.zeros((3, 3, 3), bool)
CONN6[1, 1, 1] = True
CONN6[0, 1, 1] = CONN6[2, 1, 1] = CONN6[1, 0, 1] = True
CONN6[1, 2, 1] = CONN6[1, 1, 0] = CONN6[1, 1, 2] = True


def fib_sphere(n):
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1).astype(np.float32)


SPHERE = fib_sphere(NDIR)


def make_host(seed):
    rng = np.random.default_rng(seed)
    xg, yg, zg = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing='ij')
    brain = (((xg - 19.5) / 18.0) ** 2 + ((yg - 19.5) / 17.0) ** 2 +
             ((zg - 19.5) / 18.0) ** 2) <= 1.0
    wm = ndimage.binary_erosion(brain, iterations=3)
    c = np.stack([xg, yg, zg], -1).astype(np.float64) / np.array([NX, NY, NZ], float)
    dirs = np.zeros((3, NX, NY, NZ, 3))
    wts = np.zeros((3, NX, NY, NZ))
    for k in range(3):
        b = rng.normal(size=3)
        b = b / np.linalg.norm(b)
        amp = rng.uniform(0.15, 0.40, 3)
        frq = rng.uniform(0.5, 1.5, 3)
        phs = rng.uniform(0.0, 2.0 * np.pi, 3)
        u = b.reshape(1, 1, 1, 3) + amp.reshape(1, 1, 1, 3) * np.sin(
            2.0 * np.pi * frq.reshape(1, 1, 1, 3) * c + phs.reshape(1, 1, 1, 3))
        dirs[k] = u / np.linalg.norm(u, axis=-1, keepdims=True)
        ctr = rng.uniform(0.28, 0.72, 3)
        wts[k] = np.exp(-((c - ctr) ** 2).sum(-1) / (2.0 * 0.32 ** 2))
    wts = wts / np.maximum(wts.sum(0, keepdims=True), 1e-9)
    return brain, wm, dirs, wts


def build_tensors(dirs, wts, lam1=1.7e-3, lam2=0.35e-3):
    D = np.zeros((NX, NY, NZ, 3, 3))
    eye = np.eye(3)
    for k in range(dirs.shape[0]):
        u = dirs[k]
        UU = u[..., :, None] * u[..., None, :]
        D += wts[k][..., None, None] * (lam2 * eye + (lam1 - lam2) * UU)
    return D


def build_dwi(D, rng):
    nb0 = 3
    bvals = np.concatenate([np.zeros(nb0), np.full(20, 1000.0), np.full(22, 2000.0)])
    bvecs = np.zeros((bvals.size, 3))
    for i in range(nb0, bvals.size):
        v = rng.normal(size=3)
        bvecs[i] = v / np.linalg.norm(v)
    S = np.empty((NX, NY, NZ, bvals.size), np.float32)
    for i in range(bvals.size):
        g = bvecs[i]
        gDg = np.einsum('j,xyzjk,k->xyz', g, D, g)
        S[..., i] = np.exp(-bvals[i] * gDg)
    xg, zg = np.meshgrid(np.arange(NX), np.arange(NZ), indexing='ij')
    s0 = 1200.0 * (1.0 + 0.08 * np.sin(2.0 * np.pi * xg / NX) * np.cos(2.0 * np.pi * zg / NZ))
    S = S * s0[..., None]
    sig = 1200.0 / 30.0
    n1 = rng.normal(0, sig, S.shape)
    n2 = rng.normal(0, sig, S.shape)
    return np.sqrt((S + n1) ** 2 + n2 ** 2).astype(np.float32), bvals, bvecs


def shore_odf(S, bvals, bvecs):
    gtab = gradient_table(bvals, bvecs)
    fit = ShoreModel(gtab, radial_order=4, zeta=700.0).fit(S)
    odf = np.clip(np.asarray(fit.odf(Sphere(xyz=SPHERE.astype(np.float64))), np.float32), 0.0, None)
    return odf / np.maximum(odf.sum(-1, keepdims=True), 1e-8)


def propagate(odf, wm, seeds, step, ang_deg, max_len, rng):
    shp = np.array(wm.shape)
    nsteps = int(np.ceil(max_len / step))
    K = seeds.shape[0]
    pos = seeds.astype(np.float64).copy()
    idx0 = np.clip(np.round(pos).astype(int), 0, shp - 1)
    dirs = SPHERE[np.argmax(odf[idx0[:, 0], idx0[:, 1], idx0[:, 2], :], axis=1)].astype(np.float64)
    occ = np.zeros(wm.shape, bool)
    occ[idx0[:, 0], idx0[:, 1], idx0[:, 2]] = True
    cosang = np.cos(np.deg2rad(ang_deg))
    alive = np.ones(K, bool)
    for _ in range(nsteps):
        idx = np.clip(np.round(pos).astype(int), 0, shp - 1)
        ok = alive & wm[idx[:, 0], idx[:, 1], idx[:, 2]]
        if not ok.any():
            break
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
        sel = np.clip((csum < rng.random((K, 1))).sum(1), 0, SPHERE.shape[0] - 1)
        dirs = np.where(ok[:, None], SPHERE[sel], dirs)
        pos = pos + step * dirs * ok[:, None]
        vidx = np.clip(np.round(pos).astype(int), 0, shp - 1)
        occ[vidx[ok, 0], vidx[ok, 1], vidx[ok, 2]] = True
        alive = ok
    return occ


def main():
    np.save(os.path.join(HERE, 'sphere.npy'), SPHERE)
    rng = np.random.default_rng(SEED)
    protocol = {
        'voxel_size_mm': VOX_MM,
        'grid': [NX, NY, NZ],
        'sphere_file': 'sphere.npy',
        'n_sphere_directions': NDIR,
        'seed_erosion_iterations': 2,
        'surface_tolerance_voxels': 2.0,
        'hausdorff_cap_voxels': 6.0,
        'metrics': ['dice', 'hausdorff_agreement', 'surface_dice', 'precision', 'recall'],
        'objective_weights': {'dice': 0.30, 'hausdorff_agreement': 0.20, 'surface_dice': 0.20,
                              'precision': 0.15, 'recall': 0.15},
        'parameter_bounds': {'step_size': [0.4, 1.2], 'angular_threshold': [8.0, 75.0],
                             'max_length': [8.0, 30.0], 'n_seeds': [300.0, 900.0]},
        'curriculum': [{'stage': 1, 'n_evals': 20, 'shrink': None},
                       {'stage': 2, 'n_evals': 14, 'shrink': 0.35},
                       {'stage': 3, 'n_evals': 10, 'shrink': 0.12}],
        'pairings': [],
    }
    host_data = {}
    for hi, host in enumerate(HOSTS):
        hdir = os.path.join(HERE, 'hosts', host)
        os.makedirs(hdir, exist_ok=True)
        hseed = SEED + 1000 * (hi + 1)
        brain, wm, dirs, wts = make_host(hseed)
        D = build_tensors(dirs, wts)
        S, bvals, bvecs = build_dwi(D, np.random.default_rng(hseed + 7))
        odf = shore_odf(S, bvals, bvecs)
        affine = np.diag([VOX_MM, VOX_MM, VOX_MM, 1.0])
        nib.save(nib.Nifti1Image(S.astype(np.float32), affine), os.path.join(hdir, 'dwi.nii.gz'))
        nib.save(nib.Nifti1Image(brain.astype(np.uint8), affine), os.path.join(hdir, 'brain_mask.nii.gz'))
        nib.save(nib.Nifti1Image(wm.astype(np.uint8), affine), os.path.join(hdir, 'wm_mask.nii.gz'))
        np.savetxt(os.path.join(hdir, 'bvals.txt'), bvals, fmt='%.1f')
        np.savetxt(os.path.join(hdir, 'bvecs.txt'), bvecs, fmt='%.6f')
        host_data[host] = {'brain': brain, 'wm': wm, 'odf': odf, 'affine': affine}
        print('host', host, 'built', flush=True)

    for host in HOSTS:
        hd = host_data[host]
        cand = np.argwhere(ndimage.binary_erosion(hd['wm'], structure=CONN6, iterations=5))
        order = np.random.default_rng(SEED + 500 + HOSTS.index(host)).permutation(cand.shape[0])
        chosen = []
        for i in order:
            p = cand[i]
            if all(np.linalg.norm(p - q) >= 12 for q in chosen):
                chosen.append(p)
            if len(chosen) == len(TUMOURS):
                break
        for ti, tumour in enumerate(TUMOURS):
            trng = np.random.default_rng(SEED + 31 * (HOSTS.index(host) + 1) + ti)
            centre = chosen[ti]
            xg, yg, zg = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing='ij')
            core = ((xg - centre[0]) ** 2 + (yg - centre[1]) ** 2 + (zg - centre[2]) ** 2) <= 4.0
            core &= hd['wm']
            vox = np.argwhere(core)
            step = float(trng.uniform(0.5, 0.8))
            ang = float(trng.uniform(15.0, 35.0))
            maxlen = float(trng.uniform(12.0, 22.0))
            nseed = int(trng.integers(400, 700))
            sel = trng.integers(0, vox.shape[0], size=nseed)
            seeds = vox[sel].astype(np.float64) + trng.uniform(-0.45, 0.45, (nseed, 3))
            occ = propagate(hd['odf'], hd['wm'], seeds, step, ang, maxlen, trng)
            ref = ndimage.binary_dilation(occ, structure=CONN6, iterations=1)
            shell = ref & ~occ
            ref = occ | (shell & (trng.random(shell.shape) < 0.78))
            ref = ref & hd['brain']
            pid = host + '_' + tumour
            nib.save(nib.Nifti1Image(ref.astype(np.uint8), hd['affine']),
                     os.path.join(HERE, 'tumours', pid + '_ref.nii.gz'))
            protocol['pairings'].append(pid)
            print('tumour', pid, 'voxels', int(ref.sum()), flush=True)

    with open(os.path.join(HERE, 'protocol.json'), 'w') as fh:
        json.dump(protocol, fh, indent=2)


if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'tumours'), exist_ok=True)
    main()
