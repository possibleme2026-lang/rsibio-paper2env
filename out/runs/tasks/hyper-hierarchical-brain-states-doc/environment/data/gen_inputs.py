#!/usr/bin/env python3
"""Deterministic generator of the synthetic DOC resting-state fMRI cohort (seed 20260730)."""
import argparse
import csv
import os

import numpy as np

SEED = 20260730
TR = 2.0
N_TIME = 300
N_PER_GROUP = 15
GROUPS = ('CNT', 'MCS', 'UWS')
NETWORKS = (('FRO', 20), ('PAR', 15), ('TEM', 15), ('OCC', 10),
            ('CIN', 10), ('THA', 10), ('INS', 10), ('DMN', 10))
PROFILE = {
    0: {'OCC': 0.0, 'PAR': 5.0, 'TEM': 4.5, 'CIN': 3.0, 'DMN': 2.0, 'THA': 1.0, 'FRO': 1.0, 'INS': 1.0},
    1: {'OCC': 0.0, 'PAR': 1.0, 'TEM': 2.0, 'CIN': 3.0, 'DMN': 4.0, 'THA': 6.0, 'FRO': 7.0, 'INS': 7.0},
    2: {'OCC': 2.5, 'PAR': 2.5, 'TEM': 2.5, 'CIN': 2.5, 'DMN': 2.5, 'THA': 2.5, 'FRO': 2.5, 'INS': 2.5},
    3: {'OCC': 0.0, 'PAR': 2.0, 'TEM': 5.0, 'CIN': 6.0, 'DMN': 5.5, 'THA': 2.0, 'FRO': 2.0, 'INS': 1.5},
}
GROUP_SCALE = {'CNT': 0.92, 'MCS': 0.80, 'UWS': 0.68}
HYPER_DWELL = {'CNT': 12.0, 'MCS': 24.0, 'UWS': 42.0}
P_HYPER = {'CNT': 0.27, 'MCS': 0.32, 'UWS': 0.34}
OTHER_DWELL = 40.0
HYPER = 1


def roi_layout():
    names, net, idx = [], [], {}
    for nm, n in NETWORKS:
        idx[nm] = []
        for k in range(1, n + 1):
            idx[nm].append(len(names))
            names.append('%s_%02d' % (nm, k))
            net.append(nm)
    return names, net, idx


ROI_NAMES, ROI_NET, IDX = roi_layout()
N_ROI = len(ROI_NAMES)


def fixed_structure():
    rng = np.random.default_rng(SEED)
    mask = rng.uniform(0.25, 1.0, size=(N_ROI, N_ROI))
    np.fill_diagonal(mask, 0.0)
    jitter = rng.uniform(-0.35, 0.35, size=N_ROI)
    return mask, jitter


def subject_specs():
    rng = np.random.default_rng(SEED + 1)
    specs = []
    for k in range(1, N_PER_GROUP + 1):
        for g in GROUPS:
            z = float(rng.standard_normal())
            age = float(np.clip(rng.normal(45.0, 12.0), 18.0, 80.0))
            sex = 'F' if rng.random() < 0.5 else 'M'
            noise = float(rng.standard_normal())
            if g == 'MCS':
                crsr = int(np.clip(round(10.0 + 1.8 * z + 0.4 * noise), 8, 14))
            elif g == 'UWS':
                crsr = int(np.clip(round(5.0 + 1.8 * z + 0.4 * noise), 3, 6))
            else:
                crsr = None
            sim = int(rng.integers(1, 2 ** 31 - 1))
            specs.append({'sid': 'sub-%s%02d' % (g, k), 'group': g, 'z': z, 'age': age,
                          'sex': sex, 'crsr': crsr, 'seed': sim})
    return specs


def subject_matrices(mask, jitter, sp):
    z = 0.0 if sp['group'] == 'CNT' else float(np.clip(sp['z'], -1.8, 1.8))
    mats = []
    for r in range(4):
        pot = np.array([PROFILE[r][n] for n in ROI_NET], float) + jitter
        d = pot[None, :] - pot[:, None]
        env = np.exp(-((d - 1.2) ** 2) / 1.6)
        A = 0.6 * env * mask
        np.fill_diagonal(A, 0.0)
        if z != 0.0:
            lo = float(np.percentile(pot, 40.0))
            hi = float(np.percentile(pot, 60.0))
            s = np.zeros(N_ROI)
            s[pot >= hi] = 1.0
            s[pot <= lo] = -1.0
            g = 1.0 + 0.55 * z * s
            boost = np.ones((N_ROI, N_ROI))
            boost[:, IDX['FRO'] + IDX['THA']] = g[:, None]
            boost[:, IDX['INS']] = (2.0 - g)[:, None]
            A = A * boost
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        mats.append(A * (0.9 / rho))
    return mats


def simulate(mats, sp):
    rng = np.random.default_rng(sp['seed'])
    scale = GROUP_SCALE[sp['group']]
    x = rng.standard_normal(N_ROI)
    X = np.zeros((N_TIME, N_ROI))
    nonhyper = [q for q in range(4) if q != HYPER]
    regime = nonhyper[int(rng.integers(0, len(nonhyper)))]
    t = 0
    while t < N_TIME:
        md = HYPER_DWELL[sp['group']] if regime == HYPER else OTHER_DWELL
        dur = max(1, int(round(rng.exponential(md))))
        for _ in range(dur):
            if t >= N_TIME:
                break
            x = scale * (mats[regime] @ x) + rng.standard_normal(N_ROI)
            X[t] = x
            t += 1
        if regime == HYPER:
            regime = nonhyper[int(rng.integers(0, len(nonhyper)))]
        elif rng.random() < P_HYPER[sp['group']]:
            regime = HYPER
        else:
            others = [q for q in nonhyper if q != regime]
            regime = others[int(rng.integers(0, len(others)))]
    return X


def write_dataset(outdir):
    os.makedirs(os.path.join(outdir, 'bold'), exist_ok=True)
    with open(os.path.join(outdir, 'roi_networks.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['roi', 'network'])
        for nm, net in zip(ROI_NAMES, ROI_NET):
            w.writerow([nm, net])
    specs = subject_specs()
    with open(os.path.join(outdir, 'subjects.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['subject_id', 'group', 'age', 'sex', 'crsr_score'])
        for sp in specs:
            w.writerow([sp['sid'], sp['group'], '%.1f' % sp['age'], sp['sex'],
                        '' if sp['crsr'] is None else sp['crsr']])
    mask, jitter = fixed_structure()
    for sp in specs:
        mats = subject_matrices(mask, jitter, sp)
        X = simulate(mats, sp)
        np.savetxt(os.path.join(outdir, 'bold', sp['sid'] + '.csv'), X, delimiter=',',
                   fmt='%.5f', header=','.join(ROI_NAMES), comments='')
    print('generated %d subjects in %s' % (len(specs), outdir))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)))
    write_dataset(ap.parse_args().out)
