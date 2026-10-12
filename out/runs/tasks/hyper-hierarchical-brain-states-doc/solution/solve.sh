#!/bin/bash
set -euo pipefail
mkdir -p /app/outputs
cat > /app/solve_pipeline.py <<'PYEOF'
import csv
import json
import os

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

DATA = '/app/data'
OUT = '/app/outputs'
TR = 2.0
N_TIME = 300
WINDOW = 60
STEP = 6
LAM = 20.0
EDGE_FRAC = 0.20
K_STATES = 4
FS = 1.0 / TR
GROUPS = ('CNT', 'MCS', 'UWS')

roi_rows = list(csv.DictReader(open(os.path.join(DATA, 'roi_networks.csv'))))
ROI_NAMES = [r['roi'] for r in roi_rows]
pos = {nm: i for i, nm in enumerate(ROI_NAMES)}
IDX = {}
for r in roi_rows:
    IDX.setdefault(r['network'], []).append(pos[r['roi']])
N_ROI = len(ROI_NAMES)



def zscore(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - mu) / sd


def lag1_ridge(X):
    Xl = X[:-1, :]
    Y = X[1:, :]
    G = Xl.T @ Xl + LAM * np.eye(X.shape[1])
    return np.linalg.solve(G, Xl.T @ Y)


def top_edges(B):
    absB = np.abs(B)
    off = ~np.eye(B.shape[0], dtype=bool)
    thr = float(np.percentile(absB[off], 100.0 * (1.0 - EDGE_FRAC)))
    return np.where((absB > thr) & off, absB, 0.0)


def trophic_levels(A):
    b = A.sum(axis=0)
    M = np.diag(b) - A.T
    h = np.linalg.lstsq(M, b, rcond=None)[0]
    h[b <= 0.0] = 1.0
    return h


def trophic_incoherence(A, h):
    ii, jj = np.nonzero(A)
    if ii.size == 0:
        return 0.0
    return float(np.mean((h[jj] - h[ii] - 1.0) ** 2))


def order_param(Xz):
    b, a = butter(4, [0.01 / (FS / 2.0), 0.08 / (FS / 2.0)], btype='band')
    Xf = filtfilt(b, a, Xz, axis=0)
    ph = np.angle(hilbert(Xf, axis=0))
    R = np.abs(np.mean(np.exp(1j * ph), axis=1))
    return float(R.mean()), float(R.std())


subs = []
for r in csv.DictReader(open(os.path.join(DATA, 'subjects.csv'))):
    X = np.loadtxt(os.path.join(DATA, 'bold', r['subject_id'] + '.csv'),
                   delimiter=',', skiprows=1)
    raw = str(r['crsr_score']).strip()
    crsr = None if raw in ('', 'NA', 'nan', 'None') else float(raw)
    subs.append({'sid': r['subject_id'], 'group': r['group'], 'crsr': crsr, 'X': X})

starts = list(range(0, N_TIME - WINDOW + 1, STEP))
per = []
for s in subs:
    Xz = zscore(s['X'])
    A = top_edges(lag1_ridge(Xz))
    h = trophic_levels(A)
    f0 = trophic_incoherence(A, h)
    hw = np.array([trophic_levels(top_edges(lag1_ridge(Xz[w0:w0 + WINDOW]))) for w0 in starts])
    meta, sync = order_param(Xz)
    per.append({'sid': s['sid'], 'group': s['group'], 'crsr': s['crsr'], 'h': h,
                'f0': f0, 'hw': hw, 'meta': meta, 'sync': sync})

n_win = per[0]['hw'].shape[0]
H = np.vstack([p['hw'] for p in per])
mu = H.mean(axis=0)
sd = np.where(H.std(axis=0) == 0.0, 1.0, H.std(axis=0))
Z = (H - mu) / sd
km = KMeans(n_clusters=K_STATES, n_init=10, random_state=0).fit(Z)
lab = km.labels_.reshape(len(per), n_win)
sel = IDX['FRO'] + IDX['THA'] + IDX['INS']
oth = [q for q in range(N_ROI) if q not in set(sel)]
c = km.cluster_centers_
hyper = int(np.argmax(c[:, sel].mean(axis=1) - c[:, oth].mean(axis=1)))

for p, L in zip(per, lab):
    p['lab'] = L
    p['occ'] = float(np.mean(L == hyper))
    runs = []
    cur = 0
    for v in L:
        if v == hyper:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    p['n_entries'] = int(len(runs))
    p['dwell'] = float(np.mean(runs) * STEP * TR) if runs else 0.0
    m = (L == hyper)
    p['elev'] = (p['hw'][m].mean(axis=0) - p['hw'].mean(axis=0)) if m.any() else np.zeros(N_ROI)

res = {'n_subjects': len(per), 'n_windows': int(len(per) * n_win),
       'window_params': {'window_length_tr': WINDOW, 'step_tr': STEP,
                         'n_windows_per_subject': int(n_win)},
       'n_states': K_STATES, 'hyper_state_cluster_id': hyper,
       'static_trophic': {}, 'dynamic_states': {}, 'state_occupancy_all_clusters': {},
       'metastability': {}, 'synchrony': {}}
for g in GROUPS:
    grp = [p for p in per if p['group'] == g]
    res['static_trophic'][g] = {
        'mean_f0': float(np.mean([p['f0'] for p in grp])),
        'mean_trophic_level': float(np.mean([p['h'].mean() for p in grp])),
        'sd_trophic_level': float(np.mean([p['h'].std() for p in grp]))}
    res['dynamic_states'][g] = {
        'occupancy': float(np.mean([p['occ'] for p in grp])),
        'dwell_time_s': float(np.mean([p['dwell'] for p in grp])),
        'n_entries': float(np.mean([p['n_entries'] for p in grp]))}
    res['state_occupancy_all_clusters'][g] = [
        float(np.mean([np.mean(p['lab'] == k) for p in grp])) for k in range(K_STATES)]
    res['metastability'][g] = float(np.mean([p['meta'] for p in grp]))
    res['synchrony'][g] = float(np.mean([p['sync'] for p in grp]))

pat = [p for p in per if p['crsr'] is not None]
y = np.array([p['crsr'] for p in pat], float)
assoc = {}
for net in ('FRO', 'THA', 'INS'):
    x = np.array([p['h'][IDX[net]].mean() for p in pat])
    rho, pv = spearmanr(x, y)
    assoc[net] = {'rho': float(rho), 'p': float(pv), 'n': int(len(pat))}
x = np.array([p['occ'] for p in pat])
rho, pv = spearmanr(x, y)
assoc['hyper_occupancy'] = {'rho': float(rho), 'p': float(pv), 'n': int(len(pat))}
res['crsr_association'] = assoc

E = np.mean([p['elev'] for p in per], axis=0)
res['hyper_state_top_rois'] = [ROI_NAMES[q] for q in np.argsort(-E)[:20]]

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'results.json'), 'w') as fh:
    json.dump(res, fh, indent=2)

cols = ['subject_id', 'group', 'crsr_score', 'mean_trophic_level', 'f0', 'hyper_occupancy',
        'hyper_dwell_time_s', 'hyper_n_entries', 'metastability', 'synchrony']
with open(os.path.join(OUT, 'subject_states.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for p in per:
        w.writerow([p['sid'], p['group'], '' if p['crsr'] is None else int(p['crsr']),
                    '%.6f' % p['h'].mean(), '%.6f' % p['f0'], '%.6f' % p['occ'],
                    '%.6f' % p['dwell'], p['n_entries'], '%.6f' % p['meta'], '%.6f' % p['sync']])
print('wrote results.json and subject_states.csv')
PYEOF
python /app/solve_pipeline.py
