#!/usr/bin/env python3
"""Reference cohort generator + reference trophic pipeline (gold standard)."""
import csv
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

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

WINDOW = 60
STEP = 6
LAM = 20.0
EDGE_FRAC = 0.20
K_STATES = 4
FS = 1.0 / TR


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


def generate_dataset():
    mask, jitter = fixed_structure()
    subs = []
    for sp in subject_specs():
        mats = subject_matrices(mask, jitter, sp)
        rec = dict(sp)
        rec['X'] = simulate(mats, sp)
        subs.append(rec)
    return subs


def zscore(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - mu) / sd


def lag1_ridge(X, lam=LAM):
    Xl = X[:-1, :]
    Y = X[1:, :]
    G = Xl.T @ Xl + lam * np.eye(X.shape[1])
    return np.linalg.solve(G, Xl.T @ Y)


def top_edges(B, frac=EDGE_FRAC):
    n = B.shape[0]
    absB = np.abs(B)
    off = ~np.eye(n, dtype=bool)
    thr = float(np.percentile(absB[off], 100.0 * (1.0 - frac)))
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


def analyse(subs):
    per = []
    starts = list(range(0, N_TIME - WINDOW + 1, STEP))
    for s in subs:
        Xz = zscore(s['X'])
        A = top_edges(lag1_ridge(Xz))
        h = trophic_levels(A)
        f0 = trophic_incoherence(A, h)
        hw = np.array([trophic_levels(top_edges(lag1_ridge(Xz[w0:w0 + WINDOW])))
                       for w0 in starts])
        meta, sync = order_param(Xz)
        per.append({'sid': s['sid'], 'group': s['group'], 'crsr': s['crsr'], 'h': h,
                    'f0': f0, 'hw': hw, 'meta': meta, 'sync': sync})
    n_win = per[0]['hw'].shape[0]
    H = np.vstack([p['hw'] for p in per])
    mu = H.mean(axis=0)
    sd = H.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    Z = (H - mu) / sd
    km = KMeans(n_clusters=K_STATES, n_init=10, random_state=0).fit(Z)
    lab = km.labels_.reshape(len(per), n_win)
    sel = IDX['FRO'] + IDX['THA'] + IDX['INS']
    oth = [q for q in range(N_ROI) if q not in set(sel)]
    c = km.cluster_centers_
    sc = c[:, sel].mean(axis=1) - c[:, oth].mean(axis=1)
    hyper = int(np.argmax(sc))
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
    order = np.argsort(-E)
    res['hyper_state_top_rois'] = [ROI_NAMES[q] for q in order[:20]]
    rows = []
    for p in per:
        rows.append({'subject_id': p['sid'], 'group': p['group'], 'crsr_score': p['crsr'],
                     'mean_trophic_level': float(p['h'].mean()), 'f0': float(p['f0']),
                     'hyper_occupancy': float(p['occ']), 'hyper_dwell_time_s': float(p['dwell']),
                     'hyper_n_entries': int(p['n_entries']), 'metastability': float(p['meta']),
                     'synchrony': float(p['sync'])})
    return res, rows


if __name__ == '__main__':
    r, rr = analyse(generate_dataset())
    import json
    print(json.dumps(r, indent=2)[:2000])
