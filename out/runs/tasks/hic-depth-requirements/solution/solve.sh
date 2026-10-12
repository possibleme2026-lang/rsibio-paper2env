#!/bin/bash
set -euo pipefail
mkdir -p /app/output /app/solution
cat > /app/solution/analyze.py <<'PYEOF'
import json
import os

import numpy as np
import scipy.sparse as sp

DATA = '/app/data'
OUT = '/app/output'
DEPTHS = [1, 2, 5, 10, 20, 30, 60, 100]
REPS = [1, 2, 3, 4]
NSUB = 3
METRICS = ['loop_density', 'loop_size_kb', 'insulation_strength',
           'boundary_count', 'compartment_strength']
RES_OF = {'loop_density': 'res100kb', 'loop_size_kb': 'res100kb',
          'compartment_strength': 'res100kb',
          'insulation_strength': 'res25kb', 'boundary_count': 'res25kb'}
W_INS = 8
MED_HALF = 40
BND_CUT = -0.25
BND_SEP = 20


def load(res, rep, depth, sub):
    path = os.path.join(DATA, 'matrices', res, 'rep%d' % rep,
                        'depth_%dM_sub%d.npz' % (depth, sub))
    return np.asarray(sp.load_npz(path).todense(), dtype=np.float64)


def pcurve(M):
    n = M.shape[0]
    P = np.array([M.diagonal(k).mean() for k in range(1, n)])
    Ps = np.empty_like(P)
    for i in range(len(P)):
        lo = max(0, i - 2)
        hi = min(len(P), i + 3)
        Ps[i] = P[lo:hi].mean()
    return Ps


def loop_metrics(M):
    n = M.shape[0]
    Ps = pcurve(M)
    k = np.arange(1, n, dtype=float)
    d = (np.log(Ps[1:]) - np.log(Ps[:-1])) / (np.log(k[1:]) - np.log(k[:-1]))
    alpha = d[89:149].mean()
    sep = np.arange(2, 41)
    r = alpha - d[sep - 1]
    j = int(np.argmax(r))
    return float(r[j]), float(100 * sep[j])


def insulation(M):
    n = M.shape[0]
    C = np.zeros((n + 1, n + 1))
    C[1:, 1:] = M.cumsum(0).cumsum(1)
    centres = np.arange(W_INS - 1, n - W_INS)
    raw = np.empty(len(centres))
    for idx, i in enumerate(centres):
        r0 = i - W_INS + 1
        c1 = i + W_INS + 1
        raw[idx] = C[i + 1, c1] - C[r0, c1] - C[i + 1, i + 1] + C[r0, i + 1]
    bg = np.empty_like(raw)
    for idx in range(len(raw)):
        lo = max(0, idx - MED_HALF)
        hi = min(len(raw), idx + MED_HALF + 1)
        bg[idx] = np.median(raw[lo:hi])
    z = np.log2((raw + 1.0) / (bg + 1.0))
    zs = np.empty_like(z)
    for idx in range(1, len(z) - 1):
        zs[idx] = (z[idx - 1] + z[idx] + z[idx + 1]) / 3.0
    cand = []
    for idx in range(1, len(z) - 1):
        if zs[idx] < zs[idx - 1] and zs[idx] < zs[idx + 1] and zs[idx] < BND_CUT:
            cand.append(idx)
    cand.sort(key=lambda j: zs[j])
    acc = []
    for idx in cand:
        if all(abs(int(centres[idx]) - int(centres[j])) >= BND_SEP for j in acc):
            acc.append(idx)
    bins = sorted(int(centres[j]) for j in acc)
    strength = float(np.mean([-zs[j] for j in acc])) if acc else 0.0
    return strength, len(acc), bins


def pc1_vector(M):
    n = M.shape[0]
    E = np.array([M.diagonal(k).mean() for k in range(1, n)])
    OE = np.zeros_like(M)
    for k in range(1, n):
        v = M.diagonal(k) / E[k - 1]
        OE[np.arange(n - k), np.arange(k, n)] = v
        OE[np.arange(k, n), np.arange(n - k)] = v
    X = OE - OE.mean(axis=1, keepdims=True)
    sd = X.std(axis=1)
    good = sd > 1e-12
    C = np.zeros((n, n))
    if good.sum() > 1:
        Xg = X[good] / sd[good][:, None]
        C[np.ix_(good, good)] = Xg @ Xg.T / n
    for i in range(n):
        if not good[i]:
            C[i, i] = 1.0
    w, V = np.linalg.eigh(C)
    v = V[:, -1].copy()
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    return v


def compartment_strength(v):
    pos = v[v > 0]
    neg = v[v < 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(pos.mean() - neg.mean())


raw = dict((m, dict((r, dict((d, []) for d in DEPTHS)) for r in REPS))
           for m in METRICS)
for res in ('res100kb', 'res25kb'):
    for rep in REPS:
        for d in DEPTHS:
            for s in range(1, NSUB + 1):
                M = load(res, rep, d, s)
                if res == 'res100kb':
                    ld, ls = loop_metrics(M)
                    v = pc1_vector(M)
                    vals = {'loop_density': ld, 'loop_size_kb': ls,
                            'compartment_strength': compartment_strength(v)}
                    names = ['loop_density', 'loop_size_kb', 'compartment_strength']
                else:
                    st, cnt, bins = insulation(M)
                    vals = {'insulation_strength': st, 'boundary_count': float(cnt)}
                    names = ['insulation_strength', 'boundary_count']
                for name in names:
                    raw[name][rep][d].append(vals[name])

values = {}
scatter = {}
bio = {}
threshold = {}
for name in METRICS:
    values[name] = {}
    per_depth = dict((d, []) for d in DEPTHS)
    for rep in REPS:
        values[name][rep] = {}
        for d in DEPTHS:
            vv = raw[name][rep][d]
            values[name][rep][d] = float(np.mean(vv))
            per_depth[d].append(float(np.std(vv, ddof=1)))
    scatter[name] = dict(
        (d, float(np.sqrt(np.mean(np.square(per_depth[d]))))) for d in DEPTHS)
    deep = [values[name][rep][DEPTHS[-1]] for rep in REPS]
    bio[name] = float(np.std(deep, ddof=1))
    thr = DEPTHS[-1]
    for d in DEPTHS:
        if scatter[name][d] <= bio[name]:
            thr = d
            break
    threshold[name] = thr

out = {'depths_M': DEPTHS, 'metrics': {}}
for name in METRICS:
    out['metrics'][name] = {
        'values_by_replicate': dict(
            (str(rep), dict((str(d), values[name][rep][d]) for d in DEPTHS))
            for rep in REPS),
        'subsampling_scatter': dict((str(d), scatter[name][d]) for d in DEPTHS),
        'biological_variability': bio[name],
        'threshold_M': threshold[name]}

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'depth_requirements.json'), 'w') as fh:
    json.dump(out, fh, indent=2)

sub1_25 = load('res25kb', 1, 100, 1)
bins = insulation(sub1_25)[2]
with open(os.path.join(OUT, 'boundaries_rep1_depth100M.tsv'), 'w') as fh:
    for b in bins:
        fh.write('%d\n' % b)

pc1_1 = pc1_vector(load('res100kb', 1, 100, 1))
with open(os.path.join(OUT, 'pc1_rep1_depth100M.tsv'), 'w') as fh:
    for i, val in enumerate(pc1_1):
        fh.write('%d\t%.8e\n' % (i, val))

print('thresholds:', threshold)
PYEOF
python3 /app/solution/analyze.py
ls -l /app/output
