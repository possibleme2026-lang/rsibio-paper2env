#!/usr/bin/env python3
'''Verifier for the Hi-C sequencing-depth requirement task.

Nothing the agent reports is trusted: every metric is recomputed from the raw
matrices in /app/data and compared with the agent's output under explicit
tolerances.  Score = weighted fraction of checks passed (partial credit).
'''
import json
import os
import sys
import traceback

import numpy as np
import scipy.sparse as sp

DATA = os.environ.get('HIC_DATA', '/app/data')
OUTPUT = os.environ.get('HIC_OUTPUT', '/app/output')
REWARD_PATH = os.environ.get('HIC_REWARD', '/logs/verifier/reward.json')

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
BND_TOL = 2


def matrix_path(res, rep, depth, sub):
    return os.path.join(DATA, 'matrices', res, 'rep%d' % rep,
                        'depth_%dM_sub%d.npz' % (depth, sub))


def load_matrix(res, rep, depth, sub):
    return np.asarray(sp.load_npz(matrix_path(res, rep, depth, sub)).todense(),
                      dtype=np.float64)


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
    return strength, float(len(acc)), bins


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


_SAMPLE = {}


def sample_metrics(res, rep, depth, sub):
    key = (res, rep, depth, sub)
    if key in _SAMPLE:
        return _SAMPLE[key]
    M = load_matrix(res, rep, depth, sub)
    if res == 'res100kb':
        ld, ls = loop_metrics(M)
        v = pc1_vector(M)
        out = {'loop_density': ld, 'loop_size_kb': ls,
               'compartment_strength': compartment_strength(v)}
    else:
        strength, count, bins = insulation(M)
        out = {'insulation_strength': strength, 'boundary_count': count,
               'bins': bins}
    _SAMPLE[key] = out
    return out


def build_reference():
    ref = {'values': {}, 'scatter': {}, 'bio': {}, 'threshold': {}}
    for name in METRICS:
        res = RES_OF[name]
        ref['values'][name] = {}
        per_depth = dict((d, []) for d in DEPTHS)
        for rep in REPS:
            ref['values'][name][rep] = {}
            for d in DEPTHS:
                vals = [sample_metrics(res, rep, d, s)[name]
                        for s in range(1, NSUB + 1)]
                ref['values'][name][rep][d] = float(np.mean(vals))
                per_depth[d].append(float(np.std(vals, ddof=1)))
        ref['scatter'][name] = dict(
            (d, float(np.sqrt(np.mean(np.square(per_depth[d]))))) for d in DEPTHS)
        deep = [ref['values'][name][rep][DEPTHS[-1]] for rep in REPS]
        ref['bio'][name] = float(np.std(deep, ddof=1))
        thr = None
        for d in DEPTHS:
            if ref['scatter'][name][d] <= ref['bio'][name]:
                thr = d
                break
        ref['threshold'][name] = DEPTHS[-1] if thr is None else thr
    return ref


def reference_bins():
    return insulation(load_matrix('res25kb', 1, 100, 1))[2]


def reference_pc1():
    return pc1_vector(load_matrix('res100kb', 1, 100, 1))


def keyed(d, key):
    if not isinstance(d, dict):
        return None
    candidates = [str(key), key]
    try:
        candidates.insert(1, str(float(key)))
    except (TypeError, ValueError):
        pass  # key is a metric name, not a number
    for k in candidates:
        if k in d:
            return d[k]
    return None


def as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def tol_close(a, b, rel=0.10, floor=0.02):
    a = as_float(a)
    if a is None:
        return False
    return abs(a - b) <= max(rel * abs(b), floor)


def check_schema(agent):
    if not isinstance(agent, dict):
        return False, 'output json missing or not an object'
    if agent.get('depths_M') != DEPTHS:
        return False, 'depths_M must be %s' % DEPTHS
    metrics = agent.get('metrics')
    for name in METRICS:
        m = keyed(metrics, name)
        if m is None:
            return False, 'metric %s missing' % name
        for rep in REPS:
            row = keyed(m.get('values_by_replicate'), rep)
            if row is None:
                return False, 'values_by_replicate[%d] missing' % rep
            for d in DEPTHS:
                if as_float(keyed(row, d)) is None:
                    return False, 'values[%s][%d][%d] missing' % (name, rep, d)
        for d in DEPTHS:
            if as_float(keyed(m.get('subsampling_scatter'), d)) is None:
                return False, '%s scatter[%d] missing' % (name, d)
        if as_float(m.get('biological_variability')) is None:
            return False, '%s biological_variability missing' % name
        thr = as_float(m.get('threshold_M'))
        if thr is None or thr not in [float(d) for d in DEPTHS]:
            return False, '%s threshold_M not on the grid' % name
    for p in ('boundaries_rep1_depth100M.tsv', 'pc1_rep1_depth100M.tsv'):
        if not os.path.isfile(os.path.join(OUTPUT, p)):
            return False, 'missing %s' % p
    return True, 'ok'


def check_curve(agent, ref, name, rep, rel=0.10, floor=0.02):
    m = keyed(agent.get('metrics'), name)
    row = keyed((m or {}).get('values_by_replicate'), rep)
    bad = []
    for d in DEPTHS:
        a = as_float(keyed(row, d))
        r = ref['values'][name][rep][d]
        if not tol_close(a, r, rel, floor):
            bad.append((d, a, round(r, 5)))
    return len(bad) == 0, 'mismatches %s' % bad[:3]


def check_loop_size(agent, ref):
    m = keyed(agent.get('metrics'), 'loop_size_kb')
    bad = []
    for rep in REPS:
        row = keyed((m or {}).get('values_by_replicate'), rep)
        for d in DEPTHS:
            a = as_float(keyed(row, d))
            r = ref['values']['loop_size_kb'][rep][d]
            if a is None or abs(a - r) > 100.0:
                bad.append((rep, d, a, r))
    return len(bad) == 0, 'mismatches %s' % bad[:3]


def check_scatter(agent, ref):
    ok = 0
    tot = 0
    worst = []
    for name in METRICS:
        m = keyed(agent.get('metrics'), name)
        sc = (m or {}).get('subsampling_scatter')
        for d in DEPTHS:
            tot += 1
            a = as_float(keyed(sc, d))
            r = ref['scatter'][name][d]
            if a is not None and abs(a - r) <= max(0.25 * abs(r), 0.005):
                ok += 1
            else:
                worst.append((name, d, a, round(r, 5)))
    return ok >= int(0.9 * tot), '%d/%d within tolerance, e.g. %s' % (ok, tot, worst[:2])


def check_bio(agent, ref):
    bad = []
    for name in METRICS:
        m = keyed(agent.get('metrics'), name)
        a = as_float((m or {}).get('biological_variability'))
        r = ref['bio'][name]
        scale = float(np.mean([abs(ref['values'][name][rep][100]) for rep in REPS]))
        if a is None or abs(a - r) > max(0.15 * abs(r), 0.05 * scale):
            bad.append((name, a, round(r, 5)))
    return len(bad) == 0, 'bad %s' % bad


def match_f1(agent_bins, ref_bins, tol=BND_TOL):
    used = set()
    tp = 0
    for a in sorted(agent_bins):
        for j, r in enumerate(ref_bins):
            if j in used:
                continue
            if abs(a - r) <= tol:
                used.add(j)
                tp += 1
                break
    if tp == 0:
        return 0.0
    p = tp / float(max(1, len(agent_bins)))
    rc = tp / float(max(1, len(ref_bins)))
    return 2 * p * rc / (p + rc)


def check_boundaries():
    path = os.path.join(OUTPUT, 'boundaries_rep1_depth100M.tsv')
    try:
        with open(path) as fh:
            bins = [int(l.split()[0]) for l in fh if l.strip()]
    except Exception as exc:
        return False, 'cannot read boundaries: %s' % exc
    ref_bins = reference_bins()
    f1 = match_f1(bins, ref_bins)
    return f1 >= 0.8, 'F1=%.3f (agent %d, ref %d)' % (f1, len(bins), len(ref_bins))


def check_pc1():
    path = os.path.join(OUTPUT, 'pc1_rep1_depth100M.tsv')
    try:
        vals = []
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    vals.append(float(line.split()[-1]))
    except Exception as exc:
        return False, 'cannot read pc1: %s' % exc
    ref = reference_pc1()
    if len(vals) != len(ref):
        return False, 'expected %d values, got %d' % (len(ref), len(vals))
    a = np.array(vals)
    if np.std(a) < 1e-12 or np.std(ref) < 1e-12:
        return False, 'degenerate vector'
    r = abs(float(np.corrcoef(a, ref)[0, 1]))
    return r >= 0.9, '|r|=%.3f' % r


def check_threshold(agent, ref, name):
    m = keyed(agent.get('metrics'), name)
    a = as_float((m or {}).get('threshold_M'))
    r = ref['threshold'][name]
    if a is None:
        return False, 'missing'
    ia = int(min(range(len(DEPTHS)), key=lambda i: abs(DEPTHS[i] - a)))
    ir = DEPTHS.index(r)
    return abs(ia - ir) <= 1, 'agent=%s ref=%s' % (a, r)


def check_ordering(agent, ref):
    pairs = [('loop_density', 'loop_size_kb'),
             ('loop_size_kb', 'insulation_strength'),
             ('loop_size_kb', 'boundary_count'),
             ('insulation_strength', 'compartment_strength'),
             ('boundary_count', 'compartment_strength')]
    ta = {}
    for name in METRICS:
        m = keyed(agent.get('metrics'), name)
        ta[name] = as_float((m or {}).get('threshold_M'))
    if any(v is None for v in ta.values()):
        return False, 'missing thresholds'
    bad = []
    for a, b in pairs:
        if ref['threshold'][a] < ref['threshold'][b] and ta[a] > ta[b]:
            bad.append('%s(%s) <= %s(%s)' % (a, ta[a], b, ta[b]))
    if bad:
        return False, 'violations %s' % bad
    return True, 'consistent with reference ordering'


def main():
    results = []

    def add(name, weight, ok, info):
        results.append((name, float(weight), bool(ok), info))
        print('[%s] %-30s w=%.2f %s' % ('PASS' if ok else 'FAIL', name, weight, info))

    try:
        with open(os.path.join(OUTPUT, 'depth_requirements.json')) as fh:
            agent = json.load(fh)
    except Exception as exc:
        print('could not read agent output: %s' % exc)
        agent = None

    try:
        ref = build_reference()
        print('reference thresholds    : %s' % dict((k, ref['threshold'][k]) for k in METRICS))
        print('reference bio variability: %s' % dict((k, round(ref['bio'][k], 5)) for k in METRICS))
        for name in METRICS:
            print('scatter %-22s %s' % (name, dict((d, round(ref['scatter'][name][d], 5)) for d in DEPTHS)))
    except Exception:
        traceback.print_exc()
        ref = None

    if ref is None:
        add('schema', 0.05, False, 'reference computation failed')
    else:
        add('schema', 0.05, *check_schema(agent))
        safe = agent if isinstance(agent, dict) else {}
        add('loop_density_curve', 0.08,
            *check_curve(safe, ref, 'loop_density', 1))
        add('loop_size_values', 0.08, *check_loop_size(safe, ref))
        add('insulation_curve', 0.08,
            *check_curve(safe, ref, 'insulation_strength', 2, rel=0.10, floor=0.02))
        add('boundary_f1', 0.10, *check_boundaries())
        add('pc1_correlation', 0.10, *check_pc1())
        add('scatter_curves', 0.06, *check_scatter(safe, ref))
        add('bio_variability', 0.10, *check_bio(safe, ref))
        for name in METRICS:
            add('threshold_%s' % name, 0.05, *check_threshold(safe, ref, name))
        add('threshold_ordering', 0.10, *check_ordering(safe, ref))

    score = sum(w for (_, w, ok, _) in results if ok)
    score = float(min(1.0, max(0.0, round(score, 4))))
    payload = {'score': score, 'reward': score,
               'checks': [{'name': n, 'weight': w, 'passed': ok, 'info': i}
                          for (n, w, ok, i) in results]}
    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(REWARD_PATH, 'w') as fh:
        json.dump(payload, fh)
    print('FINAL score=%.4f' % score)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        os.makedirs('/logs/verifier', exist_ok=True)
        with open('/logs/verifier/reward.json', 'w') as fh:
            json.dump({'score': 0.0, 'reward': 0.0}, fh)
        sys.exit(0)
