#!/usr/bin/env python3
'''Protected generator for the held-out HpaBC assay values.

The dataset core below is byte-identical to environment/data/gen_inputs.py (public seed),
so the verifier never trusts anything inside the container. The held-out assay values
themselves depend on HIDDEN keys that exist only in this file.
'''

from __future__ import annotations

import hashlib
import math
import random
from statistics import NormalDist

SEED = 20260731
L = 606
AA = 'ACDEFGHIKLMNPQRSTVWY'
COMP = {
    'A': 8.3, 'C': 1.4, 'D': 5.3, 'E': 6.2, 'F': 3.9, 'G': 7.1, 'H': 2.3,
    'I': 5.9, 'K': 5.8, 'L': 9.7, 'M': 2.4, 'N': 4.1, 'P': 4.7, 'Q': 3.9,
    'R': 5.5, 'S': 6.6, 'T': 5.4, 'V': 6.9, 'W': 1.1, 'Y': 2.9,
}
CONSERVED_POS = (101, 104, 106, 109, 205, 397)
CONSERVED = {101: 'G', 104: 'G', 106: 'G', 109: 'T', 205: 'R', 397: 'Y'}
PUBLIC_EPI_KEY = 'hpaBC-de-library-epistasis-v1'
PUBLIC_EPI_MASK = 0.30
PUBLIC_EPI_SD = 0.22

HIDDEN_EPI_KEY = 'hpaBC-holdout-epistasis-7f3a91c2e5'
HIDDEN_EPI_MASK = 0.25
HIDDEN_EPI_SD = 0.22
HIDDEN_RESID_KEY = 'hpaBC-holdout-unseen-4b81e0d7'
HIDDEN_RESID_SD = 0.15
HIDDEN_ASSAY_KEY = 'hpaBC-holdout-assay-c50d24a9'
HIDDEN_ASSAY_SD = 0.07
ASSAY_CAP = 8.0


def keyed_uniform(salt, *args):
    payload = '|'.join([salt] + [str(a) for a in args]).encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return (int.from_bytes(digest[:8], 'big') + 0.5) / float(1 << 64)


def keyed_normal(salt, *args, sd=1.0):
    return sd * NormalDist().inv_cdf(keyed_uniform(salt, *args))


def epistasis(salt, mask_rate, sd, p, q):
    lo, hi = (p, q) if p <= q else (q, p)
    if lo == hi:
        return 0.0
    if keyed_uniform(salt + '-mask', lo, hi) >= mask_rate:
        return 0.0
    return keyed_normal(salt + '-value', lo, hi, sd=sd)


def make_sequence(rng):
    letters = list(COMP)
    weights = [COMP[a] for a in letters]
    seq = rng.choices(letters, weights=weights, k=L)
    for pos, aa in CONSERVED.items():
        seq[pos - 1] = aa
    return ''.join(seq)


def pick_mutation(rng, pos, by_pos, seq, covered_prob=0.9):
    if pos in by_pos and rng.random() < covered_prob:
        return (pos, rng.choice(by_pos[pos]))
    return (pos, rng.choice([a for a in AA if a != seq[pos - 1]]))


def build_dataset():
    rng = random.Random(SEED)
    seq = make_sequence(rng)

    r_u = random.Random(SEED + 1)
    r_v = random.Random(SEED + 2)
    r_w = random.Random(SEED + 3)
    u, v, w = {}, {}, {}
    for pos in range(1, L + 1):
        wt = seq[pos - 1]
        for aa in AA:
            if aa == wt:
                continue
            u[(pos, aa)] = r_u.gauss(0.0, 1.0)
            v[(pos, aa)] = r_v.gauss(0.0, 1.0)
            w[(pos, aa)] = r_w.gauss(0.0, 1.0)
    zeroshot = {k: 0.177 * (u[k] + w[k]) for k in u}
    latent = {k: 0.177 * (u[k] + v[k]) for k in u}
    for pos in CONSERVED:
        for aa in AA:
            if aa != seq[pos - 1]:
                zeroshot[(pos, aa)] = -1.2 + 0.15 * w[(pos, aa)]

    r_lib = random.Random(SEED + 4)
    free = [p for p in range(1, L + 1) if p not in CONSERVED]
    lib_pos = r_lib.sample(free, 184)
    pairs = [(p, r_lib.choice([a for a in AA if a != seq[p - 1]])) for p in lib_pos]
    for p in r_lib.sample(lib_pos, 30):
        taken = {a for (q, a) in pairs if q == p}
        pairs.append((p, r_lib.choice([a for a in AA if a != seq[p - 1] and a not in taken])))
    for p, a in ((101, 'S'), (104, 'A'), (106, 'V'), (109, 'A'), (205, 'K'), (397, 'F')):
        pairs.append((p, a))
    r_lib.shuffle(pairs)

    r_meas = random.Random(SEED + 5)
    library = []
    activity = {}
    for i, (p, a) in enumerate(pairs):
        if p in CONSERVED:
            effect = -2.9 + 0.15 * r_meas.gauss(0.0, 1.0)
        else:
            effect = latent[(p, a)]
        act = math.exp(effect + 0.10 * r_meas.gauss(0.0, 1.0))
        act = round(min(max(act, 0.01), 8.0), 4)
        activity[(p, a)] = act
        library.append({
            'variant_id': 'SM%03d' % (i + 1),
            'mutation': '%s%d%s' % (seq[p - 1], p, a),
            'position': p, 'wt_aa': seq[p - 1], 'mut_aa': a,
            'relative_activity': act,
            'assay_batch': 'B%d' % (1 + i % 4),
        })
    by_pos = {}
    for (p, a) in activity:
        by_pos.setdefault(p, []).append(a)
    for p in by_pos:
        by_pos[p].sort()

    r_de = random.Random(SEED + 6)
    lib_positions = sorted(by_pos)
    de = []
    seen = set()
    while len(de) < 56:
        n = r_de.choice((2, 2, 2, 3, 3, 3, 4))
        muts = [(p, r_de.choice(by_pos[p])) for p in r_de.sample(lib_positions, n)]
        key = tuple(sorted(muts))
        if key in seen:
            continue
        seen.add(key)
        log_act = sum(math.log(activity[m]) for m in muts)
        for i in range(n):
            for j in range(i + 1, n):
                log_act += epistasis(PUBLIC_EPI_KEY, PUBLIC_EPI_MASK, PUBLIC_EPI_SD,
                                     muts[i][0], muts[j][0])
        act = math.exp(log_act + 0.08 * r_de.gauss(0.0, 1.0))
        act = round(min(max(act, 0.01), 8.0), 4)
        de.append({
            'variant_id': 'DE%02d' % (len(de) + 1),
            'mutations': ';'.join('%s%d%s' % (seq[p - 1], p, a) for p, a in muts),
            'n_mutations': n,
            'relative_activity': act,
        })

    r_test = random.Random(SEED + 7)
    novel_pool = [p for p in free if p not in by_pos]
    design_pool = lib_positions + r_test.sample(novel_pool, 60)
    strong = sorted(activity, key=lambda k: (-activity[k], k))[:60]
    designs = []
    for i in range(12):
        n = 4 if i % 2 == 0 else 5
        muts, used = [], set()
        while len(muts) < n:
            pos, aa = r_test.choice(strong)
            if pos in used:
                continue
            used.add(pos)
            muts.append((pos, aa))
        designs.append(muts)
    for i in range(10):
        pos_c = CONSERVED_POS[i % len(CONSERVED_POS)]
        aa_c = [a for a in AA if a != seq[pos_c - 1]][(i * 7) % 19]
        muts, used = [(pos_c, aa_c)], {pos_c}
        target = r_test.choice((3, 4, 5))
        while len(muts) < target:
            pos = r_test.choice(design_pool)
            if pos in used:
                continue
            used.add(pos)
            muts.append(pick_mutation(r_test, pos, by_pos, seq))
        designs.append(muts)
    while len(designs) < 64:
        target = r_test.choice((2, 3, 3, 4, 4, 5))
        muts, used = [], set()
        while len(muts) < target:
            pos = r_test.choice(design_pool)
            if pos in used:
                continue
            used.add(pos)
            muts.append(pick_mutation(r_test, pos, by_pos, seq))
        designs.append(muts)
    tests = [{
        'variant_id': 'TV%03d' % (i + 1),
        'mutations': ';'.join('%s%d%s' % (seq[p - 1], p, a) for p, a in muts),
        'n_mutations': len(muts),
    } for i, muts in enumerate(designs)]

    return {'seq': seq, 'zeroshot': zeroshot, 'library': library,
            'de': de, 'test': tests, 'activity': activity}


def slope_from_library(ds):
    xs = [ds['zeroshot'][(r['position'], r['mut_aa'])] for r in ds['library']]
    ys = [math.log(float(r['relative_activity'])) for r in ds['library']]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


def generate_gold(ds=None):
    '''Held-out assay values for the 64 designs (never visible to the agent).'''
    ds = ds if ds is not None else build_dataset()
    seq = ds['seq']
    zeroshot = ds['zeroshot']
    activity = ds['activity']
    slope = slope_from_library(ds)
    gold = {}
    for row in ds['test']:
        muts = []
        for token in row['mutations'].split(';'):
            token = token.strip()
            if not token:
                continue
            wt, pos, mut = token[0], int(token[1:-1]), token[-1]
            assert seq[pos - 1] == wt
            muts.append((pos, mut))
        log_act = 0.0
        for pos, aa in muts:
            if (pos, aa) in activity:
                log_act += math.log(activity[(pos, aa)])
            else:
                log_act += slope * zeroshot[(pos, aa)] + keyed_normal(
                    HIDDEN_RESID_KEY, pos, aa, sd=HIDDEN_RESID_SD)
        for i in range(len(muts)):
            for j in range(i + 1, len(muts)):
                log_act += epistasis(HIDDEN_EPI_KEY, HIDDEN_EPI_MASK, HIDDEN_EPI_SD,
                                     muts[i][0], muts[j][0])
        act = math.exp(log_act) * math.exp(
            keyed_normal(HIDDEN_ASSAY_KEY, row['variant_id'], sd=HIDDEN_ASSAY_SD))
        if any(pos in CONSERVED for pos, _ in muts):
            act = min(act, 0.12)
        gold[row['variant_id']] = round(min(max(act, 0.01), ASSAY_CAP), 4)
    return gold


if __name__ == '__main__':
    import json
    print(json.dumps(generate_gold(), indent=2, sort_keys=True))
