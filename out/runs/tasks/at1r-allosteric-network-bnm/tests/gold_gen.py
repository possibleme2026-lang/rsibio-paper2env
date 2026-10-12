#!/usr/bin/env python3
# Protected generator for the AT1R allosteric-network benchmark.
# Identical construction logic to environment/data/gen_at1r_data.py.
import argparse
import os

import numpy as np
import pandas as pd

N_RES = 359
N_FRAMES = 1500
PUBLIC_SEED = 20260801

REGIONS = [
    ('N-term', 1, 25), ('TM1', 26, 50), ('ICL1', 51, 55), ('TM2', 56, 81),
    ('ECL1', 82, 96), ('TM3', 97, 122), ('ICL2', 123, 139), ('TM4', 140, 163),
    ('ECL2', 164, 192), ('TM5', 193, 217), ('ICL3', 218, 243), ('TM6', 244, 268),
    ('ECL3', 269, 279), ('TM7', 280, 305), ('H8', 306, 317), ('C-term', 318, 359),
]
HELIX_REF = {1: 39, 2: 70, 3: 113, 4: 151, 5: 204, 6: 256, 7: 293}
ANGII_SITE = [23, 34, 35, 39, 108, 111, 112, 114, 163, 167, 256, 259]
GPROT_IFACE = [125, 127, 128, 130, 131, 132, 133, 136, 218, 221, 225, 229,
               306, 309, 312, 313, 315, 318]
POCKET = [44, 47, 51, 54, 60, 62, 130, 134, 137, 218, 222, 226, 290, 293,
          306, 310, 314]
AA = 'ARNDCQEGHILKMFPSTWYV'


def annotation_table():
    rows = []
    for i in range(N_RES):
        num = i + 1
        region = 'C-term'
        for name, a, b in REGIONS:
            if a <= num <= b:
                region = name
                break
        tm = int(region[2]) if region.startswith('TM') else 0
        generic = '%d.%d' % (tm, 50 + num - HELIX_REF[tm]) if tm else '.'
        if num in ANGII_SITE:
            site = 'angII_site'
        elif num in GPROT_IFACE:
            site = 'gprotein_interface'
        else:
            site = 'other'
        rows.append(dict(index=i, residue_id='R%03d' % num, residue_number=num,
                         region=region, tm_helix=tm, generic_number=generic,
                         site=site, is_pocket=int(num in POCKET)))
    return pd.DataFrame(rows)


def _chain(rng, start, end, n_mid, prefer):
    nodes = [int(start)]
    pool = [int(p) for p in rng.permutation(np.asarray(prefer, dtype=int))]
    for k in range(n_mid):
        placed = False
        for cand in pool + list(rng.integers(0, N_RES, size=600)):
            c = int(cand)
            if c in nodes or c == end or abs(c - nodes[-1]) < 4:
                continue
            if k == n_mid - 1 and abs(c - end) < 4:
                continue
            nodes.append(c)
            placed = True
            break
        if not placed:
            return None
    if abs(nodes[-1] - int(end)) < 4:
        return None
    nodes.append(int(end))
    return nodes


def planted_edges(seed):
    rng = np.random.default_rng(seed + 101)
    edges = {}
    deg = np.zeros(N_RES, dtype=int)

    def add(i, j, w, maxdeg=9):
        i, j = int(i), int(j)
        if i == j or abs(i - j) < 4:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in edges or deg[a] >= maxdeg or deg[b] >= maxdeg:
            return False
        edges[(a, b)] = float(w)
        deg[a] += 1
        deg[b] += 1
        return True

    pref = [p - 1 for p in POCKET]
    starts = [int(v) - 1 for v in rng.permutation(np.asarray(ANGII_SITE))]
    ends = [int(v) - 1 for v in rng.permutation(np.asarray(GPROT_IFACE))]
    for k in range(3):
        chain = _chain(rng, starts[k], ends[k], 5 + k, pref)
        if chain is None:
            continue
        for a, b in zip(chain[:-1], chain[1:]):
            add(a, b, rng.uniform(0.55, 0.70) * rng.choice([-1.0, 1.0]))
    hubs = rng.choice(N_RES, size=16, replace=False)
    attempts = 0
    while len(edges) < 430 and attempts < 400000:
        attempts += 1
        if rng.random() < 0.45:
            i = int(rng.choice(hubs))
            j = int(rng.integers(0, N_RES))
        else:
            i = int(rng.integers(0, N_RES))
            j = int(rng.integers(0, N_RES))
        add(i, j, rng.uniform(0.30, 0.62) * rng.choice([-1.0, 1.0]))
    return edges


def make_dataset(seed, n_frames=N_FRAMES):
    edges = planted_edges(seed)
    s = np.zeros((N_RES, N_RES))
    for (i, j), w in edges.items():
        s[i, j] = s[j, i] = w
    lam = float(np.linalg.eigvalsh(s)[0])
    d = max(0.9, -lam + 0.20)
    omega = s + d * np.eye(N_RES)
    sigma = np.linalg.inv(omega)
    chol = np.linalg.cholesky(sigma)
    rng = np.random.default_rng(seed)
    z = chol @ rng.standard_normal((N_RES, n_frames))
    rho, a = 0.5, float(np.sqrt(0.75))
    x = np.empty_like(z)
    x[:, 0] = z[:, 0]
    for t in range(1, n_frames):
        x[:, t] = rho * x[:, t - 1] + a * z[:, t]
    means = rng.uniform(-140.0, -60.0, size=N_RES)
    scales = rng.uniform(6.0, 12.0, size=N_RES)
    energies = (x * scales[:, None] + means[:, None]).T.astype(np.float32)

    pmat = -omega / d
    np.fill_diagonal(pmat, 1.0)
    wdeg = np.zeros(N_RES)
    for (i, j) in edges:
        wdeg[i] += abs(pmat[i, j])
        wdeg[j] += abs(pmat[i, j])
    zc = (wdeg - wdeg.mean()) / (wdeg.std() + 1e-12)
    ddg = np.clip(1.2 + 1.6 * zc + rng.normal(0.0, 1.0, N_RES), 0.0, 12.0)
    gq = np.clip(0.62 - 0.055 * ddg + rng.normal(0.0, 0.06, N_RES), 0.0, 1.0)
    nsub = rng.integers(12, 20, N_RES)
    ann = annotation_table()
    mut = pd.DataFrame(dict(
        index=np.arange(N_RES),
        residue_id=ann['residue_id'].to_numpy(),
        wt_aa=[AA[int(v)] for v in rng.integers(0, 20, N_RES)],
        mean_ddg=np.round(ddg, 4),
        n_substitutions=nsub,
        gq_activity_score=np.round(gq, 4),
    ))
    return energies, mut, ann


def write_dataset(outdir, energies, mut, ann):
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'at1r_energies.npy'), energies)
    mut.to_csv(os.path.join(outdir, 'at1r_mutscan.tsv'), sep='\t', index=False)
    ann.to_csv(os.path.join(outdir, 'at1r_annotation.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='/tmp/gold_data')
    ap.add_argument('--seed', type=int, default=PUBLIC_SEED)
    args = ap.parse_args()
    e, m, a = make_dataset(args.seed)
    write_dataset(args.outdir, e, m, a)
    print('wrote dataset to %s' % args.outdir)
