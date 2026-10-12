#!/usr/bin/env python3
'''Deterministic generator for the synthetic ovine Fontan sEV cohort.

Mimics the data structure of the translational ovine Fontan model used to
build the Fontan EV Score: longitudinal serum small extracellular vesicle
(sEV) small-RNA sequencing, paired clinical chemistry, liver elastography
and an ordinal fibrosis grade, with a pre-specified train/test split.

Usage: python gen_inputs.py --outdir /app/data
'''

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260731
N_MIRNA = 180
MANUAL_MIRNAS = [
    'miR-21-5p', 'miR-122-5p', 'miR-125a-5p', 'miR-193b-5p', 'let-7a-5p',
    'miR-146a-5p', 'miR-155-5p', 'miR-223-3p', 'miR-26a-5p', 'miR-192-5p',
    'miR-29a-3p', 'miR-27b-3p', 'miR-130a-3p', 'miR-99a-5p', 'miR-144-3p',
    'miR-451a', 'miR-486-5p', 'miR-16-5p',
]


def mirna_names(rng):
    used = {21, 122, 125, 193, 146, 155, 223, 26, 192, 29, 27, 130, 99,
            144, 451, 486, 16}
    pool = np.array([n for n in range(20, 700) if n not in used])
    rng.shuffle(pool)
    extra = []
    for i in range(N_MIRNA - len(MANUAL_MIRNAS)):
        arm = '5p' if i % 2 == 0 else '3p'
        extra.append('miR-%d-%s' % (int(pool[i]), arm))
    return MANUAL_MIRNAS + extra


def build_clinical(rng):
    rows = []
    for i in range(1, 20):
        sheep = 'OV%02d' % i
        group = 'Fontan' if i <= 13 else 'Sham'
        if group == 'Fontan':
            times = [3, 6, 9, 12] if i <= 7 else [3, 9, 12]
        else:
            times = [3, 12]
        for t in times:
            rows.append({'sample_id': '%s_%02dm' % (sheep, t), 'sheep_id': sheep,
                         'group': group, 'months_post_surgery': t})
    clin = pd.DataFrame(rows)
    n = len(clin)
    sheep_effect = {s: float(rng.normal(0.0, 0.45))
                    for s in sorted(clin['sheep_id'].unique())}
    latent = []
    for r in clin.itertuples():
        mu = (-1.15 + 1.05 * (r.group == 'Fontan')
              + 1.45 * np.log1p(r.months_post_surgery) / np.log1p(12)
              + sheep_effect[r.sheep_id])
        latent.append(mu + rng.normal(0.0, 0.30))
    latent = np.asarray(latent)
    q = np.quantile(latent, [0.25, 0.58, 0.80])
    grade = np.digitize(latent, q).astype(int)
    z = (latent - latent.mean()) / latent.std(ddof=1)

    clin['age_years'] = np.round(
        0.75 + clin['months_post_surgery'] / 12.0 + rng.normal(0, 0.05, n), 2)
    clin['ast_u_l'] = np.round(np.exp(3.38 + 0.14 * z + rng.normal(0, 0.20, n)), 1)
    clin['alt_u_l'] = np.round(np.exp(2.98 + 0.10 * z + rng.normal(0, 0.18, n)), 1)
    clin['platelets_10e9_l'] = np.round(
        np.exp(5.42 - 0.075 * z + rng.normal(0, 0.13, n)), 1)
    clin['liver_stiffness_kpa'] = np.round(4.3 + 1.6 * z + rng.normal(0, 0.35, n), 1)
    clin['fibrosis_grade'] = grade

    split = np.array(['train'] * n, dtype=object)
    for g in sorted(set(grade.tolist())):
        idx = np.where(grade == g)[0]
        rng.shuffle(idx)
        n_test = int(round(0.38 * len(idx)))
        split[idx[:n_test]] = 'test'
    clin['split'] = split
    return clin, z


def build_counts(rng, clin, z):
    names = mirna_names(rng)
    n = len(clin)
    base = np.exp(rng.normal(7.6, 1.05, N_MIRNA))
    rare = rng.choice(N_MIRNA, size=34, replace=False)
    base[rare] = rng.uniform(0.2, 3.5, size=34)
    pool = np.array([i for i in range(N_MIRNA) if i not in set(rare.tolist())])
    info = rng.choice(pool, size=20, replace=False)
    raw = rng.normal(0.0, 1.0, size=20)
    load = np.zeros(N_MIRNA)
    load[info] = np.sign(raw) * np.maximum(np.abs(raw), 0.55)

    sheep_ids = sorted(clin['sheep_id'].unique())
    sheep_idx = {s: i for i, s in enumerate(sheep_ids)}
    sheep_fx = rng.normal(0.0, 0.22, size=(len(sheep_ids), N_MIRNA))
    lib = rng.normal(0.0, 0.15, size=n)

    mu = np.zeros((n, N_MIRNA))
    for i, r in enumerate(clin.itertuples()):
        mu[i] = base * np.exp(load * z[i] + sheep_fx[sheep_idx[r.sheep_id]] + lib[i])
    counts = rng.negative_binomial(12, 12.0 / (12.0 + mu))
    return pd.DataFrame(counts.astype(int), index=clin['sample_id'], columns=names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='/app/data')
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    clin, z = build_clinical(rng)
    counts = build_counts(rng, clin, z)
    clin.to_csv(out / 'clinical.csv', index=False)
    counts.to_csv(out / 'mirna_counts.csv')
    print('wrote %d samples x %d miRNAs (seed %d)' % (len(clin), counts.shape[1], SEED))


if __name__ == '__main__':
    main()
