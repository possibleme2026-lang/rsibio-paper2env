#!/usr/bin/env python3
'''Deterministic synthetic-data generator for the Hippo/TEAD kidney benchmark.

This is the verifier copy of environment/data/gen_inputs.py.  It regenerates
the exact task inputs from the protected seed so that the grader never has to
trust anything inside the agent-visible container.
'''

import os
import sys

import numpy as np
import pandas as pd

SEED = 20260801
GROUPS = ['vehicle', 'TEADi_low', 'TEADi_high', 'TEADi_recovery', 'YAPTAZ_KD']
N_REP = 6
N_GENES = 2000

ALDOSTERONE_SIGNALING = [
    'Nr3c2', 'Hsd11b2', 'Scnn1a', 'Scnn1b', 'Scnn1g', 'Sgk1', 'Nedd4l', 'Wnk1',
    'Wnk4', 'Stk39', 'Oxsr1', 'Slc12a3', 'Kcnj1', 'Atp1a1', 'Atp1b1', 'Atp1b2',
    'Aqp2', 'Aqp3', 'Aqp4', 'Ppp1r1b', 'Pde4d', 'Adrb2', 'Agt', 'Agtr1a', 'Ren1',
    'Umod', 'Klk1', 'Cldn8', 'Cldn16', 'Slc8a1', 'Fxyd2', 'Fxyd4', 'Ace', 'Atp6v1a',
]
YAP_TEAD_TARGETS = [
    'Ccn1', 'Ccn2', 'Ankrd1', 'Amotl2', 'Gli1', 'Thbs1', 'Myc', 'Lats2', 'Tead1',
    'Wwc1', 'Rassf2', 'Jub', 'Arhgap29', 'Trps1', 'Foxm1', 'Cdc20', 'Cav1', 'Igfbp3',
    'Dkk1', 'Nuak2', 'F3', 'Tgfb2', 'Serpine1', 'Axl', 'Sox4', 'Tead4', 'Wwtr1', 'Rassf1',
]
KIDNEY_TUBULAR_INJURY = [
    'Havcr1', 'Lcn2', 'Clu', 'Timp1', 'Spp1', 'Icam1', 'Vim', 'Col1a1', 'Col3a1',
    'Fn1', 'Cd44', 'Sox9', 'Ncam1', 'Ccl2', 'Mki67', 'Pcna', 'Top2a', 'Birc5', 'Anln',
    'Tgfbi', 'Cxcl1', 'Cxcl2', 'Lgals3', 'S100a8', 'S100a9', 'Mmp7', 'Mmp9',
]
DECOYS = [('DECOY_MITO_OXPHOS', 0.18), ('DECOY_RIBOSOME', -0.15), ('DECOY_OLFACTORY', 0.02)]
DECOY_SIZE = 28
PREFIXES = ['Slc', 'Zfp', 'Akap', 'Cyp', 'Fam', 'Tmem', 'Nduf', 'Rps', 'Rpl', 'Usp',
            'Dnaj', 'Ccdc', 'Ankrd', 'Wdr', 'Zbtb', 'Eif', 'Pde', 'Mapk', 'Prk', 'Cacn',
            'Kcn', 'Atp', 'Abc', 'S100', 'Hspa', 'Tubb', 'Col', 'Mmp', 'Serpin', 'Gm',
            '4930', 'Ccl', 'Cxcl', 'Igf', 'Fgf', 'Lrrc', 'Ttc', 'Plek', 'Nfat', 'Rho']


def _new_symbol(rng, taken):
    while True:
        prefix = PREFIXES[int(rng.integers(0, len(PREFIXES)))]
        num = int(rng.integers(1, 999))
        roll = rng.random()
        if roll < 0.30:
            sym = '%s%d' % (prefix, num)
        elif roll < 0.55:
            sym = '%s%d%s' % (prefix, num, 'a' if rng.random() < 0.5 else 'b')
        elif roll < 0.75:
            sym = '%s%dRik' % (prefix, num)
        elif roll < 0.90:
            sym = '%s%dc' % (prefix, num)
        else:
            sym = '%s%dos' % (prefix, num)
        if sym not in taken:
            taken.add(sym)
            return sym


def generate(outdir):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    panels = [('ALDOSTERONE_SIGNALING', list(ALDOSTERONE_SIGNALING)),
              ('YAP_TEAD_TARGETS', list(YAP_TEAD_TARGETS)),
              ('KIDNEY_TUBULAR_INJURY', list(KIDNEY_TUBULAR_INJURY))]
    taken = set()
    for _, members in panels:
        taken.update(members)
    genes = []
    for _, members in panels:
        genes.extend(members)
    while len(genes) < N_GENES:
        genes.append(_new_symbol(rng, taken))
    n = len(genes)
    pos = {g: i for i, g in enumerate(genes)}
    n_panel = sum(len(m) for _, m in panels)

    # baseline expression (median ~ 450 counts) with a low-expression tail
    base = np.exp(rng.normal(6.1, 1.3, n))
    low = rng.random(n) < 0.10
    base[low] = np.exp(rng.normal(2.2, 0.7, int(low.sum())))
    base = np.clip(base, 2.0, 40000.0)

    # fold-change programme: columns = low, high, recovery, KD
    lfc = rng.normal(0.0, 0.10, (n, 4))
    mod = rng.random(n) < 0.06
    k = int(mod.sum())
    lfc[mod, 1] = rng.normal(0.0, 0.45, k)
    lfc[mod, 0] = 0.5 * lfc[mod, 1] + rng.normal(0.0, 0.15, k)
    lfc[mod, 2] = 0.5 * lfc[mod, 1] + rng.normal(0.0, 0.20, k)
    lfc[mod, 3] = 0.5 * lfc[mod, 1] + rng.normal(0.0, 0.15, k)

    others = np.arange(n_panel, n)
    perm = rng.permutation(others)
    shared = perm[0:250]
    tead_only = perm[250:380]
    kd_only = perm[380:490]
    pool = perm[490:]

    for block, direction in ((shared[:125], -1.0), (shared[125:], 1.0)):
        eff = direction * np.abs(rng.normal(1.70, 0.40, len(block)))
        lfc[block, 1] = eff
        lfc[block, 0] = 0.45 * eff + rng.normal(0.0, 0.20, len(block))
        lfc[block, 3] = 0.75 * eff + rng.normal(0.0, 0.20, len(block))
        keep = rng.random(len(block)) < 0.70
        lfc[block, 2] = np.where(keep, 0.85, 0.22) * eff + rng.normal(0.0, 0.20, len(block))

    sign = np.where(rng.random(len(tead_only)) < 0.5, -1.0, 1.0)
    eff = sign * np.abs(rng.normal(1.60, 0.35, len(tead_only)))
    lfc[tead_only, 1] = eff
    lfc[tead_only, 0] = 0.45 * eff + rng.normal(0.0, 0.20, len(tead_only))
    lfc[tead_only, 2] = 0.50 * eff + rng.normal(0.0, 0.25, len(tead_only))
    lfc[tead_only, 3] = 0.12 * eff + rng.normal(0.0, 0.15, len(tead_only))

    sign = np.where(rng.random(len(kd_only)) < 0.5, -1.0, 1.0)
    eff = sign * np.abs(rng.normal(1.60, 0.35, len(kd_only)))
    lfc[kd_only, 3] = eff
    lfc[kd_only, 1] = 0.12 * eff + rng.normal(0.0, 0.18, len(kd_only))
    lfc[kd_only, 0] = 0.06 * eff + rng.normal(0.0, 0.15, len(kd_only))
    lfc[kd_only, 2] = 0.06 * eff + rng.normal(0.0, 0.15, len(kd_only))

    ia = np.array([pos[g] for g in ALDOSTERONE_SIGNALING])
    eff = -np.abs(rng.normal(1.85, 0.35, len(ia)))
    lfc[ia, 1] = eff
    lfc[ia, 0] = 0.45 * eff + rng.normal(0.0, 0.25, len(ia))
    lfc[ia, 3] = 0.70 * eff + rng.normal(0.0, 0.20, len(ia))
    keep = rng.random(len(ia)) < 0.70
    lfc[ia, 2] = np.where(keep, 0.85, 0.25) * eff + rng.normal(0.0, 0.20, len(ia))
    nm = rng.choice(len(ia), size=6, replace=False)
    lfc[ia[nm], 0] = lfc[ia[nm], 1] * rng.uniform(1.15, 1.35, 6) + rng.normal(0.0, 0.15, 6)

    iy = np.array([pos[g] for g in YAP_TEAD_TARGETS])
    eff = -np.abs(rng.normal(1.90, 0.35, len(iy)))
    lfc[iy, 1] = eff
    lfc[iy, 0] = 0.45 * eff + rng.normal(0.0, 0.22, len(iy))
    lfc[iy, 3] = 0.80 * eff + rng.normal(0.0, 0.20, len(iy))
    keep = rng.random(len(iy)) < 0.75
    lfc[iy, 2] = np.where(keep, 0.85, 0.25) * eff + rng.normal(0.0, 0.20, len(iy))

    ii = np.array([pos[g] for g in KIDNEY_TUBULAR_INJURY])
    eff = np.abs(rng.normal(1.70, 0.35, len(ii)))
    lfc[ii, 1] = eff
    lfc[ii, 0] = 0.45 * eff + rng.normal(0.0, 0.22, len(ii))
    lfc[ii, 3] = 0.60 * eff + rng.normal(0.0, 0.22, len(ii))
    lfc[ii, 2] = 0.40 * eff + rng.normal(0.0, 0.22, len(ii))

    decoy_members = {}
    pool = rng.permutation(pool)
    for i, (name, offset) in enumerate(DECOYS):
        idx = pool[i * DECOY_SIZE:(i + 1) * DECOY_SIZE]
        lfc[idx, 1] = offset + rng.normal(0.0, 0.12, len(idx))
        lfc[idx, 0] = 0.5 * offset + rng.normal(0.0, 0.12, len(idx))
        lfc[idx, 2] = 0.5 * offset + rng.normal(0.0, 0.12, len(idx))
        lfc[idx, 3] = 0.5 * offset + rng.normal(0.0, 0.12, len(idx))
        decoy_members[name] = [genes[j] for j in idx]

    sample_ids = ['%s_%d' % (g, r + 1) for g in GROUPS for r in range(N_REP)]
    conditions = [g for g in GROUPS for _ in range(N_REP)]
    col_of = {'vehicle': None, 'TEADi_low': 0, 'TEADi_high': 1,
              'TEADi_recovery': 2, 'YAPTAZ_KD': 3}
    lfc_mat = np.zeros((n, len(sample_ids)))
    for j, cond in enumerate(conditions):
        c = col_of[cond]
        if c is not None:
            lfc_mat[:, j] = lfc[:, c]
    size_factors = np.exp(rng.normal(0.0, 0.12, len(sample_ids)))
    mu = base[:, None] * size_factors[None, :] * np.power(2.0, lfc_mat)
    phi = np.clip(rng.gamma(2.0, 0.05, n), 0.005, 0.60)
    counts = rng.negative_binomial(1.0 / phi[:, None], 1.0 / (1.0 + phi[:, None] * mu))
    counts = counts.astype(np.int64)

    order = rng.permutation(n)
    counts_df = pd.DataFrame(counts[order], index=[genes[i] for i in order], columns=sample_ids)
    counts_df.index.name = 'gene'
    counts_df.to_csv(os.path.join(outdir, 'kidney_counts.csv'))

    dose = {'vehicle': 0, 'TEADi_low': 3, 'TEADi_high': 30, 'TEADi_recovery': 30, 'YAPTAZ_KD': 0}
    kind = {'vehicle': 'none', 'TEADi_low': 'pharmacological', 'TEADi_high': 'pharmacological',
            'TEADi_recovery': 'pharmacological', 'YAPTAZ_KD': 'genetic'}
    wash = {'TEADi_recovery': 14}
    meta = pd.DataFrame({
        'sample_id': sample_ids,
        'condition': conditions,
        'perturbation': [kind[c] for c in conditions],
        'dose_mg_kg': [dose[c] for c in conditions],
        'washout_days': [wash.get(c, 0) for c in conditions],
        'replicate': [r + 1 for _ in GROUPS for r in range(N_REP)],
    })
    meta.to_csv(os.path.join(outdir, 'sample_metadata.csv'), index=False)

    with open(os.path.join(outdir, 'gene_sets.gmt'), 'w') as fh:
        for name, members in panels:
            fh.write('%s\tcurated_gene_set\t%s\n' % (name, '\t'.join(members)))
        for name, _ in DECOYS:
            fh.write('%s\tdecoy_gene_set\t%s\n' % (name, '\t'.join(decoy_members[name])))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else '/app/data'
    generate(outdir)
    print('wrote synthetic benchmark inputs to %s' % outdir)


if __name__ == '__main__':
    main()
