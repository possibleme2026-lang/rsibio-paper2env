#!/usr/bin/env python3
'''Deterministic generator for the C. elegans OXPHOS confidence-aware learning task.

The task container builds /app/data by running this script once and then deletes
it, so the task image only ever contains observable data.  The same code, run
with --labels-out, produces the hidden OXPHOS ground-truth labels used by the
verifier.
'''
import argparse
import json
import os

import numpy as np

SEED = 20260529
N_GENES = 800
N_HC = 72
N_LC = 132
N_NEG = 120
N_UNL = 476
N_LC_TRUE = 24
N_UNL_TRUE = 64
COMPLEXES = ['Complex I', 'Complex II', 'Complex III', 'Complex IV', 'Complex V']
BULK_TIMES = ['emb_1h', 'emb_3h', 'emb_5h', 'emb_7h', 'L1', 'L2a', 'L2b',
              'L3a', 'L3b', 'L4a', 'L4b', 'L4c', 'YA', 'A1', 'A2', 'A3']
N_EMB_CELLS = 1500
N_ADULT_CELLS = 1800
EMB_TYPES = ['early_emb', 'late_emb', 'mixed_emb']
ADULT_TYPES = ['neuron', 'intestine', 'hypodermis', 'body_wall_muscle', 'germline']
OTHER_TERMS = ['GO:0006412 translation', 'GO:0006099 tricarboxylic acid cycle',
               'GO:0006886 intracellular protein transport', 'GO:0006281 DNA repair',
               'GO:0035556 intracellular signal transduction', 'GO:0007017 microtubule-based process']


def _unit_profile(rng, n=16):
    x = np.linspace(0.0, 1.0, n)
    p = np.zeros(n)
    for j in range(3):
        p += rng.normal() * np.sin((j + 1) * np.pi * x * rng.uniform(0.7, 1.4) + rng.uniform(0.0, 6.28))
    p -= p.mean()
    nrm = float(np.linalg.norm(p))
    return p / nrm if nrm > 1e-9 else p


def build(seed=SEED):
    rng = np.random.default_rng(seed)
    gene_ids = np.array([f'WBGene{n:08d}' for n in range(1001, 1001 + N_GENES)])
    gene_names = np.array([f'ceg-{i:04d}' for i in range(1, N_GENES + 1)])

    roles = np.array(['HC'] * N_HC + ['LC'] * N_LC + ['NEG'] * N_NEG + ['UNL'] * N_UNL)
    roles = roles[rng.permutation(N_GENES)]

    idx_hc = np.sort(np.where(roles == 'HC')[0])
    idx_lc = np.sort(np.where(roles == 'LC')[0])
    idx_unl = np.sort(np.where(roles == 'UNL')[0])
    idx_neg = np.sort(np.where(roles == 'NEG')[0])

    is_true = np.zeros(N_GENES, dtype=bool)
    is_true[idx_hc] = True
    is_true[np.sort(rng.choice(idx_lc, size=N_LC_TRUE, replace=False))] = True
    is_true[np.sort(rng.choice(idx_unl, size=N_UNL_TRUE, replace=False))] = True

    complex_of = np.array(['' for _ in range(N_GENES)], dtype=object)
    for k, g in enumerate(rng.permutation(idx_hc)):
        complex_of[g] = COMPLEXES[k % len(COMPLEXES)]
    for g in idx_lc:
        complex_of[g] = COMPLEXES[int(rng.integers(0, len(COMPLEXES)))]

    true_idx = np.where(is_true)[0]
    order = rng.permutation(true_idx)
    n_true = int(len(order))
    n_bulk_only = int(round(0.45 * n_true))
    n_sc_only = int(round(0.45 * n_true))
    group = np.zeros(N_GENES, dtype=int)
    group[order[:n_bulk_only]] = 1
    group[order[n_bulk_only:n_bulk_only + n_sc_only]] = 2
    group[order[n_bulk_only + n_sc_only:]] = 3

    bulk_strength = rng.normal(0.0, 0.30, size=N_GENES)
    sc_strength = np.abs(rng.normal(0.0, 0.10, size=N_GENES))
    for g in range(N_GENES):
        if not is_true[g]:
            continue
        strong_b = group[g] in (1, 3)
        strong_s = group[g] in (2, 3)
        if roles[g] == 'HC':
            bulk_strength[g] = rng.uniform(2.4, 3.6) if strong_b else rng.uniform(0.0, 0.8)
            sc_strength[g] = rng.uniform(0.85, 1.10) if strong_s else rng.uniform(0.0, 0.25)
        elif roles[g] == 'LC':
            bulk_strength[g] = rng.uniform(1.7, 2.6) if strong_b else rng.uniform(0.0, 0.7)
            sc_strength[g] = rng.uniform(0.60, 0.95) if strong_s else rng.uniform(0.0, 0.20)
        else:
            bulk_strength[g] = rng.uniform(1.8, 3.0) if strong_b else rng.uniform(0.0, 0.8)
            sc_strength[g] = rng.uniform(0.65, 1.00) if strong_s else rng.uniform(0.0, 0.22)

    lc_false = np.array([g for g in idx_lc if not is_true[g]])
    unl_false = np.array([g for g in idx_unl if not is_true[g]])
    unl_shuf = rng.permutation(unl_false)
    bulk_decoy = unl_shuf[:35]
    sc_decoy_adult = unl_shuf[35:65]
    lc_shuf = rng.permutation(lc_false)
    sc_decoy_embryo = lc_shuf[:30]
    bulk_friend = lc_shuf[30:50]
    bulk_strength[bulk_decoy] = rng.uniform(0.9, 1.4, size=len(bulk_decoy))
    bulk_strength[bulk_friend] = rng.uniform(0.8, 1.3, size=len(bulk_friend))

    emb_load = sc_strength * rng.uniform(0.78, 1.05, size=N_GENES)
    adult_load = sc_strength * rng.uniform(0.78, 1.05, size=N_GENES)
    emb_load[sc_decoy_embryo] = rng.uniform(0.40, 0.55, size=len(sc_decoy_embryo))
    adult_load[sc_decoy_adult] = rng.uniform(0.40, 0.55, size=len(sc_decoy_adult))

    p_ox = _unit_profile(rng)
    factors = np.stack([_unit_profile(rng) for _ in range(8)], axis=0)
    mu_bulk = rng.uniform(8.0, 12.0, size=N_GENES)
    w_nuis = rng.normal(0.0, 0.30, size=(N_GENES, 8))
    bulk = mu_bulk[:, None] + 1.5 * bulk_strength[:, None] * p_ox[None, :] + w_nuis @ factors
    bulk = bulk + rng.normal(0.0, 1.0, size=(N_GENES, len(BULK_TIMES)))

    def make_atlas(n_cells, load0, types):
        k = 6
        u = rng.normal(0.0, 1.0, size=(n_cells, k))
        g = rng.normal(0.0, 0.18, size=(N_GENES, k))
        g[:, 0] = load0
        mu_sc = rng.uniform(0.4, 3.6, size=N_GENES)
        x = mu_sc[:, None] + g @ u.T
        x = x + rng.normal(0.0, 0.65, size=(N_GENES, n_cells))
        x = np.clip(x, 0.0, None)
        drop = rng.uniform(0.08, 0.35, size=N_GENES)
        mask = rng.random(size=(N_GENES, n_cells)) < drop[:, None]
        x[mask] = 0.0
        return x.T.astype(np.float32), np.array(rng.choice(types, size=n_cells))

    sc_emb_x, sc_emb_types = make_atlas(N_EMB_CELLS, emb_load, EMB_TYPES)
    sc_adt_x, sc_adt_types = make_atlas(N_ADULT_CELLS, adult_load, ADULT_TYPES)

    cand_idx = np.sort(np.concatenate([idx_lc, idx_unl]))
    return {
        'gene_ids': gene_ids,
        'gene_names': gene_names,
        'roles': roles,
        'is_oxphos': is_true,
        'complex_of': complex_of,
        'idx_hc': idx_hc,
        'idx_lc': idx_lc,
        'idx_neg': idx_neg,
        'idx_unl': idx_unl,
        'bulk': bulk,
        'sc_embryonic': (sc_emb_x, sc_emb_types),
        'sc_adult': (sc_adt_x, sc_adt_types),
        'candidate_idx': cand_idx,
        'candidate_ids': [str(gene_ids[i]) for i in cand_idx],
        'candidate_labels': is_true[cand_idx].astype(int).tolist(),
        'candidate_low_conf': [bool(roles[i] == 'LC') for i in cand_idx],
    }


def write_files(d, outdir):
    for sub in ('expression', 'annotations', 'candidates'):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
    ids = d['gene_ids']

    with open(os.path.join(outdir, 'expression', 'bulk_timecourse.tsv'), 'w') as fh:
        fh.write('gene_id\t' + '\t'.join(BULK_TIMES) + '\n')
        for i, g in enumerate(ids):
            vals = '\t'.join('%.4f' % v for v in d['bulk'][i])
            fh.write('%s\t%s\n' % (g, vals))

    emb_x, emb_types = d['sc_embryonic']
    np.savez_compressed(os.path.join(outdir, 'expression', 'sc_embryonic.npz'),
                        genes=ids, cells=np.arange(emb_x.shape[0]),
                        cell_types=emb_types, X=emb_x)
    adt_x, adt_types = d['sc_adult']
    np.savez_compressed(os.path.join(outdir, 'expression', 'sc_adult.npz'),
                        genes=ids, cells=np.arange(adt_x.shape[0]),
                        cell_types=adt_types, X=adt_x)

    ev = ['IDA', 'IMP']
    with open(os.path.join(outdir, 'annotations', 'high_confidence_oxphos.tsv'), 'w') as fh:
        fh.write('gene_id\tgene_name\tcomplex\tevidence\n')
        for k, i in enumerate(d['idx_hc']):
            fh.write('%s\t%s\t%s\t%s\n' % (ids[i], d['gene_names'][i], d['complex_of'][i], ev[k % 2]))

    with open(os.path.join(outdir, 'annotations', 'low_confidence_oxphos.tsv'), 'w') as fh:
        fh.write('gene_id\tgene_name\tcomplex\tevidence\n')
        for k, i in enumerate(d['idx_lc']):
            fh.write('%s\t%s\t%s\t%s\n' % (ids[i], d['gene_names'][i], d['complex_of'][i],
                                             'ISS' if k % 2 == 0 else 'IEA'))

    with open(os.path.join(outdir, 'annotations', 'other_process_annotations.tsv'), 'w') as fh:
        fh.write('gene_id\tgene_name\tgo_term\tevidence\n')
        for k, i in enumerate(d['idx_neg']):
            fh.write('%s\t%s\t%s\t%s\n' % (ids[i], d['gene_names'][i],
                                             OTHER_TERMS[k % len(OTHER_TERMS)], ev[k % 2]))

    low = set(int(i) for i in d['idx_lc'])
    with open(os.path.join(outdir, 'candidates', 'gene_universe.tsv'), 'w') as fh:
        fh.write('gene_id\tgene_name\thas_low_confidence_oxphos_annotation\n')
        for i in d['candidate_idx']:
            fh.write('%s\t%s\t%s\n' % (ids[i], d['gene_names'][i],
                                        'yes' if int(i) in low else 'no'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--labels-out', default=None)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    d = build(args.seed)
    if args.outdir:
        write_files(d, args.outdir)
    if args.labels_out:
        parent = os.path.dirname(os.path.abspath(args.labels_out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.labels_out, 'w') as fh:
            json.dump({'candidate_ids': d['candidate_ids'],
                       'candidate_labels': d['candidate_labels'],
                       'candidate_low_conf': d['candidate_low_conf'],
                       'n_high_confidence': int(len(d['idx_hc']))}, fh)


if __name__ == '__main__':
    main()
