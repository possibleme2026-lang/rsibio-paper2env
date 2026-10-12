#!/usr/bin/env python3
"""Deterministic generator of the scGEM benchmark task inputs (seed 20260802).

Run at image build time (python /app/data/gen_inputs.py). Writes
  expression_counts.tsv, expression_counts_perturbed.tsv, cell_metadata.tsv,
  model_reactions.tsv, reference_models.tsv, strategy_grid.tsv
into the directory that contains this script.
"""
import os

import numpy as np
import pandas as pd

SEED = 20260802
HERE = os.path.dirname(os.path.abspath(__file__))
N_CELLS, N_GENES, N_REACTIONS = 80, 600, 300
CELL_TYPES = ['hepatocyte', 'T_cell', 'macrophage', 'neuron']
DATASETS = ['GSE110_A', 'GSE110_B', 'GSE110_C']


def tokenize(s):
    return [t for t in s.replace('(', ' ( ').replace(')', ' ) ').split() if t]


def parse_gpr(s):
    toks = tokenize(s)
    pos = 0

    def p_or():
        nonlocal pos
        nodes = [p_and()]
        while pos < len(toks) and toks[pos].lower() == 'or':
            pos += 1
            nodes.append(p_and())
        return nodes[0] if len(nodes) == 1 else ('or', nodes)

    def p_and():
        nonlocal pos
        nodes = [p_atom()]
        while pos < len(toks) and toks[pos].lower() == 'and':
            pos += 1
            nodes.append(p_atom())
        return nodes[0] if len(nodes) == 1 else ('and', nodes)

    def p_atom():
        nonlocal pos
        t = toks[pos]
        if t == '(':
            pos += 1
            n = p_or()
            pos += 1
            return n
        pos += 1
        return ('gene', t)

    return p_or()


def eval_gpr(node, gv):
    if node[0] == 'gene':
        return float(gv.get(node[1], 0.0))
    vals = [eval_gpr(c, gv) for c in node[1]]
    return min(vals) if node[0] == 'and' else max(vals)


def main():
    rng = np.random.default_rng(SEED)
    cell_ids = ['CELL%03d' % i for i in range(1, N_CELLS + 1)]
    gene_ids = ['G%04d' % i for i in range(1, N_GENES + 1)]
    types = [CELL_TYPES[i % 4] for i in range(N_CELLS)]
    datasets = [DATASETS[(i // 4) % 3] for i in range(N_CELLS)]

    # --- expression: baseline + cell-type marker program + library size + dropout
    mu = rng.gamma(1.5, 1.0, N_GENES) * 2.0
    expectation = np.tile(mu, (N_CELLS, 1))
    for t in range(4):
        rows = np.array([i for i in range(N_CELLS) if types[i] == CELL_TYPES[t]])
        markers = rng.choice(N_GENES, size=60, replace=False)
        expectation[np.ix_(rows, markers)] *= rng.uniform(3.0, 9.0, size=60)
    lib = rng.lognormal(0.0, 0.30, N_CELLS)
    expectation = expectation * (lib / lib.mean())[:, None]
    counts = rng.poisson(expectation)
    counts[rng.random((N_CELLS, N_GENES)) < np.exp(-expectation / 2.5)] = 0
    counts = counts.astype(int)

    cdf = pd.DataFrame(counts, index=cell_ids, columns=gene_ids)
    cdf.index.name = 'cell_id'
    cdf.to_csv(os.path.join(HERE, 'expression_counts.tsv'), sep='\t')

    pert = counts.copy()
    nz = np.argwhere(pert > 0)
    sel = rng.random(len(nz)) < 0.10
    for i, j in nz[sel]:
        pert[i, j] = 0
    pdf = pd.DataFrame(pert, index=cell_ids, columns=gene_ids)
    pdf.index.name = 'cell_id'
    pdf.to_csv(os.path.join(HERE, 'expression_counts_perturbed.tsv'), sep='\t')

    pd.DataFrame({'cell_id': cell_ids, 'cell_type': types, 'dataset': datasets}).to_csv(
        os.path.join(HERE, 'cell_metadata.tsv'), sep='\t', index=False)

    # --- curated 300-reaction metabolic model with nested GPR rules
    subsystems = ['glycolysis', 'TCA cycle', 'oxidative phosphorylation',
                  'amino acid metabolism', 'lipid metabolism',
                  'nucleotide metabolism', 'transport', 'redox metabolism']
    gene_pool = gene_ids[:500]
    rids, gprs, subs, metas = [], [], [], []
    for i in range(N_REACTIONS):
        rids.append('R%04d' % (i + 1))
        subs.append(subsystems[i % len(subsystems)])
        k = int(rng.integers(2, 5))
        cpds = rng.choice(np.arange(1, 400), size=k, replace=False)
        coefs = rng.choice([-1, 1], size=k)
        metas.append(';'.join('cpd%05d:%d' % (c, s) for c, s in zip(cpds, coefs)))
        if i < 20:
            gprs.append('')
            continue
        ng = int(rng.choice([1, 1, 2, 2, 2, 3, 3]))
        gs = [str(x) for x in rng.choice(gene_pool, size=ng, replace=False)]
        if ng == 1:
            g = gs[0]
        elif ng == 2:
            g = gs[0] + ' and ' + gs[1] if rng.random() < 0.7 else gs[0] + ' or ' + gs[1]
        else:
            u = rng.random()
            if u < 0.50:
                g = '(' + gs[0] + ' and ' + gs[1] + ') or ' + gs[2]
            elif u < 0.75:
                g = '(' + gs[0] + ' or ' + gs[1] + ') and ' + gs[2]
            elif u < 0.90:
                g = gs[0] + ' and ' + gs[1] + ' and ' + gs[2]
            else:
                g = gs[0] + ' or (' + gs[1] + ' and ' + gs[2] + ')'
        gprs.append(g)

    essential = np.zeros(N_REACTIONS, dtype=int)
    essential[rng.choice(np.arange(20, N_REACTIONS), size=8, replace=False)] = 1

    mdf = pd.DataFrame({
        'reaction_id': rids,
        'reaction_name': ['rxn_%d' % (i + 1) for i in range(N_REACTIONS)],
        'gpr': gprs,
        'subsystem': subs,
        'metabolites': metas,
        'essential': essential,
    })
    mdf.to_csv(os.path.join(HERE, 'model_reactions.tsv'), sep='\t', index=False)

    # --- bulk-derived reference active sets per cell type
    cpm = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None) * 1e6
    logcpm = np.log2(cpm + 1.0)
    ref_rows = []
    for t in CELL_TYPES:
        idx = [i for i in range(N_CELLS) if types[i] == t]
        prof = logcpm[idx].mean(axis=0)
        gv = {gene_ids[j]: float(prof[j]) for j in range(N_GENES)}
        scores = [None if g == '' else eval_gpr(parse_gpr(g), gv) for g in gprs]
        vals = np.array([s for s in scores if s is not None])
        thr = float(np.percentile(vals, 50))
        for r, s, e in zip(rids, scores, essential):
            if s is None or s >= thr or e == 1:
                ref_rows.append({'cell_type': t, 'reaction_id': r})
    pd.DataFrame(ref_rows).to_csv(os.path.join(HERE, 'reference_models.tsv'), sep='\t', index=False)

    # --- the 18-strategy grid
    grid = []
    k = 1
    for pp in ['cpm', 'cpm_log2', 'zscore']:
        for mem in ['fuzzy', 'boolean']:
            for param in [50, 75, 90]:
                grid.append({'strategy_id': 'S%02d' % k, 'preprocessing': pp,
                             'mem': mem, 'param': param})
                k += 1
    pd.DataFrame(grid).to_csv(os.path.join(HERE, 'strategy_grid.tsv'), sep='\t', index=False)

    print('scGEM benchmark inputs written to', HERE)
    print('counts shape', counts.shape, 'reactions', N_REACTIONS, 'strategies', len(grid))


if __name__ == '__main__':
    main()
