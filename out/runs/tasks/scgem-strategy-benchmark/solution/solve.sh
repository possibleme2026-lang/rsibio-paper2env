#!/bin/bash
set -euo pipefail

mkdir -p /app/output/models

cat > /app/solve.py <<'PY'
import json
import os

import numpy as np
import pandas as pd

DATA = '/app/data'
OUT = '/app/output'


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


def ev(node, gidx, vec):
    if node[0] == 'gene':
        i = gidx.get(node[1])
        return vec[i] if i is not None else 0.0
    vals = [ev(c, gidx, vec) for c in node[1]]
    return min(vals) if node[0] == 'and' else max(vals)


def pre(X, m):
    if m == 'cpm':
        t = np.clip(X.sum(axis=1, keepdims=True), 1.0, None)
        return X / t * 1e6
    if m == 'cpm_log2':
        t = np.clip(X.sum(axis=1, keepdims=True), 1.0, None)
        return np.log2(X / t * 1e6 + 1.0)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    return (X - mu) / np.where(sd == 0, 1.0, sd)


def build(V, gidx, nodes, const, mem, param):
    n = V.shape[0]
    out = np.zeros((n, len(nodes)), dtype=int)
    if mem == 'boolean':
        B = np.zeros_like(V)
        for j in range(V.shape[1]):
            B[:, j] = V[:, j] > np.percentile(V[:, j], param)
        for c in range(n):
            for r in range(len(nodes)):
                out[c, r] = 1 if (const[r] or ev(nodes[r], gidx, B[c])) else 0
    else:
        ne = np.array([x is not None for x in nodes])
        for c in range(n):
            sc = np.array([0.0 if x is None else ev(x, gidx, V[c]) for x in nodes])
            tau = np.percentile(sc[ne], param)
            out[c] = ((sc >= tau) | const).astype(int)
    return out


def jacc(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def main():
    counts = pd.read_csv(os.path.join(DATA, 'expression_counts.tsv'), sep='\t', index_col=0)
    pert = pd.read_csv(os.path.join(DATA, 'expression_counts_perturbed.tsv'), sep='\t', index_col=0)
    meta = pd.read_csv(os.path.join(DATA, 'cell_metadata.tsv'), sep='\t')
    model = pd.read_csv(os.path.join(DATA, 'model_reactions.tsv'), sep='\t')
    ref = pd.read_csv(os.path.join(DATA, 'reference_models.tsv'), sep='\t')
    grid = pd.read_csv(os.path.join(DATA, 'strategy_grid.tsv'), sep='\t')

    cells = [str(c) for c in counts.index]
    rids = [str(r) for r in model['reaction_id']]
    rpos = {r: i for i, r in enumerate(rids)}
    gidx = {str(g): i for i, g in enumerate(counts.columns)}
    gprs = ['' if pd.isna(g) else str(g) for g in model['gpr']]
    nodes = [parse_gpr(g) if g.strip() else None for g in gprs]
    const = np.array([(x is None) or (int(e) == 1)
                      for x, e in zip(nodes, model['essential'])], dtype=bool)
    ct = dict(zip([str(c) for c in meta['cell_id']], [str(t) for t in meta['cell_type']]))

    refv = {}
    for t, grp in ref.groupby('cell_type'):
        v = np.zeros(len(rids), dtype=bool)
        for r in grp['reaction_id']:
            r = str(r)
            if r in rpos:
                v[rpos[r]] = True
        refv[str(t)] = v

    Xr = counts.values.astype(float)
    Xp = pert.values.astype(float)
    same = np.array([[ct[a] == ct[b] for b in cells] for a in cells])
    iu = np.triu_indices(len(cells), 1)

    os.makedirs(os.path.join(OUT, 'models'), exist_ok=True)
    rows = []
    for _, s in grid.iterrows():
        sid = str(s['strategy_id'])
        pp = str(s['preprocessing'])
        mem = str(s['mem'])
        param = int(s['param'])
        M = build(pre(Xr, pp), gidx, nodes, const, mem, param)
        Mp = build(pre(Xp, pp), gidx, nodes, const, mem, param)
        dfm = pd.DataFrame(M, index=cells, columns=rids)
        dfm.index.name = 'cell_id'
        dfm.to_csv(os.path.join(OUT, 'models', sid + '.tsv'), sep='\t')
        Mi = M.astype(np.int64)
        inter = Mi.dot(Mi.T)
        sizes = Mi.sum(axis=1)
        union = np.maximum(sizes[:, None] + sizes[None, :] - inter, 1)
        J = inter / union
        acc = float(np.mean([jacc(M[i], refv[ct[cells[i]]]) for i in range(len(cells))]))
        rob = float(np.mean([jacc(M[i], Mp[i]) for i in range(len(cells))]))
        spec = float(J[iu][same[iu]].mean() - J[iu][~same[iu]].mean())
        rows.append({'strategy_id': sid, 'preprocessing': pp, 'mem': mem, 'param': param,
                     'mean_model_size': float(M.sum(axis=1).mean()),
                     'accuracy': acc, 'robustness': rob, 'specificity': spec,
                     'composite': 0.5 * acc + 0.3 * rob + 0.2 * spec})

    met = pd.DataFrame(rows)
    met = met[['strategy_id', 'preprocessing', 'mem', 'param', 'mean_model_size',
               'accuracy', 'robustness', 'specificity', 'composite']]
    met.to_csv(os.path.join(OUT, 'strategy_metrics.tsv'), sep='\t', index=False)

    best = met.sort_values(['composite', 'strategy_id'], ascending=[False, True]).iloc[0]
    with open(os.path.join(OUT, 'selection.json'), 'w') as fh:
        json.dump({'best_strategy': str(best['strategy_id']),
                   'composite': float(best['composite'])}, fh, indent=2)
    print('best strategy:', best['strategy_id'], 'composite:', best['composite'])


if __name__ == '__main__':
    main()
PY

python /app/solve.py
