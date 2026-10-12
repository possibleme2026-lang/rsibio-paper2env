#!/usr/bin/env python3
"""Reference (gold) implementation of the scGEM strategy-benchmark pipeline.

Reads only the task inputs from <data_dir> and deterministically recomputes
  * the 18 per-strategy x per-cell reaction-activity models
  * accuracy / robustness / specificity / mean_model_size / composite
  * the winning strategy id
"""
import os

import numpy as np
import pandas as pd


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


def eval_node(node, gidx, vec):
    if node[0] == 'gene':
        i = gidx.get(node[1])
        return vec[i] if i is not None else 0.0
    vals = [eval_node(c, gidx, vec) for c in node[1]]
    return min(vals) if node[0] == 'and' else max(vals)


def preprocess(X, method):
    if method == 'cpm':
        tot = np.clip(X.sum(axis=1, keepdims=True), 1.0, None)
        return X / tot * 1e6
    if method == 'cpm_log2':
        tot = np.clip(X.sum(axis=1, keepdims=True), 1.0, None)
        return np.log2(X / tot * 1e6 + 1.0)
    if method == 'zscore':
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, ddof=0, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        return (X - mu) / sd
    raise ValueError('unknown preprocessing ' + str(method))


def build_models(V, gidx, nodes, const, mem, param):
    n_cells = V.shape[0]
    n_rx = len(nodes)
    out = np.zeros((n_cells, n_rx), dtype=int)
    if mem == 'boolean':
        B = np.zeros_like(V)
        for j in range(V.shape[1]):
            col = V[:, j]
            B[:, j] = col > np.percentile(col, param)
        for c in range(n_cells):
            vec = B[c]
            for r in range(n_rx):
                out[c, r] = 1 if (const[r] or eval_node(nodes[r], gidx, vec)) else 0
    elif mem == 'fuzzy':
        nonempty = np.array([n is not None for n in nodes])
        for c in range(n_cells):
            vec = V[c]
            scores = np.array([0.0 if n is None else eval_node(n, gidx, vec) for n in nodes])
            tau = np.percentile(scores[nonempty], param)
            out[c] = ((scores >= tau) | const).astype(int)
    else:
        raise ValueError('unknown mem ' + str(mem))
    return out


def jacc(a, b):
    u = np.logical_or(a, b).sum()
    if u == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / u)


def compute_gold(data_dir):
    counts = pd.read_csv(os.path.join(data_dir, 'expression_counts.tsv'), sep='\t', index_col=0)
    pert = pd.read_csv(os.path.join(data_dir, 'expression_counts_perturbed.tsv'), sep='\t', index_col=0)
    meta = pd.read_csv(os.path.join(data_dir, 'cell_metadata.tsv'), sep='\t')
    model = pd.read_csv(os.path.join(data_dir, 'model_reactions.tsv'), sep='\t')
    ref = pd.read_csv(os.path.join(data_dir, 'reference_models.tsv'), sep='\t')
    grid = pd.read_csv(os.path.join(data_dir, 'strategy_grid.tsv'), sep='\t')

    cells = [str(c) for c in counts.index]
    genes = [str(g) for g in counts.columns]
    rids = [str(r) for r in model['reaction_id']]
    rpos = {r: i for i, r in enumerate(rids)}
    gidx = {g: i for i, g in enumerate(genes)}

    gprs = ['' if pd.isna(g) else str(g) for g in model['gpr']]
    nodes = [parse_gpr(g) if g.strip() else None for g in gprs]
    const = np.array([(n is None) or (int(e) == 1)
                      for n, e in zip(nodes, model['essential'])], dtype=bool)

    ct = dict(zip([str(c) for c in meta['cell_id']], [str(t) for t in meta['cell_type']]))
    ref_vec = {}
    for t, grp in ref.groupby('cell_type'):
        v = np.zeros(len(rids), dtype=bool)
        for r in grp['reaction_id']:
            r = str(r)
            if r in rpos:
                v[rpos[r]] = True
        ref_vec[str(t)] = v

    Xr = counts.values.astype(float)
    Xp = pert.values.astype(float)
    same = np.array([[ct[a] == ct[b] for b in cells] for a in cells])
    iu = np.triu_indices(len(cells), 1)

    models = {}
    rows = []
    for _, s in grid.iterrows():
        sid = str(s['strategy_id'])
        pp = str(s['preprocessing'])
        mem = str(s['mem'])
        param = int(s['param'])
        M = build_models(preprocess(Xr, pp), gidx, nodes, const, mem, param)
        Mp = build_models(preprocess(Xp, pp), gidx, nodes, const, mem, param)
        models[sid] = M
        Mi = M.astype(np.int64)
        inter = Mi.dot(Mi.T)
        sizes = Mi.sum(axis=1)
        union = np.maximum(sizes[:, None] + sizes[None, :] - inter, 1)
        J = inter / union
        acc = float(np.mean([jacc(M[i].astype(bool), ref_vec[ct[cells[i]]])
                             for i in range(len(cells))]))
        rob = float(np.mean([jacc(M[i].astype(bool), Mp[i].astype(bool))
                             for i in range(len(cells))]))
        spec = float(J[iu][same[iu]].mean() - J[iu][~same[iu]].mean())
        rows.append({
            'strategy_id': sid,
            'preprocessing': pp,
            'mem': mem,
            'param': param,
            'mean_model_size': float(M.sum(axis=1).mean()),
            'accuracy': acc,
            'robustness': rob,
            'specificity': spec,
            'composite': 0.5 * acc + 0.3 * rob + 0.2 * spec,
        })

    metrics = pd.DataFrame(rows)
    order = metrics.sort_values(['composite', 'strategy_id'], ascending=[False, True])
    best = str(order.iloc[0]['strategy_id'])
    return {'cells': cells, 'genes': genes, 'reactions': rids, 'const_mask': const,
            'models': models, 'metrics': metrics, 'best': best}


if __name__ == '__main__':
    g = compute_gold(os.environ.get('SCGEM_DATA', '/app/data'))
    print(g['metrics'].to_string(index=False))
    print('best strategy:', g['best'])
