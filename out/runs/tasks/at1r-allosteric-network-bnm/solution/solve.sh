#!/bin/bash
set -euo pipefail
mkdir -p /app/output
cat > /app/analyze.py <<'PYEOF'
#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import hypergeom
from sklearn.covariance import ledoit_wolf


def coupling_matrix(energies):
    x = np.asarray(energies, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=0)
    sd[sd <= 0] = 1.0
    x = x / sd
    shrunk, _delta = ledoit_wolf(x, assume_centered=True)
    theta = np.linalg.inv(shrunk)
    d = np.sqrt(np.diag(theta))
    p = -theta / np.outer(d, d)
    np.fill_diagonal(p, 1.0)
    return 0.5 * (p + p.T)


def build_network(p, threshold, min_sep):
    n = p.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    keep = (np.abs(p[iu, ju]) >= threshold) & ((ju - iu) >= min_sep)
    ei, ej = iu[keep], ju[keep]
    ep = p[ei, ej]
    wdeg = np.zeros(n)
    np.add.at(wdeg, ei, np.abs(ep))
    np.add.at(wdeg, ej, np.abs(ep))
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for a, b, val in zip(ei, ej, ep):
        g.add_edge(int(a), int(b), weight=float(abs(val)))
    return ei, ej, ep, wdeg, g


def communities(g):
    active = [n for n in g.nodes if g.degree(n) > 0]
    sub = g.subgraph(active).copy()
    parts = list(nx.algorithms.community.greedy_modularity_communities(
        sub, weight='weight', resolution=1.0))
    assign = np.full(g.number_of_nodes(), -1, dtype=int)
    for cid, part in enumerate(parts):
        for node in part:
            assign[node] = cid
    return assign, parts


def shortest_paths(g, sources, targets, n_paths):
    h = nx.Graph()
    h.add_nodes_from(g.nodes)
    for a, b, data in g.edges(data=True):
        h.add_edge(a, b, weight=-np.log(max(abs(data['weight']), 1e-12)))
    cands = []
    for s in sorted(sources):
        dist, paths = nx.single_source_dijkstra(h, s, weight='weight')
        for t in sorted(targets):
            if t in dist:
                cands.append((float(dist[t]), int(s), int(t), [int(v) for v in paths[t]]))
    cands.sort(key=lambda r: (r[0], r[1], r[2]))
    chosen, used = [], set()
    for cost, s, t, path in cands:
        if s in used:
            continue
        chosen.append({'start': s, 'end': t, 'cost': cost, 'path': path})
        used.add(s)
        if len(chosen) == n_paths:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--energies', required=True)
    ap.add_argument('--mutations', required=True)
    ap.add_argument('--annotation', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--threshold', type=float, default=0.20)
    ap.add_argument('--min-separation', type=int, default=4)
    ap.add_argument('--top-hotspots', type=int, default=25)
    ap.add_argument('--n-paths', type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    energies = np.load(args.energies)
    mut = pd.read_csv(args.mutations, sep='\t').sort_values('index').reset_index(drop=True)
    ann = pd.read_csv(args.annotation, sep='\t').sort_values('index').reset_index(drop=True)

    p = coupling_matrix(energies)
    n = p.shape[0]
    ei, ej, ep, wdeg, g = build_network(p, args.threshold, args.min_separation)
    assign, parts = communities(g)

    ddg = mut['mean_ddg'].to_numpy(dtype=float)
    zs = lambda v: (v - v.mean()) / (v.std(ddof=0) + 1e-12)
    score = zs(wdeg) + zs(ddg)
    order = np.lexsort((np.arange(n), -score))
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)

    sites = dict(zip(ann['index'].tolist(), ann['site'].tolist()))
    sources = [i for i in range(n) if sites.get(i) == 'angII_site']
    targets = [i for i in range(n) if sites.get(i) == 'gprotein_interface']
    paths = shortest_paths(g, sources, targets, args.n_paths)

    pocket = np.zeros(n, dtype=bool)
    for idx, val in zip(ann['index'].tolist(), ann['is_pocket'].tolist()):
        pocket[int(idx)] = bool(val)
    top = [int(i) for i in order[:args.top_hotspots]]
    k = int(pocket[top].sum())
    kk = int(pocket.sum())
    pval = float(hypergeom.sf(k - 1, n, kk, args.top_hotspots))

    labels = ann.set_index('index')['residue_id'].to_dict()
    res = pd.DataFrame({'index': np.arange(n)})
    res['residue_id'] = [labels[i] for i in range(n)]
    res['weighted_degree'] = wdeg
    res['mut_ddg'] = ddg
    res['hotspot_score'] = score
    res['community'] = [int(assign[i]) for i in range(n)]
    res['hotspot_rank'] = rank
    res.to_csv(os.path.join(args.outdir, 'residue_scores.tsv'), sep='\t', index=False)

    np.save(os.path.join(args.outdir, 'coupling_matrix.npy'), np.asarray(p, dtype=np.float32))

    with open(os.path.join(args.outdir, 'network_edges.tsv'), 'w') as fh:
        fh.write('residue_i\tresidue_j\tpartial_corr\tabs_partial_corr\n')
        for a, b, val in zip(ei, ej, ep):
            fh.write('%d\t%d\t%.8f\t%.8f\n' % (int(a), int(b), float(val), abs(float(val))))

    with open(os.path.join(args.outdir, 'hotspots.tsv'), 'w') as fh:
        fh.write('rank\tindex\tresidue_id\thotspot_score\tcommunity\n')
        for r, idx in enumerate(top, start=1):
            fh.write('%d\t%d\t%s\t%.6f\t%d\n' % (r, idx, labels[idx], float(score[idx]), int(assign[idx])))

    with open(os.path.join(args.outdir, 'pathways.tsv'), 'w') as fh:
        fh.write('pathway_id\tstart_residue\tend_residue\tn_hops\ttotal_cost\tpath_residues\n')
        for pid, item in enumerate(paths, start=1):
            chain = ';'.join(labels[i] for i in item['path'])
            fh.write('%d\t%s\t%s\t%d\t%.6f\t%s\n' % (pid, labels[item['start']], labels[item['end']],
                                                       len(item['path']) - 1, item['cost'], chain))

    active_ids = sorted(set(int(assign[i]) for i in range(n) if assign[i] >= 0))
    summary = dict(n_frames=int(energies.shape[0]), n_residues=int(n),
                   threshold=float(args.threshold), min_separation=int(args.min_separation),
                   n_edges=int(len(ei)), n_communities=int(len(active_ids)),
                   n_pocket_in_top25=int(k), pocket_enrichment_p=float(pval),
                   top_hotspot_residue_id=str(labels[top[0]]))
    with open(os.path.join(args.outdir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)


if __name__ == '__main__':
    main()
PYEOF
python /app/analyze.py --energies /app/data/at1r_energies.npy --mutations /app/data/at1r_mutscan.tsv --annotation /app/data/at1r_annotation.tsv --outdir /app/output
ls -l /app/output
