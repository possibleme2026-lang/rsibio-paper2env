#!/usr/bin/env python3
# Deep grader: recomputes the full AT1R allosteric pipeline, checks every
# constraint with partial credit, and re-runs the agent's script on a held-out ensemble.
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import hypergeom
from sklearn.covariance import ledoit_wolf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gold_gen import make_dataset, write_dataset, PUBLIC_SEED

HIDDEN_SEED = 90210773
PUBLIC_OUT = '/app/output'
PUBLIC_DATA = '/app/data'
HIDDEN_DATA = '/tmp/hidden_data'
HIDDEN_OUT = '/tmp/hidden_out'
REWARD = '/logs/verifier/reward.json'
DETAILS = '/logs/verifier/details.json'
THRESHOLD = 0.20
MIN_SEP = 4
TOPK = 25
NPATHS = 3


# ------------------------------------------------------------------ reference
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


def build_network(p, threshold=THRESHOLD, min_sep=MIN_SEP):
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
    mod = float(nx.algorithms.community.modularity(sub, parts, weight='weight')) if parts else 0.0
    return assign, parts, mod


def shortest_paths(g, sources, targets, n_paths=NPATHS):
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


def reference(energies, mut, ann):
    p = coupling_matrix(energies)
    n = p.shape[0]
    ei, ej, ep, wdeg, g = build_network(p)
    assign, parts, mod = communities(g)
    ddg = mut.sort_values('index')['mean_ddg'].to_numpy(dtype=float)

    def zs(v):
        return (v - v.mean()) / (v.std(ddof=0) + 1e-12)

    score = zs(wdeg) + zs(ddg)
    order = np.lexsort((np.arange(n), -score))
    sites = dict(zip(ann['index'].tolist(), ann['site'].tolist()))
    sources = [i for i in range(n) if sites.get(i) == 'angII_site']
    targets = [i for i in range(n) if sites.get(i) == 'gprotein_interface']
    paths = shortest_paths(g, sources, targets)
    top = [int(i) for i in order[:TOPK]]
    pocket = np.zeros(n, dtype=bool)
    for idx, val in zip(ann['index'].tolist(), ann['is_pocket'].tolist()):
        pocket[int(idx)] = bool(val)
    k = int(pocket[top].sum())
    kk = int(pocket.sum())
    pval = float(hypergeom.sf(k - 1, n, kk, TOPK))
    active = np.array([i for i in range(n) if g.degree(i) > 0], dtype=int)
    isolated = np.array([i for i in range(n) if g.degree(i) == 0], dtype=int)
    return dict(P=p, edges=set(zip(ei.tolist(), ej.tolist())), wdeg=wdeg, score=score,
                ddg=ddg, comm=assign, modularity=mod, graph=g, active=active,
                isolated=isolated, top25=top, n=n, n_edges=int(len(ei)),
                n_comm=int(len(set(assign[active].tolist()))) if len(active) else 0,
                k_pocket=k, pvalue=pval, paths=paths, sites=sites,
                labels=ann.set_index('index')['residue_id'].to_dict())


# --------------------------------------------------------------------- loaders
def load_outputs(outdir):
    out = {}
    try:
        out['P'] = np.load(os.path.join(outdir, 'coupling_matrix.npy'))
    except Exception:
        out['P'] = None
    for key, fname in (('edges', 'network_edges.tsv'), ('residue_scores', 'residue_scores.tsv'),
                       ('hotspots', 'hotspots.tsv'), ('pathways', 'pathways.tsv')):
        try:
            out[key] = pd.read_csv(os.path.join(outdir, fname), sep='\t')
        except Exception:
            out[key] = None
    try:
        with open(os.path.join(outdir, 'summary.json')) as fh:
            out['summary'] = json.load(fh)
    except Exception:
        out['summary'] = None
    return out


# ---------------------------------------------------------------------- checks
def check_coupling(out, ref):
    p = out.get('P')
    if p is None:
        return 0.0, 'coupling_matrix.npy missing or unreadable'
    p = np.asarray(p, dtype=np.float64)
    if p.shape != ref['P'].shape:
        return 0.0, 'shape %s' % (p.shape,)
    if not np.allclose(p, p.T, atol=1e-5):
        return 0.0, 'matrix not symmetric'
    if not np.allclose(np.diag(p), 1.0, atol=1e-5):
        return 0.0, 'diagonal is not 1'
    dev = float(np.max(np.abs(p - ref['P'])))
    return float(max(0.0, min(1.0, 1.0 - dev / 0.05))), 'max abs deviation %.3e' % dev


def check_edges(out, ref):
    df = out.get('edges')
    if df is None:
        return 0.0, 'network_edges.tsv missing or unreadable'
    need = ['residue_i', 'residue_j', 'partial_corr', 'abs_partial_corr']
    if list(df.columns) != need:
        return 0.0, 'bad header %s' % (list(df.columns),)
    if len(df) == 0:
        return 0.0, 'empty edge list'
    i = df['residue_i'].to_numpy(dtype=int)
    j = df['residue_j'].to_numpy(dtype=int)
    p = df['partial_corr'].to_numpy(dtype=float)
    ap = df['abs_partial_corr'].to_numpy(dtype=float)
    if not np.all(i < j):
        return 0.0, 'rows must satisfy residue_i < residue_j'
    if len(set(zip(i.tolist(), j.tolist()))) != len(df):
        return 0.0, 'duplicate edges'
    if not np.all(np.abs(p) >= THRESHOLD - 1e-9):
        return 0.0, 'edge below |partial_corr| >= 0.20'
    if not np.all((j - i) >= MIN_SEP):
        return 0.0, 'sequence separation j-i >= 4 violated'
    if not np.allclose(np.abs(p), ap, atol=1e-5):
        return 0.0, 'abs_partial_corr inconsistent with partial_corr'
    got = set(zip(i.tolist(), j.tolist()))
    inter = len(got & ref['edges'])
    union = len(got | ref['edges'])
    jac = inter / max(union, 1)
    sc = max(0.0, min(1.0, (jac - 0.5) / 0.35))
    return float(sc), 'jaccard %.3f (%d agent edges vs %d reference)' % (jac, len(got), len(ref['edges']))


def check_scores(out, ref):
    df = out.get('residue_scores')
    if df is None:
        return 0.0, 'residue_scores.tsv missing or unreadable'
    need = ['index', 'residue_id', 'weighted_degree', 'mut_ddg', 'hotspot_score',
            'community', 'hotspot_rank']
    if list(df.columns) != need:
        return 0.0, 'bad header %s' % (list(df.columns),)
    n = ref['n']
    if len(df) != n:
        return 0.0, 'expected %d rows, got %d' % (n, len(df))
    idx = df['index'].to_numpy(dtype=int)
    if not np.array_equal(idx, np.arange(n)):
        return 0.0, 'index column must be 0..%d in ascending order' % (n - 1)
    wd = df['weighted_degree'].to_numpy(dtype=float)
    hs = df['hotspot_score'].to_numpy(dtype=float)
    dd = df['mut_ddg'].to_numpy(dtype=float)
    dw = float(np.max(np.abs(wd - ref['wdeg'])))
    ds = float(np.max(np.abs(hs - ref['score'])))
    dq = float(np.max(np.abs(dd - ref['ddg'])))
    rk = df['hotspot_rank'].to_numpy(dtype=int)
    order = np.lexsort((np.arange(n), -hs))
    expected = np.empty(n, dtype=int)
    expected[order] = np.arange(1, n + 1)
    rank_ok = bool(np.array_equal(rk, expected))
    sc = (0.4 * max(0.0, 1.0 - dw / 0.05) + 0.4 * max(0.0, 1.0 - ds / 0.05)
          + 0.2 * (1.0 if rank_ok else 0.0))
    return float(sc), 'wdeg dev %.1e, score dev %.1e, ddg dev %.1e, ranks_ok=%s' % (dw, ds, dq, rank_ok)


def check_communities(out, ref):
    df = out.get('residue_scores')
    if df is None or 'community' not in df.columns:
        return 0.0, 'community column missing'
    comm = df['community'].to_numpy(dtype=int)
    if len(comm) != ref['n']:
        return 0.0, 'wrong number of residues'
    active = ref['active']
    isolated = ref['isolated']
    if np.any(comm[isolated] != -1):
        return 0.0, 'isolated residues must have community -1'
    if np.any(comm[active] < 0):
        return 0.0, 'coupled residues must have community >= 0'
    ids = sorted(set(comm[active].tolist()))
    if ids != list(range(len(ids))):
        return 0.0, 'community ids must be contiguous 0..k-1'
    k = len(ids)
    if not (4 <= k <= 40):
        return 0.0, 'n_communities=%d outside [4, 40]' % k
    sub = ref['graph'].subgraph(active.tolist()).copy()
    parts = [set(active[comm[active] == c].tolist()) for c in ids]
    mod = float(nx.algorithms.community.modularity(sub, parts, weight='weight'))
    ratio = mod / max(ref['modularity'], 1e-9)
    return float(max(0.0, min(1.0, ratio / 0.6))), 'modularity %.4f vs reference %.4f' % (mod, ref['modularity'])


def check_hotspots(out, ref):
    df = out.get('hotspots')
    if df is None:
        return 0.0, 'hotspots.tsv missing or unreadable'
    need = ['rank', 'index', 'residue_id', 'hotspot_score', 'community']
    if list(df.columns) != need:
        return 0.0, 'bad header %s' % (list(df.columns),)
    if len(df) != TOPK:
        return 0.0, 'expected %d rows, got %d' % (TOPK, len(df))
    if df['rank'].to_numpy(dtype=int).tolist() != list(range(1, TOPK + 1)):
        return 0.0, 'rank column must be 1..25'
    sc = df['hotspot_score'].to_numpy(dtype=float)
    if np.any(np.diff(sc) > 1e-9):
        return 0.0, 'hotspots must be sorted by descending score'
    got = set(df['index'].to_numpy(dtype=int).tolist())
    refset = set(ref['top25'])
    ov = len(got & refset)
    dev = max(abs(sc[i] - ref['score'][int(idx)]) for i, idx in enumerate(df['index']))
    score = 0.7 * max(0.0, min(1.0, (ov - 15) / 7.0)) + 0.3 * max(0.0, min(1.0, 1.0 - dev / 0.05))
    return float(score), 'overlap %d/25, score dev %.1e' % (ov, dev)


def check_paths(out, ref):
    df = out.get('pathways')
    if df is None:
        return 0.0, 'pathways.tsv missing or unreadable'
    need = ['pathway_id', 'start_residue', 'end_residue', 'n_hops', 'total_cost', 'path_residues']
    if list(df.columns) != need:
        return 0.0, 'bad header %s' % (list(df.columns),)
    refsets = [tuple(p['path']) for p in ref['paths']]
    want = max(1, min(NPATHS, len(refsets)))
    if len(df) != want:
        return 0.0, 'expected %d pathway rows, got %d' % (want, len(df))
    labels = ref['labels']
    id2idx = {v: k for k, v in labels.items()}
    weights = {}
    e = out.get('edges')
    if e is not None and 'residue_i' in e.columns:
        for a, b, val in zip(e['residue_i'].tolist(), e['residue_j'].tolist(), e['partial_corr'].tolist()):
            weights[(int(a), int(b))] = abs(float(val))
            weights[(int(b), int(a))] = abs(float(val))
    valid, matched = 0, 0
    for _, row in df.iterrows():
        try:
            nodes = [id2idx[x] for x in str(row['path_residues']).split(';')]
        except KeyError:
            continue
        if len(nodes) < 2:
            continue
        if ref['sites'].get(nodes[0]) != 'angII_site' or ref['sites'].get(nodes[-1]) != 'gprotein_interface':
            continue
        cost, ok = 0.0, True
        for a, b in zip(nodes[:-1], nodes[1:]):
            w = weights.get((a, b))
            if w is None or w <= 0:
                ok = False
                break
            cost += -np.log(w)
        if not ok:
            continue
        if abs(cost - float(row['total_cost'])) > 1e-3:
            continue
        if int(row['n_hops']) != len(nodes) - 1:
            continue
        valid += 1
        if tuple(nodes) in refsets:
            matched += 1
    sc = 0.5 * (valid / float(want)) + 0.5 * (matched / float(max(1, len(refsets))))
    return float(min(1.0, sc)), 'valid %d/%d, matched %d/%d reference paths' % (valid, want, matched, len(refsets))


def check_summary(out, ref):
    js = out.get('summary')
    if js is None:
        return 0.0, 'summary.json missing or unreadable'
    need = ['n_frames', 'n_residues', 'threshold', 'min_separation', 'n_edges',
            'n_communities', 'n_pocket_in_top25', 'pocket_enrichment_p',
            'top_hotspot_residue_id']
    missing = [k for k in need if k not in js]
    if missing:
        return 0.0, 'missing keys %s' % (missing,)
    parts = []
    parts.append(1.0 if int(js['n_residues']) == ref['n'] else 0.0)
    parts.append(1.0 if int(js['n_edges']) == ref['n_edges'] else 0.0)
    parts.append(1.0 if int(js['n_communities']) == ref['n_comm'] else 0.0)
    parts.append(1.0 if int(js['n_pocket_in_top25']) == ref['k_pocket'] else 0.0)
    try:
        pv = float(js['pocket_enrichment_p'])
        rp = ref['pvalue']
        parts.append(1.0 if abs(pv - rp) <= max(1e-9, 1e-6 * abs(rp)) else 0.0)
    except Exception:
        parts.append(0.0)
    return float(np.mean(parts)), 'summary fields: %s' % (parts,)


def evaluate(outdir, ref):
    out = load_outputs(outdir)
    checks = [('coupling_matrix', check_coupling), ('network_edges', check_edges),
              ('residue_scores', check_scores), ('communities', check_communities),
              ('hotspots', check_hotspots), ('pathways', check_paths),
              ('summary', check_summary)]
    res = {}
    for name, fn in checks:
        try:
            res[name] = fn(out, ref)
        except Exception as exc:
            res[name] = (0.0, 'checker exception: %r' % (exc,))
    return res


# ------------------------------------------------------------------------ main
def main():
    report = {}
    try:
        pe = np.load(os.path.join(PUBLIC_DATA, 'at1r_energies.npy'))
        pm = pd.read_csv(os.path.join(PUBLIC_DATA, 'at1r_mutscan.tsv'), sep='\t')
        pa = pd.read_csv(os.path.join(PUBLIC_DATA, 'at1r_annotation.tsv'), sep='\t')
    except Exception as exc:
        report['public_data_fallback'] = repr(exc)
        pe, pm, pa = make_dataset(PUBLIC_SEED)
    ref_pub = reference(pe, pm, pa)
    pub = evaluate(PUBLIC_OUT, ref_pub)

    he, hm, ha = make_dataset(HIDDEN_SEED)
    write_dataset(HIDDEN_DATA, he, hm, ha)
    ref_hid = reference(he, hm, ha)
    ok = True
    try:
        proc = subprocess.run([sys.executable, '/app/analyze.py',
                               '--energies', os.path.join(HIDDEN_DATA, 'at1r_energies.npy'),
                               '--mutations', os.path.join(HIDDEN_DATA, 'at1r_mutscan.tsv'),
                               '--annotation', os.path.join(HIDDEN_DATA, 'at1r_annotation.tsv'),
                               '--outdir', HIDDEN_OUT],
                              cwd='/app', capture_output=True, timeout=600)
        ok = proc.returncode == 0
        report['heldout_stdout'] = proc.stdout.decode('utf-8', 'replace')[-1500:]
        report['heldout_stderr'] = proc.stderr.decode('utf-8', 'replace')[-1500:]
    except Exception as exc:
        ok = False
        report['heldout_error'] = repr(exc)
    if ok:
        hid = evaluate(HIDDEN_OUT, ref_hid)
    else:
        hid = {k: (0.0, 'held-out re-run failed') for k in pub}

    pub_score = float(np.mean([v[0] for v in pub.values()]))
    hid_score = float(np.mean([v[0] for v in hid.values()]))
    total = float(0.35 * pub_score + 0.65 * hid_score)

    details = {'public_score': pub_score, 'heldout_score': hid_score, 'score': total,
               'public': {k: [v[0], v[1]] for k, v in pub.items()},
               'heldout': {k: [v[0], v[1]] for k, v in hid.items()}}
    details.update(report)
    try:
        os.makedirs(os.path.dirname(REWARD), exist_ok=True)
        with open(REWARD, 'w') as fh:
            json.dump({'score': total, 'reward': total}, fh)
        with open(DETAILS, 'w') as fh:
            json.dump(details, fh, indent=2, default=str)
    except Exception:
        pass
    print(json.dumps({'score': total, 'public_score': pub_score, 'heldout_score': hid_score}, indent=2))
    for name in pub:
        print('public   %-16s %.3f  %s' % (name, pub[name][0], pub[name][1]))
        print('heldout  %-16s %.3f  %s' % (name, hid[name][0], hid[name][1]))


if __name__ == '__main__':
    main()
