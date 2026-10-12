#!/usr/bin/env python3
'''Verifier for the GRANPA node-phenotype association task.'''

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

from scipy.stats import hypergeom

RES = '/app/results'
GOLD_GEN = '/tests/gold_gen.py'
REWARD_PATH = '/logs/verifier/reward.json'

NODE_HEADER = ['node_id', 'n_case_carriers', 'n_control_carriers', 'PS',
               'p_value', 'q_value', 'significant']
REG_HEADER = ['region_id', 'n_nodes', 'node_ids', 'region_score', 'min_q_value']

WEIGHTS = {
    'table_structure': 0.10,
    'carrier_counts': 0.12,
    'phenotype_score': 0.12,
    'p_values': 0.14,
    'q_values': 0.10,
    'significance': 0.10,
    'regions': 0.17,
    'ordering': 0.05,
    'summary': 0.10,
}


def parse_gfa(path):
    segs, paths, links = [], {}, set()
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line:
                continue
            f = line.split('\t')
            if f[0] == 'S':
                segs.append(f[1])
            elif f[0] == 'P':
                paths[f[1]] = [t[:-1] for t in f[2].split(',') if t]
            elif f[0] == 'L':
                links.add((f[1], f[3]))
    return segs, paths, links


def parse_phenotypes(path):
    pheno = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            line = line.rstrip('\n')
            if not line:
                continue
            f = line.split('\t')
            pheno[f[0]] = f[1]
    return pheno


def bh_qvalues(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = pvals[i] * m / rank
        if val < prev:
            prev = val
        q[i] = min(prev, 1.0)
    return q


def connected_components(links, sig):
    sigset = set(sig)
    adj = defaultdict(set)
    for u, v in links:
        if u in sigset and v in sigset and u != v:
            adj[u].add(v)
            adj[v].add(u)
    seen, comps = set(), []
    for n in sorted(sigset):
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(sorted(comp))
    return comps


def build_gold(data_dir):
    segs, paths, links = parse_gfa(os.path.join(data_dir, 'graph.gfa'))
    pheno = parse_phenotypes(os.path.join(data_dir, 'phenotypes.tsv'))
    cases = sorted([s for s, p in pheno.items() if p == 'case'])
    ctrls = sorted([s for s, p in pheno.items() if p == 'control'])
    N = len(cases) + len(ctrls)

    carriers = defaultdict(set)
    for s, toks in paths.items():
        for t in toks:
            carriers[t].add(s)

    info = {}
    pvals = []
    for seg in segs:
        cs = carriers.get(seg, set())
        a = sum(1 for s in cases if s in cs)
        c = sum(1 for s in ctrls if s in cs)
        ps = a / len(cases) - c / len(ctrls)
        p = float(hypergeom.sf(a - 1, N, a + c, len(cases)))
        p = min(max(p, 0.0), 1.0)
        info[seg] = {'a': a, 'c': c, 'ps': ps, 'p': p}
        pvals.append(p)

    qs = bh_qvalues(pvals)
    for seg, q in zip(segs, qs):
        info[seg]['q'] = q
        info[seg]['sig'] = bool(q <= 0.05)

    sig = [s for s in segs if info[s]['sig']]
    regions = []
    for comp in connected_components(links, sig):
        regions.append({
            'nodes': comp,
            'score': sum(info[n]['ps'] for n in comp) / len(comp),
            'min_q': min(info[n]['q'] for n in comp),
        })
    regions.sort(key=lambda r: (r['min_q'], r['nodes'][0]))
    for i, r in enumerate(regions):
        r['id'] = 'R%d' % (i + 1)

    return {
        'info': info,
        'nodes': segs,
        'regions': regions,
        'n_cases': len(cases),
        'n_ctrl': len(ctrls),
        'n_samples': N,
    }


def read_tsv(path):
    if not os.path.exists(path):
        return None, None
    with open(path) as fh:
        lines = [l.rstrip('\n') for l in fh if l.strip()]
    if not lines:
        return None, []
    return lines[0].split('\t'), [l.split('\t') for l in lines[1:]]


def close(x, y, rel=1e-6, abs_=1e-12):
    try:
        xf, yf = float(x), float(y)
    except Exception:
        return False
    return abs(xf - yf) <= max(abs_, rel * abs(yf))


def as_bool(tok):
    t = str(tok).strip().lower()
    if t in ('true', '1', 'yes'):
        return True
    if t in ('false', '0', 'no'):
        return False
    return None


def main():
    tmp = tempfile.mkdtemp(prefix='granpa_gold_')
    subprocess.run([sys.executable, GOLD_GEN, tmp], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gold = build_gold(tmp)
    ginfo = gold['info']
    gnodes = gold['nodes']

    frac = {k: 0.0 for k in WEIGHTS}

    n_header, n_rows = read_tsv(os.path.join(RES, 'node_association.tsv'))
    agent = {}
    if n_rows:
        for r in n_rows:
            if len(r) == 7:
                agent[r[0]] = r
    if n_header == NODE_HEADER:
        inter = set(agent) & set(gnodes)
        union = set(agent) | set(gnodes)
        frac['table_structure'] = len(inter) / len(union) if union else 0.0

    ok = 0
    for n in gnodes:
        r = agent.get(n)
        if not r:
            continue
        try:
            if int(r[1]) == ginfo[n]['a'] and int(r[2]) == ginfo[n]['c']:
                ok += 1
        except Exception:
            pass
    frac['carrier_counts'] = ok / len(gnodes) if gnodes else 0.0

    ok = 0
    for n in gnodes:
        r = agent.get(n)
        if not r:
            continue
        try:
            if abs(float(r[3]) - ginfo[n]['ps']) <= 2e-6:
                ok += 1
        except Exception:
            pass
    frac['phenotype_score'] = ok / len(gnodes) if gnodes else 0.0

    ok = 0
    for n in gnodes:
        r = agent.get(n)
        if r and close(r[4], ginfo[n]['p'], rel=1e-6, abs_=1e-12):
            ok += 1
    frac['p_values'] = ok / len(gnodes) if gnodes else 0.0

    ok = 0
    for n in gnodes:
        r = agent.get(n)
        if r and close(r[5], ginfo[n]['q'], rel=1e-6, abs_=1e-12):
            ok += 1
    frac['q_values'] = ok / len(gnodes) if gnodes else 0.0

    ok = 0
    for n in gnodes:
        r = agent.get(n)
        if r and as_bool(r[6]) == ginfo[n]['sig']:
            ok += 1
    frac['significance'] = ok / len(gnodes) if gnodes else 0.0

    r_header, r_rows = read_tsv(os.path.join(RES, 'regions.tsv'))
    agent_regions = [r for r in (r_rows or []) if len(r) == 5]
    if r_header == REG_HEADER:
        gold_sets = [frozenset(g['nodes']) for g in gold['regions']]
        agent_sets = [frozenset(r[2].split(';')) for r in agent_regions]
        denom = max(len(gold_sets), len(agent_sets))
        if denom == 0:
            frac['regions'] = 1.0
        else:
            matched, val_ok, used = 0, 0, set()
            for i, gs in enumerate(gold_sets):
                for j, aset in enumerate(agent_sets):
                    if j in used or aset != gs:
                        continue
                    used.add(j)
                    matched += 1
                    gr = gold['regions'][i]
                    r = agent_regions[j]
                    if (close(r[3], gr['score'], rel=0.0, abs_=2e-6)
                            and close(r[4], gr['min_q'], rel=1e-6, abs_=1e-12)):
                        val_ok += 1
                    break
            frac['regions'] = 0.5 * matched / denom + 0.5 * val_ok / denom

    ids = [r[0] for r in (n_rows or []) if len(r) == 7]
    node_order_ok = 0.0
    if ids and len(set(ids)) == len(ids) and ids == sorted(ids) and set(ids) == set(gnodes):
        node_order_ok = 1.0
    reg_order_ok = 0.0
    keys = []
    ok_keys = True
    for r in agent_regions:
        try:
            keys.append((float(r[4]), min(r[2].split(';'))))
        except Exception:
            ok_keys = False
    if keys and ok_keys and keys == sorted(keys):
        reg_order_ok = 1.0
    frac['ordering'] = 0.5 * node_order_ok + 0.5 * reg_order_ok

    s = {}
    spath = os.path.join(RES, 'summary.json')
    if os.path.exists(spath):
        try:
            s = json.load(open(spath))
        except Exception:
            s = {}
    exp = {
        'n_nodes': len(gnodes),
        'n_samples': gold['n_samples'],
        'n_cases': gold['n_cases'],
        'n_controls': gold['n_ctrl'],
        'n_significant_nodes': sum(1 for n in gnodes if ginfo[n]['sig']),
        'n_regions': len(gold['regions']),
    }
    hits = sum(1 for k, v in exp.items() if s.get(k) == v)
    if gold['regions']:
        top_ok = s.get('top_region') == gold['regions'][0]['nodes']
    else:
        top_ok = s.get('top_region') in (None, [])
    frac['summary'] = (hits + (1 if top_ok else 0)) / (len(exp) + 1)

    gold_sig = set(n for n in gnodes if ginfo[n]['sig'])
    pred_sig = set()
    for n, r in agent.items():
        if as_bool(r[6]) is True:
            pred_sig.add(n)
    tp = len(gold_sig & pred_sig)
    fp = len(pred_sig - gold_sig)
    fn = len(gold_sig - pred_sig)
    f1 = (2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0

    score = sum(WEIGHTS[k] * frac[k] for k in WEIGHTS)
    score = round(float(score), 6)
    out = {
        'score': score,
        'reward': score,
        'metric_name': 'f1',
        'metric_value': round(float(f1), 6),
        'n_gold_significant': len(gold_sig),
        'n_pred_significant': len(pred_sig),
        'checks': {k: round(float(frac[k]), 6) for k in frac},
    }
    os.makedirs('/logs/verifier', exist_ok=True)
    with open(REWARD_PATH, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        os.makedirs('/logs/verifier', exist_ok=True)
        with open(REWARD_PATH, 'w') as fh:
            json.dump({'score': 0.0, 'reward': 0.0, 'error': str(exc)}, fh)
        raise
