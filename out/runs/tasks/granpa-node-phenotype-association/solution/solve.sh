#!/bin/bash
set -euo pipefail
mkdir -p /app/results
cat > /app/solve_granpa.py <<'PYEOF'
import json
import os
from collections import defaultdict

from scipy.stats import hypergeom

DATA = '/app/data'
OUT = '/app/results'


def parse_gfa(path):
    segs, paths, links = [], {}, set()
    for line in open(path):
        f = line.rstrip('\n').split('\t')
        if f[0] == 'S':
            segs.append(f[1])
        elif f[0] == 'P':
            paths[f[1]] = [t[:-1] for t in f[2].split(',') if t]
        elif f[0] == 'L':
            links.add((f[1], f[3]))
    return segs, paths, links


def bh(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        q[i] = min(prev, 1.0)
    return q


segs, paths, links = parse_gfa(os.path.join(DATA, 'graph.gfa'))
pheno = {}
with open(os.path.join(DATA, 'phenotypes.tsv')) as fh:
    fh.readline()
    for line in fh:
        s, ph = line.rstrip('\n').split('\t')
        pheno[s] = ph
cases = sorted(s for s, v in pheno.items() if v == 'case')
ctrls = sorted(s for s, v in pheno.items() if v == 'control')
N = len(cases) + len(ctrls)

carriers = defaultdict(set)
for s, toks in paths.items():
    for t in toks:
        carriers[t].add(s)

info, pvals = {}, []
for seg in segs:
    cs = carriers.get(seg, set())
    a = sum(1 for s in cases if s in cs)
    c = sum(1 for s in ctrls if s in cs)
    ps = a / len(cases) - c / len(ctrls)
    p = min(max(float(hypergeom.sf(a - 1, N, a + c, len(cases))), 0.0), 1.0)
    info[seg] = {'a': a, 'c': c, 'ps': ps, 'p': p}
    pvals.append(p)
qs = bh(pvals)
for seg, q in zip(segs, qs):
    info[seg]['q'] = q
    info[seg]['sig'] = q <= 0.05

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'node_association.tsv'), 'w') as fh:
    fh.write('node_id\tn_case_carriers\tn_control_carriers\tPS\tp_value\tq_value\tsignificant\n')
    for seg in sorted(segs):
        r = info[seg]
        fh.write('%s\t%d\t%d\t%.6f\t%.6e\t%.6e\t%s\n' %
                 (seg, r['a'], r['c'], r['ps'], r['p'], r['q'], str(r['sig'])))

sig = set(s for s in segs if info[s]['sig'])
adj = defaultdict(set)
for u, v in links:
    if u in sig and v in sig:
        adj[u].add(v)
        adj[v].add(u)
seen, comps = set(), []
for n in sorted(sig):
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

regions = [{'nodes': c,
            'score': sum(info[n]['ps'] for n in c) / len(c),
            'min_q': min(info[n]['q'] for n in c)} for c in comps]
regions.sort(key=lambda r: (r['min_q'], r['nodes'][0]))

with open(os.path.join(OUT, 'regions.tsv'), 'w') as fh:
    fh.write('region_id\tn_nodes\tnode_ids\tregion_score\tmin_q_value\n')
    for i, r in enumerate(regions):
        fh.write('R%d\t%d\t%s\t%.6f\t%.6e\n' %
                 (i + 1, len(r['nodes']), ';'.join(r['nodes']), r['score'], r['min_q']))

summary = {
    'n_nodes': len(segs),
    'n_samples': N,
    'n_cases': len(cases),
    'n_controls': len(ctrls),
    'n_significant_nodes': len(sig),
    'n_regions': len(regions),
    'top_region': regions[0]['nodes'] if regions else None,
}
with open(os.path.join(OUT, 'summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary))
PYEOF
python3 /app/solve_granpa.py
