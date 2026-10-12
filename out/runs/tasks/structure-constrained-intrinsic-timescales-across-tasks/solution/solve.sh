#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import csv, json, math
from collections import defaultdict

base = '/app/environment/data'
tau = defaultdict(dict)
with open(base + '/autocorr.csv') as f:
    for row in csv.DictReader(f):
        r = float(row['r1'])
        t = -1.0 / math.log(r) if 0.0 < r < 1.0 else 0.0
        tau[row['condition']][row['region']] = t

dist = {}
with open(base + '/sc_distance.csv') as f:
    for row in csv.DictReader(f):
        dist[row['region']] = float(row['distance'])

def pearson(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)

coupling = {}
for cond in sorted(tau):
    regions = sorted(tau[cond])
    coupling[cond] = pearson([tau[cond][r] for r in regions], [dist[r] for r in regions])

rest = max(coupling, key=lambda k: coupling[k])
labels = {c: ('rest' if c == rest else 'task') for c in tau}
out = {'labels': labels, 'rest_condition': rest, 'coupling': coupling}
with open('/app/output.json', 'w') as f:
    json.dump(out, f, indent=2)
PY
