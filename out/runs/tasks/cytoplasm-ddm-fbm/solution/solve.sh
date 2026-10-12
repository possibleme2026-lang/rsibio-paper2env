#!/bin/bash
set -euo pipefail
python - <<'PY'
import csv, json, math
from collections import defaultdict
import numpy as np
rows = defaultdict(list)
with open('/app/data/ddm_scaling.csv') as f:
    for r in csv.DictReader(f):
        rows[r['sample_id']].append((float(r['q']), float(r['t']), float(r['E'])))
results = []
for sid, data in sorted(rows.items()):
    X = np.array([[1.0, math.log(t), math.log(q)] for q, t, E in data])
    y = np.array([math.log(E - 1.0) for q, t, E in data])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(coef[1])
    D_f = float(math.exp(coef[0]))
    results.append({'id': sid, 'D_f': D_f, 'alpha': alpha})
with open('/app/output.json', 'w') as f:
    json.dump({'samples': results}, f)
PY
