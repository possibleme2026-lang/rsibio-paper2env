#!/bin/bash
set -euo pipefail
python - <<'PY'
import csv, json, math, numpy as np
scaling = {}
series = {}
with open('/app/environment/data/chain_scaling.csv') as f:
    for row in csv.DictReader(f):
        scaling.setdefault(row['topology'], []).append((float(row['N']), float(row['Rg_mean'])))
with open('/app/environment/data/rg_timeseries.csv') as f:
    for row in csv.DictReader(f):
        series.setdefault(row['topology'], []).append(float(row['Rg']))

def slope(pairs):
    x = np.array([a for a, b in pairs])
    y = np.array([b for a, b in pairs])
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])

def rel(arr):
    x = np.array(arr)
    return float(x.std() / x.mean())

def tau(arr):
    x = np.array(arr)
    y = x - x.mean()
    den = float((y * y).sum())
    thr = math.exp(-1.0)
    if den == 0.0:
        return float(len(y) - 1)
    for k in range(1, len(y)):
        rho = float((y[:-k] * y[k:]).sum() / den)
        if rho <= thr:
            return float(k)
    return float(len(y) - 1)

tc = tau(series['circular'])
tl = tau(series['linear'])
out = {
    'nu_circular': slope(scaling['circular']),
    'nu_linear': slope(scaling['linear']),
    'rel_fluct_circular': rel(series['circular']),
    'rel_fluct_linear': rel(series['linear']),
    'tau_circular': tc,
    'tau_linear': tl,
    'tau_ratio': tl / max(tc, 1e-9),
}
with open('/app/output.json', 'w') as f:
    json.dump(out, f)
PY
