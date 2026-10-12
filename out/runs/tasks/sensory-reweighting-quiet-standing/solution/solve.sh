#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import csv, json
out = []
with open('/app/environment/data/conditions.csv') as f:
    for r in csv.DictReader(f):
        p0 = float(r['pi_prior'])
        pv = float(r['pi_visual'])
        pp = float(r['pi_proprio'])
        pb = float(r['pi_vestibular'])
        bv = float(r['bias_visual'])
        bp = float(r['bias_proprio'])
        bb = float(r['bias_vestibular'])
        gain = float(r['action_gain'])
        d = p0 + pv + pp + pb
        if d <= 0:
            mu = cv = cp = cb = shift = 0.0
        else:
            cv = pv * bv / d
            cp = pp * bp / d
            cb = pb * bb / d
            mu = cv + cp + cb
            shift = gain * mu
        out.append({
            'condition_id': r['condition_id'],
            'belief_bias': mu,
            'contribution_visual': cv,
            'contribution_proprio': cp,
            'contribution_vestibular': cb,
            'postural_shift': shift
        })
with open('/app/output/predictions.json', 'w') as f:
    json.dump(out, f)
PY
