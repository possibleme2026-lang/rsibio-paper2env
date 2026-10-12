#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import csv
import json
import math

train_path = '/app/environment/data/train.csv'
test_path = '/app/environment/data/test.csv'
out_path = '/app/predictions.json'

with open(train_path, newline='') as f:
    train = list(csv.DictReader(f))

features = [k for k in train[0].keys() if k not in ('participant_id', 'triad_id', 'nonfriend')]
stats = {}
counts = {'0': 0, '1': 0}
for row in train:
    counts[row['nonfriend']] += 1
for cls in ('0', '1'):
    vals = {feat: [] for feat in features}
    for row in train:
        if row['nonfriend'] == cls:
            for feat in features:
                vals[feat].append(float(row[feat]))
    stats[cls] = {}
    for feat in features:
        mean = sum(vals[feat]) / max(1, len(vals[feat]))
        var = sum((x - mean) ** 2 for x in vals[feat]) / max(1, len(vals[feat]))
        stats[cls][feat] = (mean, var + 1e-6)

priors = {cls: counts[cls] / max(1, len(train)) for cls in ('0', '1')}

def logpdf(x, mean, var):
    return -0.5 * math.log(2 * math.pi * var) - ((x - mean) ** 2) / (2 * var)

preds = []
with open(test_path, newline='') as f:
    for row in csv.DictReader(f):
        scores = {}
        for cls in ('0', '1'):
            lp = math.log(max(1e-12, priors[cls]))
            for feat in features:
                mean, var = stats[cls][feat]
                lp += logpdf(float(row[feat]), mean, var)
            scores[cls] = lp
        mx = max(scores['0'], scores['1'])
        p0 = math.exp(scores['0'] - mx)
        p1 = math.exp(scores['1'] - mx)
        prob = p1 / (p0 + p1)
        preds.append({'participant_id': row['participant_id'], 'nonfriend_prob': round(prob, 4)})

with open(out_path, 'w') as f:
    json.dump({'predictions': preds}, f)
PY
