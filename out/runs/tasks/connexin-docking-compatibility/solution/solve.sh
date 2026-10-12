#!/bin/bash
set -euo pipefail
python - <<'PY'
import csv

train = []
with open('/app/environment/data/training_labels.csv', newline='') as f:
    for row in csv.DictReader(f):
        train.append((float(row['fetch_score']), int(row['label'])))

candidates = sorted({s for s, _ in train}) + [0.0, 1.0]
best_t = 0.0
best_f = -1.0
for t in candidates:
    tp = fp = fn = 0
    for s, y in train:
        p = 1 if s >= t else 0
        if y == 1 and p == 1:
            tp += 1
        elif y == 0 and p == 1:
            fp += 1
        elif y == 1 and p == 0:
            fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    if f1 > best_f:
        best_f = f1
        best_t = t

with open('/app/environment/data/test_pairs.csv', newline='') as f, open('/app/predictions.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['pair_id', 'prediction'])
    for row in csv.DictReader(f):
        writer.writerow([row['pair_id'], 1 if float(row['fetch_score']) >= best_t else 0])
PY
