#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import csv
import os

rows = []
with open('/app/environment/data/variants.tsv') as f:
    reader = csv.DictReader(f, delimiter=chr(9))
    for row in reader:
        rows.append(row)

os.makedirs('/app/output', exist_ok=True)
with open('/app/output/predictions.tsv', 'w', newline='') as out:
    writer = csv.writer(out, delimiter=chr(9))
    writer.writerow(['variant_id', 'predicted_class'])
    for row in rows:
        total = float(row['total_depth'])
        minor = float(row['minor_allele_depth'])
        base = float(row['baseline_depth'])
        frac = minor / total if total else 0.0
        if frac > 0.5:
            frac = 1.0 - frac
        ratio = total / base if base else 1.0
        if 0.20 <= frac <= 0.28:
            pred = 'CN4'
        elif 0.30 <= frac <= 0.38 and ratio < 1.2:
            pred = 'CN3'
        elif 0.46 <= frac <= 0.54:
            pred = 'CN4' if ratio >= 1.25 else 'CN2'
        else:
            centroids = {'CN2': (0.5, 1.0), 'CN3': (0.33, 1.0), 'CN4': (0.25, 1.0)}
            pred = min(centroids.items(), key=lambda kv: (kv[1][0] - frac) ** 2 + (kv[1][1] - ratio) ** 2)[0]
        writer.writerow([row['variant_id'], pred])
PY
