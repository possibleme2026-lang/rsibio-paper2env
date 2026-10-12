import csv
import json

GOLD = {
    'Cx30_Cx31': 1,
    'Cx30_Cx32': 1,
    'Cx30_Cx33': 1,
    'Cx30_Cx34': 1,
    'Cx31_Cx32': 0,
    'Cx31_Cx33': 0,
    'Cx31_Cx34': 0,
    'Cx32_Cx33': 0,
    'Cx32_Cx34': 0,
    'Cx33_Cx34': 0,
}

pred = {}
try:
    with open('/app/predictions.csv', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            pid = (row.get('pair_id') or '').strip()
            val = (row.get('prediction') or '').strip()
            if pid in GOLD:
                pred[pid] = 1 if val in ('1', 'True', 'true', 'compatible', '1.0') else 0
except Exception:
    pred = {}

tp = fp = fn = 0
for pid, y in GOLD.items():
    p = pred.get(pid, 0)
    if y == 1 and p == 1:
        tp += 1
    elif y == 0 and p == 1:
        fp += 1
    elif y == 1 and p == 0:
        fn += 1

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

with open('/logs/verifier/reward.json', 'w') as f:
    json.dump({'score': round(f1, 6)}, f)
