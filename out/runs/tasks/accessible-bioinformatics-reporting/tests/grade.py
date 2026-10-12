import json
GOLD = {'s1','s2','s3','s5','s8'}
try:
    with open('/app/output.json') as f:
        out = json.load(f)
except Exception:
    out = {}
summary = out.get('summary', '') if isinstance(out, dict) else ''
selected = out.get('selected_statement_ids', []) if isinstance(out, dict) else []
if not isinstance(selected, list):
    selected = []
pred = set()
for x in selected:
    if isinstance(x, str):
        pred.add(x)
if not isinstance(summary, str) or len(summary.strip()) == 0:
    score = 0.0
else:
    tp = len(pred & GOLD)
    fp = len(pred - GOLD)
    fn = len(GOLD - pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
with open('/logs/verifier/reward.json', 'w') as f:
    json.dump({'score': score}, f)
