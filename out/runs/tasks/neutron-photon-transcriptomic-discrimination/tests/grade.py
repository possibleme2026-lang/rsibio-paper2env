import csv, json, math, os

GOLD = {'R31': 0, 'R32': 1, 'R33': 0, 'R34': 1, 'R35': 0, 'R36': 1, 'R37': 0, 'R38': 1, 'R39': 0, 'R40': 1, 'R41': 0, 'R42': 1, 'R43': 0, 'R44': 1, 'R45': 1}

def auroc(y, p):
    n = len(y)
    pos = sum(y)
    neg = n - pos
    if pos == 0 or neg == 0:
        return 0.5
    order = sorted(range(n), key=lambda i: p[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and p[order[j + 1]] == p[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(ranks[i] for i in range(n) if y[i] == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)

score = 0.0
path = '/app/predictions.csv'
if os.path.exists(path):
    try:
        preds = {}
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and 'sample_id' in reader.fieldnames and 'prob_neutron' in reader.fieldnames:
                for row in reader:
                    sid = (row.get('sample_id') or '').strip()
                    val = (row.get('prob_neutron') or '').strip()
                    if sid in GOLD:
                        x = float(val)
                        if math.isfinite(x):
                            preds[sid] = x
        if all(sid in preds for sid in GOLD):
            ids = sorted(GOLD)
            y = [GOLD[sid] for sid in ids]
            p = [preds[sid] for sid in ids]
            score = auroc(y, p)
    except Exception:
        import traceback as _tb, sys as _sys
        _err = _tb.format_exc()
        try:
            import os as _os
            _os.makedirs('/logs/verifier', exist_ok=True)
            open('/logs/verifier/grader_error.txt','w').write(_err)
        except Exception:
            pass
        print('GRADER_EXCEPTION\n' + _err, file=_sys.stderr, flush=True)
        score = 0.0
score = max(0.0, min(1.0, score))
with open('/logs/verifier/reward.json', 'w') as f:
    json.dump({'score': round(score, 6)}, f)
