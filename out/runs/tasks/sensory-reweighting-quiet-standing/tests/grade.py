import json, math, os

def zero():
    os.makedirs('/app', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': 0.0}, f)

try:
    input_rows = [
        ('C01', 1.0, 0.8, 1.2, 0.5, 0.02, -0.01, 0.005, 1.0),
        ('C02', 1.0, 0.1, 1.2, 0.5, 0.02, -0.01, 0.005, 1.0),
        ('C03', 1.0, 0.8, 0.1, 0.5, 0.02, -0.01, 0.005, 1.0),
        ('C04', 1.0, 0.8, 1.2, 0.05, 0.02, -0.01, 0.005, 1.0),
        ('C05', 1.0, 0.4, 0.6, 0.25, 0.02, -0.01, 0.005, 1.0),
        ('C06', 1.0, 0.8, 1.2, 0.5, 0.04, -0.02, 0.01, 1.5),
        ('C07', 1.0, 0.2, 1.5, 1.0, -0.03, 0.02, -0.01, 0.8),
        ('C08', 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 1.2)
    ]
    gold = {}
    fields = ['belief_bias', 'contribution_visual', 'contribution_proprio', 'contribution_vestibular', 'postural_shift']
    for cid, p0, pv, pp, pb, bv, bp, bb, gain in input_rows:
        d = p0 + pv + pp + pb
        if d <= 0:
            mu = cv = cp = cb = shift = 0.0
        else:
            cv = pv * bv / d
            cp = pp * bp / d
            cb = pb * bb / d
            mu = cv + cp + cb
            shift = gain * mu
        gold[cid] = {
            'belief_bias': mu,
            'contribution_visual': cv,
            'contribution_proprio': cp,
            'contribution_vestibular': cb,
            'postural_shift': shift
        }
    if not os.path.exists('/app/output/predictions.json'):
        zero()
        raise SystemExit
    with open('/app/output/predictions.json', 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get('predictions', data.get('rows', list(data.values())))
    if not isinstance(data, list):
        zero()
        raise SystemExit
    preds = {}
    for item in data:
        if isinstance(item, dict):
            preds[str(item.get('condition_id', ''))] = item
    err = 0.0
    n = 0
    for cid, g in gold.items():
        p = preds.get(cid)
        if not p:
            zero()
            raise SystemExit
        for k in fields:
            try:
                v = float(p[k])
            except Exception:
                zero()
                raise SystemExit
            if not math.isfinite(v):
                zero()
                raise SystemExit
            err += (v - g[k]) ** 2
            n += 1
    if n == 0:
        zero()
        raise SystemExit
    rmse = math.sqrt(err / n)
    score = max(0.0, 1.0 - rmse / 0.005)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': float(score)}, f)
except SystemExit:
    pass
except Exception:
    zero()
