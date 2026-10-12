import json
import math
import numpy as np

T = 256
NS = np.array([1000.0, 2000.0, 3000.0, 4000.0, 4969.0])
SEED = 20260831

def ar_series(phi, mean, std, rng):
    x = np.empty(T)
    sig = math.sqrt(max(0.0, 1.0 - phi * phi))
    for i in range(T):
        if i == 0:
            x[i] = rng.normal()
        else:
            x[i] = phi * x[i - 1] + sig * rng.normal()
    x = (x - x.mean()) / x.std() * std + mean
    return x

def gen():
    rng = np.random.default_rng(SEED)
    circ = ar_series(0.05, 90.0, 6.12, rng)
    lin = ar_series(math.exp(-1.0 / 35.0), 100.0, 17.2, rng)
    rc = 0.5 * NS ** 0.73
    rl = 0.5 * NS ** 0.62
    return rc, rl, circ, lin

def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])

def rel(x):
    return float(x.std() / x.mean())

def tau(x):
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

def main():
    rc, rl, c, l = gen()
    tc = tau(c)
    tl = tau(l)
    gold = {
        'nu_circular': slope(NS, rc),
        'nu_linear': slope(NS, rl),
        'rel_fluct_circular': rel(c),
        'rel_fluct_linear': rel(l),
        'tau_circular': tc,
        'tau_linear': tl,
        'tau_ratio': tl / max(tc, 1e-9),
    }
    score = 0.0
    try:
        with open('/app/output.json') as f:
            pred = json.load(f)
        if not all(isinstance(pred.get(k), (int, float)) and not isinstance(pred.get(k), bool) for k in gold):
            raise ValueError('missing or non-numeric metric')
        errs = []
        for k, g in gold.items():
            d = max(1e-9, abs(float(g)))
            errs.append(min(1.0, abs(float(pred.get(k)) - float(g)) / d))
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        score = max(0.0, 1.0 - 5.0 * rmse)
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
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': score}, f)

if __name__ == '__main__':
    main()
