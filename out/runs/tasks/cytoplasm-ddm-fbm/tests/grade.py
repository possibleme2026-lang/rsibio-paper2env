import json, math
GOLD = {
    's1': {'D_f': 10.0, 'alpha': 0.50},
    's2': {'D_f': 5.0, 'alpha': 0.30},
    's3': {'D_f': 2.0, 'alpha': 0.70},
    's4': {'D_f': 8.0, 'alpha': 0.40},
    's5': {'D_f': 3.0, 'alpha': 0.60},
}
def main():
    score = 0.0
    try:
        with open('/app/output.json') as f:
            data = json.load(f)
        preds = {}
        for item in data.get('samples', []):
            sid = str(item.get('id', ''))
            preds[sid] = {'D_f': float(item.get('D_f')), 'alpha': float(item.get('alpha'))}
        errs = []
        for sid, g in GOLD.items():
            if sid not in preds:
                raise ValueError('missing sample')
            p = preds[sid]
            d_err = (p['D_f'] - g['D_f']) / g['D_f']
            a_err = p['alpha'] - g['alpha']
            errs.append(d_err * d_err + a_err * a_err)
        rmse = math.sqrt(sum(errs) / len(errs))
        score = max(0.0, 1.0 - rmse)
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
        json.dump({'score': round(float(score), 6)}, f)
if __name__ == '__main__':
    main()
