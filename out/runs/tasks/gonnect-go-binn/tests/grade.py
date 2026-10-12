import json, csv, math
GOLD='/tests/gold.csv'
PRED='/app/output.json'
REWARD='/logs/verifier/reward.json'

def pearson(x,y):
    n=len(x)
    if n<3:
        return 0.0
    mx=sum(x)/n; my=sum(y)/n
    sx=sum((a-mx)*(a-mx) for a in x)
    sy=sum((b-my)*(b-my) for b in y)
    if sx<=0 or sy<=0:
        return 0.0
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y))
    return sxy/math.sqrt(sx*sy)

def write(score):
    score=max(0.0, min(1.0, float(score)))
    with open(REWARD,'w') as f:
        json.dump({'score': score}, f)

try:
    with open(GOLD, newline='') as f:
        gold_rows=list(csv.DictReader(f))
    with open(PRED) as f:
        pred=json.load(f)
    gold_samples=[r['sample_id'] for r in gold_rows]
    gold_terms=[k for k in gold_rows[0].keys() if k!='sample_id']
    gold={t:{r['sample_id']: float(r[t]) for r in gold_rows} for t in gold_terms}
    samples=pred['samples']; terms=pred['terms']; acts=pred['activations']
    pred_map={t:{s: float(acts[j][i]) for j,s in enumerate(samples)} for i,t in enumerate(terms)}
    corrs=[]
    for t in gold_terms:
        if t not in pred_map:
            corrs.append(0.0); continue
        common=[s for s in gold_samples if s in pred_map[t]]
        x=[pred_map[t][s] for s in common]
        y=[gold[t][s] for s in common]
        corrs.append(pearson(x,y))
    score=sum(corrs)/len(corrs) if corrs else 0.0
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
    score=0.0
write(score)
