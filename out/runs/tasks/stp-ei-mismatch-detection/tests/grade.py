import json, math
HEX='AA5555AACC3333CCFF0000FF55AAFF00AA5500FFCC3333CCCC33AA5500FF55AAFF0033CC33CCFF0055AACC33AA5500FFAA5500FFCC33FF0055AA33CC55AACC33FF0033CC00FFAA55FF00AA5555AA00FFCC3333CC00FF33CCAA55CC3355AAFF00CC3300FF33CCFF00AA5555AA33CCAA5500FF55AACC33FF00'
EPT=6
U=0.5
ALPHA=0.5
THRESH=1.5
def stp_scores(h):
    n=len(h)//4
    out=[]
    for t in range(0,n,EPT):
        xe=[1.0]*8
        ue=[0.5]*8
        xi=[1.0]*8
        ui=[0.5]*8
        for k in range(t,min(t+EPT,n)):
            tok=h[k*4:k*4+4]
            ev=int(tok[:2],16)
            iv=int(tok[2:],16)
            es=0.0
            isum=0.0
            for b in range(8):
                if (ev>>b)&1:
                    r=ue[b]*xe[b]
                    xe[b]-=r
                    ue[b]+=U*(1-ue[b])
                    es+=r
                if (iv>>b)&1:
                    r=ui[b]*xi[b]
                    xi[b]-=r
                    ui[b]+=U*(1-ui[b])
                    isum+=r
            for b in range(8):
                xe[b]+=ALPHA*(1-xe[b])
                xi[b]+=ALPHA*(1-xi[b])
            out.append(abs(es-isum))
    return out
def auroc(y,s):
    pos=[a for a,b in zip(s,y) if b==1]
    neg=[a for a,b in zip(s,y) if b==0]
    if not pos or not neg:
        return 0.0
    lt=sum(1 for a in pos for b in neg if a>b)
    tie=sum(1 for a in pos for b in neg if a==b)
    return (lt+0.5*tie)/(len(pos)*len(neg))
try:
    gold=stp_scores(HEX)
    labels=[1 if v>THRESH else 0 for v in gold]
    with open('/app/predictions.json') as f:
        pred=json.load(f)
    scores=pred.get('scores',[])
    if len(scores)!=len(labels):
        raise ValueError
    vals=[float(x) for x in scores]
    if any(math.isnan(x) or math.isinf(x) for x in vals):
        raise ValueError
    score=auroc(labels,vals)
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
with open('/logs/verifier/reward.json','w') as f:
    json.dump({'score':score},f)
