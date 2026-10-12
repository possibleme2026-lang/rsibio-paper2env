import csv, json, math, os
N=125; R=56; NB=5
BW=[0.15,-0.10,0.55,0.05,-0.05]
GW=0.5
def pseudo(x):
    return ((x * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff * 2 - 1
def rw(r):
    return round(((r * 37) % 101 - 50) / 100.0, 4)
def bp(s,r,b):
    v = pseudo(s * 1000003 + r * 10007 + b * 101)
    if b == 2:
        v += 0.6 * pseudo(s * 9973 + 17)
    return round(v, 4)
def logit(s):
    rs=[]
    for r in range(R):
        rs.append(sum(BW[b] * bp(s,r,b) for b in range(NB)) + rw(r))
    gs=[]
    for r in range(R):
        gs.append((rs[r] + rs[(r-1)%R] + rs[(r+1)%R]) / 3.0)
    return (1-GW) * sum(rs)/R + GW * sum(gs)/R
def auroc(scores, labels):
    pos=sum(labels); neg=len(labels)-pos
    if pos==0 or neg==0: return 0.0
    if len(set(scores)) < 2: return 0.0
    pairs=sorted(zip(scores, labels), key=lambda x: x[0])
    ranks=[0.0]*len(pairs)
    i=0; n=len(pairs)
    while i<n:
        j=i
        while j<n and pairs[j][0]==pairs[i][0]: j+=1
        avg=(i+1+j)/2.0
        for k in range(i,j): ranks[k]=avg
        i=j
    sp=sum(ranks[k] for k in range(n) if pairs[k][1]==1)
    return (sp - pos*(pos+1)/2.0)/(pos*neg)
def main():
    L=[logit(i) for i in range(N)]
    m=sum(L)/N
    sd=math.sqrt(sum((x-m)**2 for x in L)/N)
    if sd == 0: sd=1.0
    z=[(x-m)/sd for x in L]
    noisy=[z[i] + 1.8 * pseudo(i * 7919 + 1234567) for i in range(N)]
    order=sorted(range(N), key=lambda i: noisy[i], reverse=True)
    gold={i:1 for i in order[:74]}
    pred={}
    try:
        with open('/app/output/predictions.csv', newline='') as f:
            reader=csv.DictReader(f)
            for row in reader:
                sid=(row.get('subject_id') or '').strip()
                val=row.get('pain_probability')
                if val is None: val=row.get('score')
                if val is None: val=row.get('prediction')
                try:
                    v=float(val)
                    if math.isfinite(v): pred[sid]=v
                except Exception: pass
    except Exception: pass
    if len(pred) < N: score=0.0
    else:
        scores=[]; labels=[]
        for i in range(N):
            sid='sub-%03d' % i
            scores.append(pred.get(sid, 0.0))
            labels.append(gold.get(i, 0))
        score=auroc(scores, labels)
    os.makedirs('/app', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': score}, f)
main()
