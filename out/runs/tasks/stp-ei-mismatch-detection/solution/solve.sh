#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import json
d=json.load(open('/app/environment/data/inputs.json'))
h=d['patterns']; ept=d['events_per_trial']; U=0.5; A=0.5
n=len(h)//4; out=[]
for t in range(0,n,ept):
  xe=[1.0]*8; ue=[0.5]*8; xi=[1.0]*8; ui=[0.5]*8
  for k in range(t,min(t+ept,n)):
    tok=h[k*4:k*4+4]; ev=int(tok[:2],16); iv=int(tok[2:],16)
    es=0.0; isum=0.0
    for b in range(8):
      if (ev>>b)&1:
        r=ue[b]*xe[b]; xe[b]-=r; ue[b]+=U*(1-ue[b]); es+=r
      if (iv>>b)&1:
        r=ui[b]*xi[b]; xi[b]-=r; ui[b]+=U*(1-ui[b]); isum+=r
    for b in range(8):
      xe[b]+=A*(1-xe[b]); xi[b]+=A*(1-xi[b])
    out.append(abs(es-isum))
json.dump({'scores':out}, open('/app/predictions.json','w'))
PY
