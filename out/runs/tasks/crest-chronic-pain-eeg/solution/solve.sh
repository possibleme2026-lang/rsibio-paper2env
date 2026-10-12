#!/bin/bash
set -euo pipefail
python - <<'PY'
import csv, json, math
model=json.load(open('/app/data/crest_model.json'))
bands=model['bands']
bw=model['band_weights']
rw=model['region_weights']
gw=model['graph_weight']
def sig(x):
    x=max(-50.0, min(50.0, x))
    return 1.0/(1.0+math.exp(-x))
with open('/app/data/spectrograms.csv', newline='') as f, open('/app/output/predictions.csv', 'w', newline='') as o:
    r=csv.DictReader(f)
    w=csv.DictWriter(o, fieldnames=['subject_id','pain_probability'])
    w.writeheader()
    for row in r:
        rs=[]
        for i in range(len(rw)):
            val=sum(bw[b]*float(row['r%d_%s' % (i, bands[b])]) for b in range(len(bands))) + rw[i]
            rs.append(val)
        gs=[(rs[i]+rs[(i-1)%len(rs)]+rs[(i+1)%len(rs)])/3.0 for i in range(len(rs))]
        logit=(1-gw)*sum(rs)/len(rs)+gw*sum(gs)/len(gs)
        w.writerow({'subject_id': row['subject_id'], 'pain_probability': round(sig(logit), 6)})
PY
