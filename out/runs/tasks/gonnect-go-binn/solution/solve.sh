#!/bin/bash
set -euo pipefail
python - <<'PY'
import json, csv
expr='/app/data/expression.csv'
go='/app/data/go_terms.csv'
with open(expr, newline='') as f:
    rows=list(csv.DictReader(f))
samples=[r['sample_id'] for r in rows]
with open(go, newline='') as f:
    golist=list(csv.DictReader(f))
terms=[]
term_genes={}
for r in golist:
    t=r['term_id']; g=r['gene_id']
    if t not in term_genes:
        terms.append(t); term_genes[t]=[]
    term_genes[t].append(g)
acts=[]
for r in rows:
    vals=[]
    for t in terms:
        vals.append(sum(float(r[g]) for g in term_genes[t])/len(term_genes[t]))
    acts.append(vals)
with open('/app/output.json','w') as f:
    json.dump({'samples':samples,'terms':terms,'activations':acts}, f)
PY
