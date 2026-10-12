#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import csv, json, os
from collections import Counter, defaultdict

DATA = '/app/data'
OUT = '/app/output'
KEYS = ['cell_id','donor_id','tissue_ontology_term_id','tissue','cell_type_ontology_term_id',
        'cell_type','assay_ontology_term_id','assay','disease_ontology_term_id','disease',
        'organism_ontology_term_id','organism','sex','development_stage_ontology_term_id',
        'development_stage','n_genes_detected']
TABLES = ['cl','uberon','efo','ncbitaxon','mondo','pato','hsapdv','mmusdv']

def norm(s):
    return ' '.join(s.strip().split()).lower()

def tsv(path):
    with open(path, newline='') as fh:
        return [r for r in csv.reader(fh, delimiter='\t') if r]

lut, id2label = {}, {}
for name in TABLES:
    rows = tsv(os.path.join(DATA, 'ontology', name + '.tsv'))
    lut[name], id2label[name] = {}, {}
    for tid, label, syns in rows[1:]:
        id2label[name][tid] = label
        for v in [tid, label] + (syns.split('|') if syns else []):
            k = norm(v)
            if k and k not in lut[name]:
                lut[name][k] = (tid, label)

meta = tsv(os.path.join(DATA, 'submission', 'cell_metadata.tsv'))
head = meta[0]
rows = [dict(zip(head, r)) for r in meta[1:]]
genes = tsv(os.path.join(DATA, 'submission', 'genes.tsv'))
grows = [dict(zip(genes[0], r)) for r in genes[1:]]

entries = []
dims = None
with open(os.path.join(DATA, 'submission', 'counts.mtx')) as fh:
    for line in fh:
        if line.startswith('%'):
            continue
        p = line.split()
        if not p:
            continue
        if dims is None:
            dims = [int(x) for x in p]
        else:
            entries.append((int(p[0]), int(p[1]), int(p[2])))
detected = Counter()
for r, c, v in entries:
    if v:
        detected[r] += 1

col_gene = [None] + [g['gene_id'].split('.')[0] for g in grows]
SEX = {'female': ['female','f','woman'], 'male': ['male','m','man'], 'mixed': ['mixed']}
HLT = {'healthy': ['healthy','healthy donor','healthy control','normal','control','none','unaffected'],
       'diseased': ['diseased','disease','sick','patient','case','affected']}
def enum(t, table):
    k = norm(t)
    for key, vals in table.items():
        if k in vals:
            return key
    return 'unknown'
def maj(vals):
    c = Counter(vals)
    b = max(c.values())
    return sorted(v for v, n in c.items() if n == b)[0]

kept, seen = [], set()
n_dup = n_unres = n_low = 0
for idx, r in enumerate(rows, 1):
    cid = r['cell_id']
    if cid in seen:
        n_dup += 1; continue
    seen.add(cid)
    ct = lut['cl'].get(norm(r['cell_type_free_text']))
    ti = lut['uberon'].get(norm(r['tissue_free_text']))
    asy = lut['efo'].get(norm(r['assay_free_text']))
    org = lut['ncbitaxon'].get(norm(r['organism_free_text']))
    if not (ct and ti and asy and org):
        n_unres += 1; continue
    if detected.get(idx, 0) < 100:
        n_low += 1; continue
    kept.append(dict(row=idx, cid=cid, donor=r['donor_id'], ct=ct, ti=ti, asy=asy, org=org,
                     sex=enum(r['donor_sex_free_text'], SEX),
                     dev=r['development_stage_free_text'],
                     health=enum(r['donor_health_status_free_text'], HLT),
                     dis=r['donor_disease_free_text'], ng=detected.get(idx, 0)))

by_donor = defaultdict(list)
for k in kept:
    by_donor[k['donor']].append(k)
dvals = {}
for donor, ks in by_donor.items():
    org = maj([k['org'][0] for k in ks])
    sex = maj([k['sex'] for k in ks])
    dl = lut['hsapdv'] if org == 'NCBITaxon:9606' else (lut['mmusdv'] if org == 'NCBITaxon:10090' else None)
    devs = []
    for k in ks:
        d = dl.get(norm(k['dev'])) if dl else None
        devs.append(d[0] if d else 'unknown')
    health = maj([k['health'] for k in ks])
    if health == 'diseased':
        dis = maj([(lut['mondo'].get(norm(k['dis'])) or ('unknown',))[0] for k in ks])
    else:
        dis = 'PATO:0000461'
    dvals[donor] = (org, sex, maj(devs), health, dis)

obs = []
for k in kept:
    org, sex, dev, health, dis = dvals[k['donor']]
    dis = dis if health == 'diseased' else 'PATO:0000461'
    dl = id2label['hsapdv'].get(dev) or id2label['mmusdv'].get(dev) or 'unknown'
    disl = id2label['pato'].get(dis) or id2label['mondo'].get(dis) or 'unknown'
    obs.append({**k, 'row_dict': None, 'dev_id': dev, 'dev_label': dl, 'dis_id': dis,
                'dis_label': disl, 'org_id': org, 'org_label': id2label['ncbitaxon'][org],
                'sex': sex})
obs.sort(key=lambda d: d['cid'])

n = 0
with open(os.path.join(OUT, 'obs.tsv'), 'w') as fh:
    fh.write('\t'.join(KEYS) + '\n')
    for o in obs:
        fh.write('\t'.join([o['cid'], o['donor'], o['ti'][0], o['ti'][1], o['ct'][0], o['ct'][1],
                            o['asy'][0], o['asy'][1], o['dis_id'], o['dis_label'], o['org_id'],
                            o['org_label'], o['sex'], o['dev_id'], o['dev_label'], str(o['ng'])]) + '\n')
        n += 1

first = {}
for g in grows:
    gid = g['gene_id'].split('.')[0]
    if gid not in first:
        first[gid] = g['gene_name']
keeprows = set(k['row'] for k in kept)
totals = defaultdict(int)
for r, c, v in entries:
    if v and r in keeprows:
        gid = col_gene[c]
        if gid:
            totals[gid] += v
filt = 0
with open(os.path.join(OUT, 'var.tsv'), 'w') as fh:
    fh.write('gene_id\tgene_name\tfeature_is_filtered\n')
    for gid in sorted(first):
        flag = 'TRUE' if totals.get(gid, 0) == 0 else 'FALSE'
        if flag == 'TRUE':
            filt += 1
        fh.write('%s\t%s\t%s\n' % (gid, first[gid], flag))

report = {'n_input_rows': len(rows), 'n_duplicate_rows_removed': n_dup,
          'n_dropped_unresolved_ontology': n_unres, 'n_dropped_low_complexity': n_low,
          'n_retained_cells': n, 'n_retained_donors': len(by_donor),
          'n_genes_input_rows': len(grows), 'n_genes_output_rows': len(first),
          'n_genes_filtered': filt}
with open(os.path.join(OUT, 'report.json'), 'w') as fh:
    json.dump(report, fh, indent=2)
print(json.dumps(report, indent=2))
PY
