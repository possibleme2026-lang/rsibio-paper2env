#!/usr/bin/env python3
'''Deterministic generator + hidden gold computation.

The generator body below is a byte-for-byte copy of the task-side
gen_inputs.py (same SEED, same call order), so the grader can rebuild the
submission from a protected seed even if the agent edits /app/data.
'''
import argparse
import os
import random
from collections import Counter, defaultdict

SEED = 20260602
N_DONORS = 60
CELLS_PER_DONOR = (14, 28)
N_GENES = 520
N_ZERO_GENES = 18
N_DUPLICATE_GENES = 12
N_DUPLICATE_CELLS = 18
N_LOW_COMPLEXITY = 45
DENSITY = 0.40

GARBAGE = ['', 'NA', 'N/A', 'not collected', 'unknown', 'not available', '.']
GARBAGE_DEV = ['unknown', 'not collected', 'NA', '']

CL = [
    ('CL:0000084', 'T cell', ['T lymphocyte', 'T-cell']),
    ('CL:0000624', 'CD4-positive, alpha-beta T cell', ['CD4 T cell', 'CD4+ T cell']),
    ('CL:0000625', 'CD8-positive, alpha-beta T cell', ['CD8 T cell', 'CD8+ T cell']),
    ('CL:0000236', 'B cell', ['B lymphocyte', 'B-cell']),
    ('CL:0000623', 'natural killer cell', ['NK cell', 'NK lymphocyte']),
    ('CL:0000576', 'monocyte', ['mononuclear phagocyte']),
    ('CL:0000235', 'macrophage', ['macrophage cell']),
    ('CL:0000451', 'dendritic cell', ['DC', 'dendritic leukocyte']),
    ('CL:0000115', 'endothelial cell', ['endothelium cell']),
    ('CL:0000057', 'fibroblast', ['fibroblast cell']),
    ('CL:0000182', 'hepatocyte', ['liver parenchymal cell']),
    ('CL:0000066', 'epithelial cell', ['epithelium cell']),
    ('CL:0000192', 'smooth muscle cell', ['SMC']),
    ('CL:0000746', 'cardiac muscle cell', ['cardiomyocyte']),
    ('CL:0000187', 'skeletal muscle cell', ['skeletal myocyte']),
    ('CL:0000540', 'neuron', ['nerve cell']),
    ('CL:0000127', 'astrocyte', ['astroglia']),
    ('CL:0000128', 'oligodendrocyte', ['oligodendroglia']),
    ('CL:0000129', 'microglial cell', ['microglia']),
    ('CL:0000136', 'adipocyte', ['fat cell']),
    ('CL:0000312', 'keratinocyte', ['skin keratinocyte']),
    ('CL:0000775', 'neutrophil', ['neutrophilic granulocyte']),
    ('CL:0000037', 'hematopoietic stem cell', ['HSC']),
    ('CL:0000653', 'podocyte', ['glomerular podocyte']),
    ('CL:0002064', 'pancreatic acinar cell', ['acinar cell of pancreas']),
    ('CL:0000097', 'mast cell', ['mastocyte']),
]

UBERON = [
    ('UBERON:0002048', 'lung', ['pulmonary tissue']),
    ('UBERON:0002107', 'liver', ['hepatic tissue']),
    ('UBERON:0000948', 'heart', ['cardiac tissue']),
    ('UBERON:0000955', 'brain', ['brain tissue']),
    ('UBERON:0002113', 'kidney', ['renal tissue']),
    ('UBERON:0002097', 'skin of body', ['skin']),
    ('UBERON:0001134', 'skeletal muscle tissue', ['skeletal muscle']),
    ('UBERON:0000178', 'blood', ['peripheral blood', 'whole blood']),
    ('UBERON:0002371', 'bone marrow', ['marrow']),
    ('UBERON:0001155', 'colon', ['large intestine']),
    ('UBERON:0001264', 'pancreas', ['pancreatic tissue']),
    ('UBERON:0001013', 'adipose tissue', ['fat tissue']),
    ('UBERON:0002106', 'spleen', ['splenic tissue']),
    ('UBERON:0000029', 'lymph node', ['lymphatic node']),
    ('UBERON:0000966', 'retina', ['retinal tissue']),
    ('UBERON:0001987', 'placenta', ['placental tissue']),
    ('UBERON:0000945', 'stomach', ['gastric tissue']),
    ('UBERON:0002046', 'thyroid gland', ['thyroid']),
    ('UBERON:0000473', 'testis', ['testicular tissue']),
    ('UBERON:0000992', 'ovary', ['ovarian tissue']),
    ('UBERON:0001043', 'esophagus', ['oesophagus']),
    ('UBERON:0001295', 'endometrium', ['uterine endometrium']),
]

EFO = [
    ("EFO:0009899", "10x 3' v2", ["10X Chromium 3' v2"]),
    ("EFO:0009922", "10x 3' v3", ["10X Chromium 3' v3"]),
    ("EFO:0009920", "10x 3' v1", ["10X Chromium 3' v1"]),
    ("EFO:0009900", "10x 5' v2", ["10X Chromium 5' v2"]),
    ("EFO:0009901", "10x 5' v1", ["10X Chromium 5' v1"]),
    ('EFO:0011025', '10x multiome', ['10x Genomics Multiome']),
    ('EFO:0008931', 'Smart-seq2', ['SmartSeq2', 'Smart-seq 2']),
    ('EFO:0008953', 'CEL-seq2', ['CELSeq2']),
    ('EFO:0008440', 'Drop-seq', ['DropSeq']),
    ('EFO:0700003', 'Seq-Well', ['SeqWell']),
]

NCBITAXON = [
    ('NCBITaxon:9606', 'Homo sapiens', ['human', 'H. sapiens']),
    ('NCBITaxon:10090', 'Mus musculus', ['mouse', 'house mouse']),
    ('NCBITaxon:10116', 'Rattus norvegicus', ['rat', 'brown rat']),
    ('NCBITaxon:9544', 'Macaca mulatta', ['rhesus macaque']),
    ('NCBITaxon:7955', 'Danio rerio', ['zebrafish']),
]

PATO = [('PATO:0000461', 'normal', ['healthy', 'unaffected'])]

MONDO = [
    ('MONDO:0005148', 'type 2 diabetes mellitus', ['T2D', 'type II diabetes']),
    ('MONDO:0004979', 'asthma', ['bronchial asthma']),
    ('MONDO:0007254', 'breast cancer', ['breast carcinoma']),
    ('MONDO:0005105', 'melanoma', ['malignant melanoma']),
    ('MONDO:0005061', 'lung adenocarcinoma', ['LUAD']),
    ('MONDO:0005575', 'colorectal cancer', ['CRC', 'colon cancer']),
    ('MONDO:0004975', 'Alzheimer disease', ['AD', "Alzheimer's disease"]),
    ('MONDO:0005252', 'heart failure', ['CHF']),
    ('MONDO:0005300', 'chronic kidney disease', ['CKD']),
    ('MONDO:0005015', 'diabetes mellitus', ['diabetes']),
]

HSAPDV = [
    ('HsapDv:0000002', 'human embryo stage', ['embryo', 'embryonic stage']),
    ('HsapDv:0000019', 'human fetal stage', ['fetus', 'fetal stage']),
    ('HsapDv:0000020', 'human infant stage', ['infant', 'infancy']),
    ('HsapDv:0000027', 'human child stage', ['child', 'childhood']),
    ('HsapDv:0000035', 'human adolescent stage', ['adolescent', 'teenager']),
    ('HsapDv:0000087', 'human adult stage', ['adult', 'adulthood']),
    ('HsapDv:0000091', 'human late adult stage', ['elderly', 'late adulthood']),
]

MMUSDV = [
    ('MmusDv:0000002', 'mouse embryo stage', ['mouse embryo', 'embryonic stage']),
    ('MmusDv:0000007', 'mouse neonate stage', ['neonate', 'newborn']),
    ('MmusDv:0000010', 'mouse young adult stage', ['young adult']),
    ('MmusDv:0000013', 'mouse adult stage', ['adult']),
    ('MmusDv:0000016', 'mouse middle aged stage', ['middle aged']),
    ('MmusDv:0000019', 'mouse aged stage', ['aged', 'old']),
]

TABLES = [('cl', CL), ('uberon', UBERON), ('efo', EFO), ('ncbitaxon', NCBITAXON),
          ('pato', PATO), ('mondo', MONDO), ('hsapdv', HSAPDV), ('mmusdv', MMUSDV)]

TISSUE_CELLTYPES = {
    'UBERON:0002048': ['CL:0000066', 'CL:0000235', 'CL:0000084', 'CL:0000057', 'CL:0000115', 'CL:0000192', 'CL:0000623'],
    'UBERON:0002107': ['CL:0000182', 'CL:0000115', 'CL:0000084', 'CL:0000235', 'CL:0000057', 'CL:0000623'],
    'UBERON:0000948': ['CL:0000746', 'CL:0000057', 'CL:0000115', 'CL:0000192', 'CL:0000235'],
    'UBERON:0000955': ['CL:0000540', 'CL:0000127', 'CL:0000128', 'CL:0000129', 'CL:0000115'],
    'UBERON:0002113': ['CL:0000653', 'CL:0000066', 'CL:0000115', 'CL:0000084', 'CL:0000235'],
    'UBERON:0002097': ['CL:0000312', 'CL:0000057', 'CL:0000115', 'CL:0000084', 'CL:0000235'],
    'UBERON:0001134': ['CL:0000187', 'CL:0000057', 'CL:0000115', 'CL:0000084'],
    'UBERON:0000178': ['CL:0000084', 'CL:0000236', 'CL:0000576', 'CL:0000623', 'CL:0000235', 'CL:0000451', 'CL:0000037', 'CL:0000775'],
    'UBERON:0002371': ['CL:0000037', 'CL:0000236', 'CL:0000084', 'CL:0000576', 'CL:0000775'],
    'UBERON:0001155': ['CL:0000066', 'CL:0000084', 'CL:0000057', 'CL:0000115', 'CL:0000235'],
    'UBERON:0001264': ['CL:0002064', 'CL:0000066', 'CL:0000115', 'CL:0000057'],
    'UBERON:0001013': ['CL:0000136', 'CL:0000115', 'CL:0000084', 'CL:0000057'],
    'UBERON:0002106': ['CL:0000084', 'CL:0000236', 'CL:0000235', 'CL:0000451'],
    'UBERON:0000029': ['CL:0000084', 'CL:0000236', 'CL:0000451', 'CL:0000235'],
    'UBERON:0000966': ['CL:0000540', 'CL:0000115', 'CL:0000057'],
    'UBERON:0001987': ['CL:0000066', 'CL:0000057', 'CL:0000115'],
    'UBERON:0000945': ['CL:0000066', 'CL:0000084', 'CL:0000057', 'CL:0000115'],
    'UBERON:0002046': ['CL:0000066', 'CL:0000115', 'CL:0000084'],
    'UBERON:0000473': ['CL:0000066', 'CL:0000057', 'CL:0000115', 'CL:0000235'],
    'UBERON:0000992': ['CL:0000066', 'CL:0000057', 'CL:0000115', 'CL:0000235'],
    'UBERON:0001043': ['CL:0000066', 'CL:0000057', 'CL:0000115', 'CL:0000084'],
    'UBERON:0001295': ['CL:0000066', 'CL:0000057', 'CL:0000115', 'CL:0000623'],
}

SEX_VARIANTS = {'female': ['female', 'Female', 'F', 'woman'],
                'male': ['male', 'Male', 'M', 'man'],
                'unknown': ['unknown', 'not collected', 'NA', ''],
                'mixed': ['mixed', 'Mixed']}
HEALTHY_TEXT = ['healthy', 'healthy donor', 'healthy control', 'normal', 'control', 'none']
DISEASED_TEXT = ['diseased', 'disease', 'sick', 'patient', 'case']
DISEASE_FREE_HEALTHY = ['none', 'healthy', 'not applicable']

GENE_SYMBOLS = ['MALAT1', 'ACTB', 'GAPDH', 'CD3D', 'CD3E', 'CD3G', 'TRAC', 'IL7R', 'NKG7',
                'GNLY', 'KLRD1', 'MS4A1', 'CD79A', 'CD79B', 'CD19', 'LYZ', 'S100A8', 'S100A9',
                'FCGR3A', 'CST3', 'COL1A1', 'COL1A2', 'DCN', 'PECAM1', 'VWF', 'CLDN5', 'ALB',
                'APOA1', 'APOA2', 'TTR', 'SFTPC', 'SFTPB', 'SCGB1A1', 'EPCAM', 'KRT5', 'KRT14',
                'KRT1', 'AQP4', 'GFAP', 'SLC1A2', 'MBP', 'PLP1', 'MOBP', 'CX3CR1', 'P2RY12',
                'TMEM119', 'PTPRC', 'ITGAX', 'CLEC9A', 'LILRA4', 'GZMB', 'PRF1', 'MKI67',
                'TOP2A', 'CD34', 'KIT', 'PPBP', 'PF4', 'HBB', 'HBA1', 'ALAS2', 'MZB1',
                'JCHAIN', 'IGHG1']

META_COLS = ['cell_id', 'donor_id', 'tissue_free_text', 'cell_type_free_text',
             'assay_free_text', 'organism_free_text', 'donor_sex_free_text',
             'development_stage_free_text', 'donor_health_status_free_text',
             'donor_disease_free_text']

MIN_GENES_RETAINED = 100


def pick(rng, values, weights):
    x = rng.random()
    acc = 0.0
    for v, w in zip(values, weights):
        acc += w
        if x < acc:
            return v
    return values[-1]


def render_term(rng, term, garbage_prob=0.04):
    if rng.random() < garbage_prob:
        return rng.choice(GARBAGE)
    r = rng.random()
    if r < 0.12:
        s = term[0]
    elif r < 0.55:
        s = term[1]
    else:
        s = rng.choice(term[2]) if term[2] else term[1]
    if rng.random() < 0.35:
        s = s.upper() if rng.random() < 0.6 else s.lower()
    if rng.random() < 0.30:
        pad = ' ' * rng.randint(1, 3)
        s = pad + s if rng.random() < 0.5 else s + pad
    if rng.random() < 0.15 and ' ' in s:
        s = s.replace(' ', '  ')
    return s


def render_sex(rng, sex):
    if rng.random() < 0.05:
        return rng.choice(GARBAGE)
    s = rng.choice(SEX_VARIANTS[sex])
    if rng.random() < 0.30:
        s = s.upper()
    if rng.random() < 0.20:
        s = ' ' + s + ' '
    return s


def write_vocab(root):
    vos = os.path.join(root, 'ontology')
    os.makedirs(vos, exist_ok=True)
    for name, table in TABLES:
        seen = {}
        for tid, label, syns in table:
            for v in [tid, label] + list(syns):
                k = ' '.join(v.strip().split()).lower()
                if k in seen and seen[k] != tid:
                    raise SystemExit('duplicate vocabulary key %r in %s' % (k, name))
                seen[k] = tid
        with open(os.path.join(vos, name + '.tsv'), 'w') as fh:
            fh.write('id\tlabel\tsynonyms\n')
            for tid, label, syns in table:
                fh.write('%s\t%s\t%s\n' % (tid, label, '|'.join(syns)))


def write_inputs(root):
    sub = os.path.join(root, 'submission')
    os.makedirs(sub, exist_ok=True)
    write_vocab(root)
    rng = random.Random(SEED)

    donors = []
    for _ in range(N_DONORS):
        r = rng.random()
        if r < 0.78:
            org = NCBITAXON[0]
        elif r < 0.93:
            org = NCBITAXON[1]
        elif r < 0.97:
            org = NCBITAXON[2]
        else:
            org = NCBITAXON[3]
        tissue = rng.choice([t for t in UBERON if t[0] in TISSUE_CELLTYPES])
        sex = pick(rng, ['female', 'male', 'unknown', 'mixed'], [0.42, 0.42, 0.10, 0.06])
        if org[0] == 'NCBITaxon:9606':
            dev = rng.choice(HSAPDV)
        elif org[0] == 'NCBITaxon:10090':
            dev = rng.choice(MMUSDV)
        else:
            dev = None
        healthy = rng.random() < 0.55
        disease = None if healthy else rng.choice(MONDO)
        assay = rng.choice(EFO)
        donors.append(dict(org=org, tissue=tissue, sex=sex, dev=dev, healthy=healthy,
                           disease=disease, assay=assay,
                           n=rng.randint(*CELLS_PER_DONOR)))

    rows = []
    n = 0
    for di, d in enumerate(donors):
        donor_id = 'DONOR-%03d' % (di + 1)
        for _ in range(d['n']):
            n += 1
            if rng.random() < 0.90:
                tis = d['tissue']
            else:
                tis = rng.choice([t for t in UBERON if t[0] in TISSUE_CELLTYPES])
            cid = rng.choice(TISSUE_CELLTYPES[tis[0]])
            cterm = next(t for t in CL if t[0] == cid)
            asy = d['assay'] if rng.random() < 0.85 else rng.choice(EFO)
            og = d['org'] if rng.random() < 0.90 else rng.choice(NCBITAXON)
            sx = d['sex'] if rng.random() < 0.90 else rng.choice(['female', 'male', 'unknown'])
            if d['dev'] is None:
                dv_text = rng.choice(GARBAGE_DEV)
            elif rng.random() < 0.90:
                dv_text = render_term(rng, d['dev'])
            else:
                pool = HSAPDV if d['org'][0] == 'NCBITaxon:9606' else MMUSDV
                dv_text = render_term(rng, rng.choice(pool))
            if rng.random() < 0.90:
                h_state = d['healthy']
            else:
                h_state = not d['healthy']
            if h_state:
                h_text = rng.choice(HEALTHY_TEXT)
                d_text = rng.choice(DISEASE_FREE_HEALTHY)
            else:
                h_text = rng.choice(DISEASED_TEXT)
                if d['disease'] is not None and rng.random() < 0.90:
                    d_text = render_term(rng, d['disease'])
                else:
                    d_text = render_term(rng, rng.choice(MONDO))
            if rng.random() < 0.06:
                h_text = rng.choice(GARBAGE)
            rows.append({
                'cell_id': 'C%06d' % n,
                'donor_id': donor_id,
                'tissue_free_text': render_term(rng, tis),
                'cell_type_free_text': render_term(rng, cterm),
                'assay_free_text': render_term(rng, asy),
                'organism_free_text': render_term(rng, og, 0.0),
                'donor_sex_free_text': render_sex(rng, sx),
                'development_stage_free_text': dv_text,
                'donor_health_status_free_text': h_text,
                'donor_disease_free_text': d_text,
            })

    insertions = []
    for i in rng.sample(range(len(rows)), N_DUPLICATE_CELLS):
        pos = rng.randint(i + 1, min(len(rows), i + 25))
        cp = dict(rows[i])
        cp['assay_free_text'] = render_term(rng, rng.choice(EFO))
        insertions.append((pos, cp))
    for pos, cp in sorted(insertions, reverse=True):
        rows.insert(pos, cp)

    low_idx = sorted(rng.sample(range(len(rows)), N_LOW_COMPLEXITY))

    genes = []
    used = set()
    while len(genes) < N_GENES:
        if rng.random() < 0.82:
            gid = 'ENSG' + '%011d' % rng.randint(1, 99999999999)
        else:
            gid = 'ENSMUSG' + '%011d' % rng.randint(1, 99999999999)
        if gid in used:
            continue
        used.add(gid)
        if len(genes) < len(GENE_SYMBOLS):
            name = GENE_SYMBOLS[len(genes)]
        else:
            name = 'LOC%09d' % (100000000 + len(genes) * 137)
        if rng.random() < 0.35:
            gid = gid + '.%d' % rng.randint(1, 15)
        genes.append({'gene_id': gid, 'gene_name': name})

    gins = []
    for j in rng.sample(range(len(genes)), N_DUPLICATE_GENES):
        cp = dict(genes[j])
        base = cp['gene_id'].split('.')[0]
        cp['gene_id'] = base + '.%d' % rng.randint(16, 30)
        gins.append((j + 1, cp))
    for pos, cp in sorted(gins, reverse=True):
        genes.insert(pos, cp)

    stripped = [g['gene_id'].split('.')[0] for g in genes]
    uniq = []
    for s in stripped:
        if s not in uniq:
            uniq.append(s)
    zero_ids = set(rng.sample(uniq, N_ZERO_GENES))

    ncells = len(rows)
    expressed = [g for g in uniq if g not in zero_ids]
    by_gene = {}
    for g in expressed:
        vec = [0] * ncells
        for ci in range(ncells):
            if rng.random() < DENSITY:
                vec[ci] = 1 + rng.randint(0, 29)
        by_gene[g] = vec
    for ci in low_idx:
        for g in expressed:
            by_gene[g][ci] = 0
        for g in rng.sample(expressed, rng.randint(20, 90)):
            by_gene[g][ci] = 1 + rng.randint(0, 29)

    with open(os.path.join(sub, 'cell_metadata.tsv'), 'w') as fh:
        fh.write('\t'.join(META_COLS) + '\n')
        for r in rows:
            fh.write('\t'.join(r[c] for c in META_COLS) + '\n')

    with open(os.path.join(sub, 'genes.tsv'), 'w') as fh:
        fh.write('gene_id\tgene_name\n')
        for g in genes:
            fh.write('%s\t%s\n' % (g['gene_id'], g['gene_name']))

    nz_lines = []
    for ci in range(ncells):
        r = ci + 1
        for cj, s in enumerate(stripped):
            vec = by_gene.get(s)
            if vec is None:
                continue
            v = vec[ci]
            if v:
                nz_lines.append('%d %d %d' % (r, cj + 1, v))
    with open(os.path.join(sub, 'counts.mtx'), 'w') as fh:
        fh.write('%%MatrixMarket matrix coordinate integer general\n')
        fh.write('%%synthetic submission slice, seed %d\n' % SEED)
        fh.write('%d %d %d\n' % (ncells, len(stripped), len(nz_lines)))
        fh.write('\n'.join(nz_lines))
        fh.write('\n')


def norm(s):
    return ' '.join(s.strip().split()).lower()


def strip_version(gene_id):
    return gene_id.split('.', 1)[0]


def read_tsv(path):
    import csv
    with open(path, newline='') as fh:
        return [row for row in csv.reader(fh, delimiter='\t') if row]


def build_lookup(table):
    lut = {}
    for tid, label, syns in table:
        for v in [tid, label] + list(syns):
            k = norm(v)
            if k and k not in lut:
                lut[k] = (tid, label)
    return lut


def read_mtx(path):
    dims = None
    entries = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('%'):
                continue
            parts = line.split()
            if not parts:
                continue
            if dims is None:
                dims = tuple(int(x) for x in parts)
            else:
                entries.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return dims, entries


SEX_MAP = {'female': ['female', 'f', 'woman'], 'male': ['male', 'm', 'man'],
           'mixed': ['mixed'], 'unknown': ['unknown']}
HEALTH_MAP = {'healthy': ['healthy', 'healthy donor', 'healthy control', 'normal', 'control',
                          'none', 'unaffected'],
              'diseased': ['diseased', 'disease', 'sick', 'patient', 'case', 'affected']}


def norm_sex(t):
    k = norm(t)
    for sex, vals in SEX_MAP.items():
        if k in vals:
            return sex
    return 'unknown'


def norm_health(t):
    k = norm(t)
    for st, vals in HEALTH_MAP.items():
        if k in vals:
            return st
    return 'unknown'


def majority(values):
    c = Counter(values)
    best = max(c.values())
    return sorted(v for v, n in c.items() if n == best)[0]


def build_gold(root):
    if not os.path.exists(os.path.join(root, 'submission', 'cell_metadata.tsv')):
        write_inputs(root)
    sub = os.path.join(root, 'submission')
    meta = read_tsv(os.path.join(sub, 'cell_metadata.tsv'))
    mrows = [dict(zip(meta[0], r)) for r in meta[1:]]
    gtab = read_tsv(os.path.join(sub, 'genes.tsv'))
    grows = [dict(zip(gtab[0], r)) for r in gtab[1:]]
    dims, entries = read_mtx(os.path.join(sub, 'counts.mtx'))
    assert dims[0] == len(mrows)

    lut = {name: build_lookup(table) for name, table in TABLES}
    id2label = {name: {tid: label for tid, label, _ in table} for name, table in TABLES}

    col_gene = [None]
    for g in grows:
        col_gene.append(strip_version(g['gene_id']))

    detected = Counter()
    for (r, c, v) in entries:
        if v != 0:
            detected[r] += 1

    kept = []
    seen = set()
    n_dup = n_unres = n_low = 0
    for idx, r in enumerate(mrows, start=1):
        cid = r['cell_id']
        if cid in seen:
            n_dup += 1
            continue
        seen.add(cid)
        ct = lut['cl'].get(norm(r['cell_type_free_text']))
        ti = lut['uberon'].get(norm(r['tissue_free_text']))
        asy = lut['efo'].get(norm(r['assay_free_text']))
        org = lut['ncbitaxon'].get(norm(r['organism_free_text']))
        if ct is None or ti is None or asy is None or org is None:
            n_unres += 1
            continue
        if detected.get(idx, 0) < MIN_GENES_RETAINED:
            n_low += 1
            continue
        kept.append(dict(row=idx, cell_id=cid, donor=r['donor_id'], ct=ct, ti=ti, assay=asy,
                         org=org, sex=norm_sex(r['donor_sex_free_text']),
                         dev_text=r['development_stage_free_text'],
                         health=norm_health(r['donor_health_status_free_text']),
                         dis_text=r['donor_disease_free_text'], n_genes=detected.get(idx, 0)))

    by_donor = defaultdict(list)
    for k in kept:
        by_donor[k['donor']].append(k)

    donor_vals = {}
    for donor, ks in by_donor.items():
        org = majority([k['org'][0] for k in ks])
        sex = majority([k['sex'] for k in ks])
        if org == 'NCBITaxon:9606':
            dlut = lut['hsapdv']
        elif org == 'NCBITaxon:10090':
            dlut = lut['mmusdv']
        else:
            dlut = None
        devs = []
        for k in ks:
            if dlut is None:
                devs.append('unknown')
            else:
                d = dlut.get(norm(k['dev_text']))
                devs.append(d[0] if d else 'unknown')
        dev = majority(devs)
        health = majority([k['health'] for k in ks])
        if health == 'diseased':
            ds = []
            for k in ks:
                d = lut['mondo'].get(norm(k['dis_text']))
                ds.append(d[0] if d else 'unknown')
            dis = majority(ds)
        else:
            dis = 'PATO:0000461'
        donor_vals[donor] = dict(org=org, sex=sex, dev=dev, health=health, dis=dis)

    obs = []
    for k in kept:
        dv = donor_vals[k['donor']]
        dev_id = dv['dev']
        if dev_id == 'unknown':
            dev_label = 'unknown'
        else:
            dev_label = id2label['hsapdv'].get(dev_id) or id2label['mmusdv'].get(dev_id)
        dis_id = dv['dis'] if dv['health'] == 'diseased' else 'PATO:0000461'
        if dis_id == 'unknown':
            dis_label = 'unknown'
        else:
            dis_label = id2label['pato'].get(dis_id) or id2label['mondo'].get(dis_id)
        obs.append({
            'cell_id': k['cell_id'],
            'donor_id': k['donor'],
            'tissue_ontology_term_id': k['ti'][0],
            'tissue': k['ti'][1],
            'cell_type_ontology_term_id': k['ct'][0],
            'cell_type': k['ct'][1],
            'assay_ontology_term_id': k['assay'][0],
            'assay': k['assay'][1],
            'disease_ontology_term_id': dis_id,
            'disease': dis_label,
            'organism_ontology_term_id': dv['org'],
            'organism': id2label['ncbitaxon'][dv['org']],
            'sex': dv['sex'],
            'development_stage_ontology_term_id': dev_id,
            'development_stage': dev_label,
            'n_genes_detected': str(k['n_genes']),
        })
    obs.sort(key=lambda d: d['cell_id'])

    first = {}
    for g in grows:
        gid = strip_version(g['gene_id'])
        if gid not in first:
            first[gid] = g['gene_name']
    retained_rows = set(k['row'] for k in kept)
    totals = defaultdict(int)
    for (r, c, v) in entries:
        if r in retained_rows and v:
            gid = col_gene[c]
            if gid is not None:
                totals[gid] += v
    var = []
    for gid in sorted(first):
        var.append({'gene_id': gid, 'gene_name': first[gid],
                    'feature_is_filtered': 'TRUE' if totals.get(gid, 0) == 0 else 'FALSE'})

    report = {
        'n_input_rows': len(mrows),
        'n_duplicate_rows_removed': n_dup,
        'n_dropped_unresolved_ontology': n_unres,
        'n_dropped_low_complexity': n_low,
        'n_retained_cells': len(kept),
        'n_retained_donors': len(by_donor),
        'n_genes_input_rows': len(grows),
        'n_genes_output_rows': len(first),
        'n_genes_filtered': sum(1 for v in var if v['feature_is_filtered'] == 'TRUE'),
    }
    return {'obs': obs, 'var': var, 'report': report,
            'obs_ids': set(d['cell_id'] for d in obs),
            'donor_ids': set(by_donor.keys())}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='/tmp/gold_inputs')
    args = ap.parse_args()
    write_inputs(args.outdir)
    print('wrote inputs to', args.outdir)
