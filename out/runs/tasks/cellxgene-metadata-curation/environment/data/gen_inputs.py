#!/usr/bin/env python3
'''Deterministic generator of a synthetic CELLxGENE-style submission slice.

Only rendered (messy) metadata strings plus vocabulary tables are emitted; no
canonical answer key is stored, so the standardized output is fully
recoverable from the emitted files alone.

Usage: python gen_inputs.py --outdir /app/data
'''
import argparse
import os
import random

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='/app/data')
    args = ap.parse_args()
    root = args.outdir
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


if __name__ == '__main__':
    main()
