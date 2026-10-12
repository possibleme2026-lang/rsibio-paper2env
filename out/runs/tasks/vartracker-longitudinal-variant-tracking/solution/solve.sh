#!/bin/bash
set -euo pipefail
mkdir -p /app/output
cat > /app/solve.py <<'PY'
#!/usr/bin/env python3
import os, json, csv
from collections import defaultdict

DATA = '/app/data'
OUT = '/app/output'
os.makedirs(OUT, exist_ok=True)

def read_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                continue
            seq.append(line.strip())
    return ''.join(seq)

ref = read_fasta(os.path.join(DATA, 'reference.fa'))

genes = []
with open(os.path.join(DATA, 'genes.gff3')) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue
        if parts[2] == 'CDS':
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            name = None
            for kv in parts[8].split(';'):
                if kv.startswith('Name='):
                    name = kv[5:]
                elif kv.startswith('ID='):
                    name = kv[3:]
            genes.append((name, start, end, strand))
genes.sort(key=lambda x: x[1])

samples = []
with open(os.path.join(DATA, 'manifest.tsv')) as f:
    header = f.readline().strip().split('\t')
    for line in f:
        parts = line.strip().split('\t')
        samples.append(dict(zip(header, parts)))
samples.sort(key=lambda x: int(x['timepoint']))

cov = {}
for s in samples:
    sid = s['sample_id']
    cov[sid] = {}
    path = os.path.join(DATA, s['coverage_path'])
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            cov[sid][int(parts[1])] = int(parts[2])

variant_samples = defaultdict(set)
variant_cov = defaultdict(dict)
total_records = 0
low_quality = 0
low_coverage = 0
pass_records = 0

for s in samples:
    sid = s['sample_id']
    vcf_path = os.path.join(DATA, s['vcf_path'])
    with open(vcf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            total_records += 1
            chrom, pos, vid, ref_b, alt, qual, filt, info = parts[:8]
            pos = int(pos)
            variant_id = f'{chrom}:{pos}:{ref_b}:{alt}'
            c = cov[sid].get(pos, 0)
            if filt != 'PASS':
                low_quality += 1
                continue
            if c < 50:
                low_coverage += 1
                continue
            pass_records += 1
            variant_samples[variant_id].add(sid)
            variant_cov[variant_id][sid] = c

codon_table = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

def translate(codon):
    return codon_table.get(codon.upper(), 'X')

def annotate(pos, ref_b, alt):
    for name, start, end, strand in genes:
        if start <= pos <= end:
            if strand == '+':
                offset = pos - start
                codon_start = start + (offset // 3) * 3
                codon_pos = offset // 3 + 1
                ref_codon = ref[codon_start-1:codon_start+2]
                alt_codon = list(ref_codon)
                idx = offset % 3
                alt_codon[idx] = alt
                ref_aa = translate(ref_codon)
                alt_aa = translate(''.join(alt_codon))
                return name, f'{name}:{ref_aa}{codon_pos}{alt_aa}'
            else:
                pass
    return 'NA', 'NA'

rows = []
for vid, sample_set in variant_samples.items():
    chrom, pos_s, ref_b, alt = vid.split(':')
    pos = int(pos_s)
    gene, aa = annotate(pos, ref_b, alt)
    present_samples = sorted(sample_set, key=lambda sid: int(next(s['timepoint'] for s in samples if s['sample_id'] == sid)))
    first_sid = present_samples[0]
    last_sid = present_samples[-1]
    first_tp = int(next(s['timepoint'] for s in samples if s['sample_id'] == first_sid))
    last_tp = int(next(s['timepoint'] for s in samples if s['sample_id'] == last_sid))
    n_present = len(present_samples)
    var_type = 'original' if first_tp == 0 else 'emerging'
    temp_class = 'persistent' if n_present >= 2 else 'transient'
    covs = [variant_cov[vid][sid] for sid in present_samples]
    mean_cov = round(sum(covs)/len(covs), 1)
    rows.append({
        'variant_id': vid,
        'pos': pos,
        'ref': ref_b,
        'alt': alt,
        'gene': gene,
        'aa_change': aa,
        'first_sample': first_sid,
        'first_timepoint': first_tp,
        'last_sample': last_sid,
        'last_timepoint': last_tp,
        'n_present': n_present,
        'variant_type': var_type,
        'temporal_class': temp_class,
        'mean_coverage': mean_cov,
        'present_samples': ','.join(present_samples),
    })
rows.sort(key=lambda r: (r['pos'], r['ref'], r['alt']))

cols = ['variant_id','pos','ref','alt','gene','aa_change','first_sample','first_timepoint','last_sample','last_timepoint','n_present','variant_type','temporal_class','mean_coverage','present_samples']
with open(os.path.join(OUT, 'variants.tsv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter='\t')
    w.writeheader()
    for r in rows:
        w.writerow(r)

sample_ids = [s['sample_id'] for s in samples]
with open(os.path.join(OUT, 'presence_matrix.tsv'), 'w', newline='') as f:
    f.write('variant_id\t' + '\t'.join(sample_ids) + '\n')
    for r in rows:
        vals = []
        for sid in sample_ids:
            vals.append('1' if sid in variant_samples[r['variant_id']] else '0')
        f.write(r['variant_id'] + '\t' + '\t'.join(vals) + '\n')

summary = {
    'total_variants': len(rows),
    'original_variants': sum(1 for r in rows if r['variant_type'] == 'original'),
    'emerging_variants': sum(1 for r in rows if r['variant_type'] == 'emerging'),
    'persistent_variants': sum(1 for r in rows if r['temporal_class'] == 'persistent'),
    'transient_variants': sum(1 for r in rows if r['temporal_class'] == 'transient'),
    'coding_variants': sum(1 for r in rows if r['gene'] != 'NA'),
    'noncoding_variants': sum(1 for r in rows if r['gene'] == 'NA'),
    'n_vcf_records': total_records,
    'n_pass_records': pass_records,
    'n_low_quality_records': low_quality,
    'n_low_coverage_records': low_coverage,
}
with open(os.path.join(OUT, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
PY
python /app/solve.py
