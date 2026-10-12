#!/usr/bin/env python3
import os, random

SEED = 20260506
GENOME_LEN = 29903
REF_NAME = 'synthetic_ref'
DATA = os.path.dirname(os.path.abspath(__file__))

GENES = [
    ('ORF1ab', 266, 13468),
    ('S', 21563, 25384),
    ('ORF3a', 25393, 26220),
    ('E', 26245, 26472),
    ('M', 26523, 27191),
    ('ORF6', 27202, 27387),
    ('ORF7a', 27394, 27759),
    ('ORF7b', 27760, 27891),
    ('ORF8', 27894, 28259),
    ('N', 28274, 29533),
    ('ORF10', 29558, 29674),
]

def make_ref(seed):
    rng = random.Random(seed)
    return ''.join(rng.choice('ACGT') for _ in range(GENOME_LEN))

def write_fasta(path, name, seq, width=60):
    with open(path, 'w') as f:
        f.write(f'>{name}\n')
        for i in range(0, len(seq), width):
            f.write(seq[i:i+width] + '\n')

def write_gff(path):
    with open(path, 'w') as f:
        f.write('##gff-version 3\n')
        for name, start, end in GENES:
            f.write(f'{REF_NAME}\tsynthetic\tCDS\t{start}\t{end}\t.\t+\t0\tID=gene:{name};Name={name}\n')

def main():
    rng = random.Random(SEED)
    ref = make_ref(SEED)
    write_fasta(os.path.join(DATA, 'reference.fa'), REF_NAME, ref)
    write_gff(os.path.join(DATA, 'genes.gff3'))
    os.makedirs(os.path.join(DATA, 'vcf'), exist_ok=True)
    os.makedirs(os.path.join(DATA, 'coverage'), exist_ok=True)

    samples = [
        ('P1_D0', 0, 0),
        ('P1_D3', 0, 3),
        ('P1_D6', 0, 6),
        ('P1_D10', 0, 10),
        ('P1_D14', 0, 14),
        ('P1_D21', 0, 21),
        ('P1_D28', 0, 28),
        ('P1_D35', 0, 35),
    ]
    with open(os.path.join(DATA, 'manifest.tsv'), 'w') as f:
        f.write('sample_id\tpatient_id\ttimepoint\tdays_since_onset\tvcf_path\tcoverage_path\n')
        for i, (sid, pat, days) in enumerate(samples):
            f.write(f'{sid}\tP1\t{i}\t{days}\tvcf/{sid}.vcf\tcoverage/{sid}.coverage.tsv\n')

    n_variants = 150
    positions = rng.sample(range(1, GENOME_LEN + 1), n_variants)
    variants = []
    for pos in positions:
        ref_base = ref[pos - 1]
        alt = rng.choice([b for b in 'ACGT' if b != ref_base])
        r = rng.random()
        if r < 0.25:
            present = set(range(len(samples)))
        elif r < 0.45:
            present = {0}
        elif r < 0.75:
            start = rng.randint(1, len(samples) - 2)
            present = set(range(start, len(samples)))
        elif r < 0.90:
            start = rng.randint(1, len(samples) - 1)
            present = {start}
        else:
            start = rng.randint(0, len(samples) - 4)
            present = {start, start + 2, start + 4}
        variants.append({
            'pos': pos,
            'ref': ref_base,
            'alt': alt,
            'present': present,
        })
    variants.sort(key=lambda x: (x['pos'], x['ref'], x['alt']))

    sample_records = {sid: [] for sid, _, _ in samples}
    total_records = 0
    low_quality_records = 0
    low_coverage_records = 0
    pass_records = 0

    for v in variants:
        for si, (sid, pat, days) in enumerate(samples):
            if si not in v['present']:
                continue
            if rng.random() < 0.12:
                cov = rng.randint(10, 45)
            else:
                cov = rng.randint(60, 1200)
            if rng.random() < 0.08:
                filt = 'LowQual'
            else:
                filt = 'PASS'
            sample_records[sid].append({
                'pos': v['pos'],
                'ref': v['ref'],
                'alt': v['alt'],
                'cov': cov,
                'filter': filt,
            })
            total_records += 1
            if filt != 'PASS':
                low_quality_records += 1
            elif cov < 50:
                low_coverage_records += 1
            else:
                pass_records += 1

    for sid, _, _ in samples:
        recs = sorted(sample_records[sid], key=lambda x: x['pos'])
        vcf_path = os.path.join(DATA, 'vcf', f'{sid}.vcf')
        with open(vcf_path, 'w') as f:
            f.write('##fileformat=VCFv4.2\n')
            f.write(f'##contig=<ID={REF_NAME},length={GENOME_LEN}>\n')
            f.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth">\n')
            f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
            f.write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n')
            f.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + sid + '\n')
            for rec in recs:
                f.write(f'{REF_NAME}\t{rec["pos"]}\t.\t{rec["ref"]}\t{rec["alt"]}\t1000\t{rec["filter"]}\tDP=100\tGT:DP\t0/1:100\n')

    sample_pos_cov = {sid: {} for sid, _, _ in samples}
    for sid, _, _ in samples:
        for rec in sample_records[sid]:
            sample_pos_cov[sid][rec['pos']] = rec['cov']
    for sid, _, _ in samples:
        cov_path = os.path.join(DATA, 'coverage', f'{sid}.coverage.tsv')
        with open(cov_path, 'w') as f:
            f.write('chrom\tpos\tcoverage\n')
            for pos in range(1, GENOME_LEN + 1):
                cov = sample_pos_cov[sid].get(pos)
                if cov is None:
                    cov = rng.randint(80, 250)
                f.write(f'{REF_NAME}\t{pos}\t{cov}\n')

if __name__ == '__main__':
    main()
