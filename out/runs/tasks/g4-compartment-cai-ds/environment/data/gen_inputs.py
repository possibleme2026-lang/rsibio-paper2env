import os, random, math, re
from collections import defaultdict

SEED = 20260521

CODONS = {
    'A': ['GCT','GCC','GCA','GCG'],
    'R': ['CGT','CGC','CGA','CGG','AGA','AGG'],
    'N': ['AAT','AAC'],
    'D': ['GAT','GAC'],
    'C': ['TGT','TGC'],
    'Q': ['CAA','CAG'],
    'E': ['GAA','GAG'],
    'G': ['GGT','GGC','GGA','GGG'],
    'H': ['CAT','CAC'],
    'I': ['ATT','ATC','ATA'],
    'L': ['TTA','TTG','CTT','CTC','CTA','CTG'],
    'K': ['AAA','AAG'],
    'M': ['ATG'],
    'F': ['TTT','TTC'],
    'P': ['CCT','CCC','CCA','CCG'],
    'S': ['TCT','TCC','TCA','TCG','AGT','AGC'],
    'T': ['ACT','ACC','ACA','ACG'],
    'W': ['TGG'],
    'Y': ['TAT','TAC'],
    'V': ['GTT','GTC','GTA','GTG'],
}

def revcomp(seq):
    comp = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
    return ''.join(comp[b] for b in reversed(seq))

def random_g4_motif():
    while True:
        tracts = [random.randint(3,5) for _ in range(4)]
        loops = [random.randint(1,7) for _ in range(3)]
        seq = 'G'*tracts[0]
        for i in range(3):
            seq += ''.join(random.choice('ACGT') for _ in range(loops[i]))
            seq += 'G'*tracts[i+1]
        if len(seq) <= 45:
            return seq

def find_g4_intervals(seq):
    intervals = []
    pattern = re.compile(r'G{3,5}(?:[ACGT]{1,7}G{3,5}){3}')
    for m in pattern.finditer(seq):
        if len(m.group()) <= 45:
            intervals.append((m.start(), m.end()))
    rc = revcomp(seq)
    for m in pattern.finditer(rc):
        if len(m.group()) <= 45:
            s = len(seq) - m.end()
            e = len(seq) - m.start()
            intervals.append((s, e))
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged

def generate_ortholog_pair():
    seq1 = []
    for _ in range(200):
        aa = random.choice(list(CODONS.keys()))
        seq1.append(random.choice(CODONS[aa]))
    seq1 = ''.join(seq1)
    motif = random_g4_motif()
    pos = random.randint(0, len(seq1)-len(motif))
    seq1 = seq1[:pos] + motif + seq1[pos+len(motif):]
    g4_intervals = find_g4_intervals(seq1)
    cai = random.uniform(0.2, 0.9)
    seq2 = list(seq1)
    for i in range(0, len(seq1), 3):
        codon1 = seq1[i:i+3]
        if len(codon1) < 3:
            break
        is_g4 = any(not (i+3 <= s or i >= e) for s, e in g4_intervals)
        aa = None
        for a, codons in CODONS.items():
            if codon1 in codons:
                aa = a
                break
        if aa is None:
            continue
        if is_g4:
            syn_p = 0.10
            nonsyn_p = 0.10
        else:
            syn_p = 0.30
            nonsyn_p = 0.10
        factor = max(0.1, 1.5 - cai)
        syn_p *= factor
        nonsyn_p *= factor
        r = random.random()
        if r < syn_p:
            syn_codons = [c for c in CODONS[aa] if c != codon1]
            if syn_codons:
                seq2[i:i+3] = list(random.choice(syn_codons))
        elif r < syn_p + nonsyn_p:
            other_aas = [a for a in CODONS if a != aa]
            new_aa = random.choice(other_aas)
            seq2[i:i+3] = list(random.choice(CODONS[new_aa]))
    return seq1, ''.join(seq2), cai

def generate_genome(species):
    chroms = []
    for chrom_id in range(2):
        seq = [random.choice('ACGT') for _ in range(300000)]
        chroms.append(seq)
    genes = []
    for chrom_id in range(2):
        seq = chroms[chrom_id]
        occupied = []
        num_genes = 50
        placed = 0
        attempts = 0
        while placed < num_genes and attempts < 20000:
            attempts += 1
            exons = [150]*5
            introns = [random.randint(200,800) for _ in range(4)]
            gene_len = sum(exons) + sum(introns)
            start = random.randint(2000, 300000 - gene_len - 2000)
            end = start + gene_len
            overlap = False
            for s, e in occupied:
                if not (end + 2000 < s or start - 2000 > e):
                    overlap = True
                    break
            if overlap:
                continue
            occupied.append((start, end))
            placed += 1
            strand = random.choice(['+', '-'])
            intervals = []
            pos = start
            for i in range(5):
                intervals.append(('exon', pos, pos+150))
                pos += 150
                if i < 4:
                    intervals.append(('intron', pos, pos+introns[i]))
                    pos += introns[i]
            genes.append({
                'chrom': f"{species}_chr{chrom_id+1}",
                'start': start + 1,
                'end': end,
                'strand': strand,
                'intervals': intervals,
            })
    cds_intervals = []
    intron_intervals = []
    promoter_intervals = []
    for g in genes:
        for typ, s, e in g['intervals']:
            if typ == 'exon':
                cds_intervals.append((g['chrom'], s, e))
            else:
                intron_intervals.append((g['chrom'], s, e))
        if g['strand'] == '+':
            p_start = max(0, g['start'] - 1001)
            p_end = min(300000, g['start'] - 1)
        else:
            p_start = max(0, g['end'])
            p_end = min(300000, g['end'] + 1000)
        if p_end > p_start:
            promoter_intervals.append((g['chrom'], p_start, p_end))
    def insert_motifs(intervals, n, chroms):
        chrom_map = {f"{species}_chr{i+1}": i for i in range(2)}
        for _ in range(n):
            chrom, s, e = random.choice(intervals)
            if e - s < 45:
                continue
            motif = random_g4_motif()
            max_pos = e - len(motif)
            if max_pos < s:
                continue
            pos = random.randint(s, max_pos)
            seq = chroms[chrom_map[chrom]]
            for i, base in enumerate(motif):
                seq[pos+i] = base
    insert_motifs(cds_intervals, 5, chroms)
    insert_motifs(intron_intervals, 40, chroms)
    insert_motifs(promoter_intervals, 30, chroms)
    return chroms, genes

def main():
    random.seed(SEED)
    base_dir = '/app/data'
    os.makedirs(os.path.join(base_dir, 'genomes'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'orthologs'), exist_ok=True)
    species_list = [
        ('Homo_sapiens', 'Animalia'),
        ('Arabidopsis_thaliana', 'Plantae'),
        ('Saccharomyces_cerevisiae', 'Fungi'),
        ('Plasmodium_falciparum', 'Protozoa'),
    ]
    all_genes = []
    for species, kingdom in species_list:
        chroms, genes = generate_genome(species)
        for g in genes:
            g['species'] = species
        all_genes.extend(genes)
        with open(os.path.join(base_dir, 'genomes', f'{species}.fa'), 'w') as f:
            for i, seq in enumerate(chroms):
                f.write(f">{species}_chr{i+1}\n")
                for j in range(0, len(seq), 60):
                    f.write(''.join(seq[j:j+60]) + '\n')
    with open(os.path.join(base_dir, 'annotation.gff3'), 'w') as f:
        for g in all_genes:
            gid = f"gene_{g['species']}_{g['chrom']}_{g['start']}"
            f.write(f"{g['chrom']}\tsynthetic\tgene\t{g['start']}\t{g['end']}\t.\t{g['strand']}\t.\tID={gid}\n")
            for typ, s, e in g['intervals']:
                if typ == 'exon':
                    f.write(f"{g['chrom']}\tsynthetic\tCDS\t{s+1}\t{e}\t.\t{g['strand']}\t0\tParent={gid}\n")
                else:
                    f.write(f"{g['chrom']}\tsynthetic\tintron\t{s+1}\t{e}\t.\t{g['strand']}\t.\tParent={gid}\n")
    with open(os.path.join(base_dir, 'codon_usage.tsv'), 'w') as f:
        f.write('species\tcodon\tRSCU\n')
        for species, kingdom in species_list:
            for aa, codons in CODONS.items():
                vals = [random.random() for _ in codons]
                s = sum(vals)
                rscu = [v / s * len(codons) for v in vals]
                for c, r in zip(codons, rscu):
                    f.write(f'{species}\t{c}\t{r:.4f}\n')
    species_names = [s for s, k in species_list]
    pairs = []
    pair_idx = 0
    for i in range(len(species_names)):
        for j in range(i+1, len(species_names)):
            sp1 = species_names[i]
            sp2 = species_names[j]
            for k in range(20):
                seq1, seq2, cai = generate_ortholog_pair()
                pair_id = f'pair_{pair_idx:03d}'
                with open(os.path.join(base_dir, 'orthologs', f'{pair_id}.fa'), 'w') as f:
                    f.write(f'>{sp1}_{pair_id}_1\n{seq1}\n')
                    f.write(f'>{sp2}_{pair_id}_2\n{seq2}\n')
                pairs.append((pair_id, sp1, sp2, f'{pair_id}.fa'))
                pair_idx += 1
    with open(os.path.join(base_dir, 'ortholog_pairs.tsv'), 'w') as f:
        f.write('pair_id\tspecies1\tspecies2\tfile\n')
        for row in pairs:
            f.write('\t'.join(row) + '\n')

if __name__ == '__main__':
    main()
