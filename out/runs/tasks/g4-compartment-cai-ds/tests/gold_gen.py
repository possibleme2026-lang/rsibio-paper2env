import os, random, math, re, json
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

AA_TO_CODONS = {}
for aa, codons in CODONS.items():
    for c in codons:
        AA_TO_CODONS[c] = aa

def codon_to_aa(codon):
    return AA_TO_CODONS.get(codon)

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
        aa = codon_to_aa(codon1)
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

def generate_data():
    random.seed(SEED)
    species_list = [
        ('Homo_sapiens', 'Animalia'),
        ('Arabidopsis_thaliana', 'Plantae'),
        ('Saccharomyces_cerevisiae', 'Fungi'),
        ('Plasmodium_falciparum', 'Protozoa'),
    ]
    species_data = {}
    for species, kingdom in species_list:
        chroms, genes = generate_genome(species)
        chroms_dict = {f"{species}_chr{i+1}": ''.join(chroms[i]) for i in range(2)}
        species_data[species] = {'chroms': chroms_dict, 'genes': genes}
    codon_usage = {}
    for species, kingdom in species_list:
        usage = {}
        for aa, codons in CODONS.items():
            vals = [random.random() for _ in codons]
            s = sum(vals)
            rscu = [v / s * len(codons) for v in vals]
            for c, r in zip(codons, rscu):
                usage[c] = r
        codon_usage[species] = usage
    species_names = [s for s, k in species_list]
    pairs = []
    pair_idx = 0
    for i in range(len(species_names)):
        for j in range(i+1, len(species_names)):
            sp1 = species_names[i]
            sp2 = species_names[j]
            for k in range(20):
                seq1, seq2, cai = generate_ortholog_pair()
                pair_id = f"pair_{pair_idx:03d}"
                pairs.append({
                    'pair_id': pair_id,
                    'species1': sp1,
                    'species2': sp2,
                    'seq1': seq1,
                    'seq2': seq2,
                })
                pair_idx += 1
    return {
        'species': species_data,
        'codon_usage': codon_usage,
        'pairs': pairs,
    }

def compute_cai(seq, codon_usage):
    aa_to_codons = defaultdict(list)
    for codon in codon_usage:
        aa = codon_to_aa(codon)
        if aa:
            aa_to_codons[aa].append(codon)
    w = {}
    for aa, codons in aa_to_codons.items():
        max_rscu = max(codon_usage[c] for c in codons)
        for c in codons:
            if max_rscu > 0:
                w[c] = codon_usage[c] / max_rscu
            else:
                w[c] = 0.0
    log_sum = 0.0
    L = 0
    for i in range(0, len(seq), 3):
        codon = seq[i:i+3]
        if len(codon) < 3:
            break
        if codon in w:
            val = w[codon]
            if val == 0:
                val = 0.5
            log_sum += math.log(val)
            L += 1
    if L == 0:
        return 0.0
    return math.exp(log_sum / L)

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged

def compute_gold():
    data = generate_data()
    species_results = {}
    for species, info in data['species'].items():
        chroms = info['chroms']
        genes = info['genes']
        cds_intervals = defaultdict(list)
        intron_intervals = defaultdict(list)
        promoter_intervals = defaultdict(list)
        for g in genes:
            chrom = g['chrom']
            for typ, s, e in g['intervals']:
                if typ == 'exon':
                    cds_intervals[chrom].append((s, e))
                else:
                    intron_intervals[chrom].append((s, e))
            if g['strand'] == '+':
                p_start = max(0, g['start'] - 1001)
                p_end = min(len(chroms[chrom]), g['start'] - 1)
            else:
                p_start = max(0, g['end'])
                p_end = min(len(chroms[chrom]), g['end'] + 1000)
            if p_end > p_start:
                promoter_intervals[chrom].append((p_start, p_end))
        cds_merged = {c: merge_intervals(cds_intervals[c]) for c in chroms}
        intron_merged = {c: merge_intervals(intron_intervals[c]) for c in chroms}
        promoter_merged = {c: merge_intervals(promoter_intervals[c]) for c in chroms}
        g4_by_chrom = {}
        for chrom, seq in chroms.items():
            g4_by_chrom[chrom] = find_g4_intervals(seq)
        counts = {'CDS': 0, 'intron': 0, 'promoter': 0}
        lengths = {'CDS': 0, 'intron': 0, 'promoter': 0}
        for chrom in chroms:
            for comp, merged in [('CDS', cds_merged[chrom]), ('intron', intron_merged[chrom]), ('promoter', promoter_merged[chrom])]:
                for s, e in merged:
                    lengths[comp] += (e - s)
            for s, e in g4_by_chrom[chrom]:
                mid = (s + e) / 2
                for comp, merged in [('CDS', cds_merged[chrom]), ('intron', intron_merged[chrom]), ('promoter', promoter_merged[chrom])]:
                    for ms, me in merged:
                        if ms <= mid < me:
                            counts[comp] += 1
                            break
        densities = {}
        for comp in counts:
            if lengths[comp] > 0:
                densities[comp] = counts[comp] / (lengths[comp] / 1000.0)
            else:
                densities[comp] = 0.0
        total_g4 = sum(counts.values())
        total_len_kb = sum(lengths.values()) / 1000.0
        avg_density = total_g4 / total_len_kb if total_len_kb > 0 else 0.0
        log2_irr = {}
        for comp in counts:
            if densities[comp] > 0 and avg_density > 0:
                log2_irr[comp] = math.log2(densities[comp] / avg_density)
            else:
                log2_irr[comp] = float('nan')
        species_results[species] = {
            'g4_counts': counts,
            'g4_density_per_kb': densities,
            'log2_IRR': log2_irr
        }
    species_pairs = defaultdict(list)
    for p in data['pairs']:
        key = f"{p['species1']}-{p['species2']}"
        species_pairs[key].append(p)
    pair_results = {}
    for key, pairs in species_pairs.items():
        S_G4 = N_G4 = S_nonG4 = N_nonG4 = 0
        X_adj = []
        y_adj = []
        for p in pairs:
            seq1 = p['seq1']
            seq2 = p['seq2']
            sp1 = p['species1']
            cai = compute_cai(seq1, data['codon_usage'][sp1])
            g4_intervals = find_g4_intervals(seq1)
            for i in range(0, len(seq1), 3):
                codon1 = seq1[i:i+3]
                codon2 = seq2[i:i+3]
                if len(codon1) < 3 or len(codon2) < 3:
                    break
                aa1 = codon_to_aa(codon1)
                aa2 = codon_to_aa(codon2)
                if aa1 is None or aa2 is None:
                    continue
                is_g4 = any(not (i+3 <= s or i >= e) for s, e in g4_intervals)
                syn = 1 if aa1 == aa2 else 0
                if is_g4:
                    if syn:
                        S_G4 += 1
                    else:
                        N_G4 += 1
                else:
                    if syn:
                        S_nonG4 += 1
                    else:
                        N_nonG4 += 1
                X_adj.append([1 if is_g4 else 0, cai])
                y_adj.append(syn)
        if N_G4 > 0 and N_nonG4 > 0:
            naive_OR = (S_G4 / N_G4) / (S_nonG4 / N_nonG4)
        else:
            naive_OR = float('nan')
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        clf.fit(X_adj, y_adj)
        adjusted_OR = math.exp(clf.coef_[0][0])
        pair_results[key] = {
            'naive_OR': naive_OR,
            'adjusted_OR': adjusted_OR,
            'n_genes': len(pairs)
        }
    n_naive_lt1 = sum(1 for v in pair_results.values() if v['naive_OR'] < 1)
    n_adj_lt1 = sum(1 for v in pair_results.values() if v['adjusted_OR'] < 1)
    cai_examples = {}
    for p in data['pairs'][:3]:
        cai = compute_cai(p['seq1'], data['codon_usage'][p['species1']])
        cai_examples[p['pair_id']] = cai
    return {
        'species': species_results,
        'species_pairs': pair_results,
        'summary': {
            'n_pairs_naive_OR_lt_1': n_naive_lt1,
            'n_pairs_adjusted_OR_lt_1': n_adj_lt1
        },
        'cai_examples': cai_examples
    }

if __name__ == '__main__':
    gold = compute_gold()
    print(json.dumps(gold, indent=2))
