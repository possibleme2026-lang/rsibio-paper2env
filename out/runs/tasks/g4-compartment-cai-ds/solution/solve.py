import os, re, math, json
from collections import defaultdict

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

# Read FASTA
def read_fasta(path):
    seqs = {}
    name = None
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name:
                    seqs[name] = ''.join(seq)
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name:
            seqs[name] = ''.join(seq)
    return seqs

# Read GFF3
def read_gff(path):
    genes = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('	')
            if len(parts) < 9:
                continue
            chrom, source, typ, start, end, score, strand, phase, attrs = parts
            start = int(start)
            end = int(end)
            if typ == 'gene':
                genes.append({'chrom': chrom, 'start': start, 'end': end, 'strand': strand, 'cds': [], 'introns': []})
            elif typ == 'CDS':
                for g in reversed(genes):
                    if g['chrom'] == chrom and g['strand'] == strand:
                        g['cds'].append((start-1, end))
                        break
            elif typ == 'intron':
                for g in reversed(genes):
                    if g['chrom'] == chrom and g['strand'] == strand:
                        g['introns'].append((start-1, end))
                        break
    return genes

# Read data
genome_dir = '/app/data/genomes'
fa_files = [f for f in os.listdir(genome_dir) if f.endswith('.fa')]
species_genomes = {}
for f in fa_files:
    sp = f.replace('.fa', '')
    species_genomes[sp] = read_fasta(os.path.join(genome_dir, f))

genes = read_gff('/app/data/annotation.gff3')

# Read codon usage
codon_usage = defaultdict(dict)
with open('/app/data/codon_usage.tsv') as f:
    next(f)
    for line in f:
        sp, codon, rscu = line.strip().split('	')
        codon_usage[sp][codon] = float(rscu)

# Read ortholog pairs
pairs = []
with open('/app/data/ortholog_pairs.tsv') as f:
    next(f)
    for line in f:
        pair_id, sp1, sp2, file = line.strip().split('	')
        seqs = read_fasta(os.path.join('/app/data/orthologs', file))
        keys = list(seqs.keys())
        seq1 = seqs[keys[0]]
        seq2 = seqs[keys[1]]
        pairs.append({'pair_id': pair_id, 'species1': sp1, 'species2': sp2, 'seq1': seq1, 'seq2': seq2})

# Compute species results
species_results = {}
for species, chroms in species_genomes.items():
    sp_genes = [g for g in genes if g['chrom'].startswith(species)]
    cds_intervals = defaultdict(list)
    intron_intervals = defaultdict(list)
    promoter_intervals = defaultdict(list)
    for g in sp_genes:
        chrom = g['chrom']
        for s, e in g['cds']:
            cds_intervals[chrom].append((s, e))
        for s, e in g['introns']:
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
    g4_by_chrom = {c: find_g4_intervals(seq) for c, seq in chroms.items()}
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
        densities[comp] = counts[comp] / (lengths[comp] / 1000.0) if lengths[comp] > 0 else 0.0
    total_g4 = sum(counts.values())
    total_len_kb = sum(lengths.values()) / 1000.0
    avg_density = total_g4 / total_len_kb if total_len_kb > 0 else 0.0
    log2_irr = {}
    for comp in counts:
        if densities[comp] > 0 and avg_density > 0:
            log2_irr[comp] = math.log2(densities[comp] / avg_density)
        else:
            log2_irr[comp] = float('nan')
    species_results[species] = {'g4_counts': counts, 'g4_density_per_kb': densities, 'log2_IRR': log2_irr}

# Ortholog pairs analysis
species_pairs = defaultdict(list)
for p in pairs:
    key = f"{p['species1']}-{p['species2']}"
    species_pairs[key].append(p)
pair_results = {}
for key, plist in species_pairs.items():
    S_G4 = N_G4 = S_nonG4 = N_nonG4 = 0
    X_adj = []
    y_adj = []
    for p in plist:
        seq1 = p['seq1']
        seq2 = p['seq2']
        sp1 = p['species1']
        cai = compute_cai(seq1, codon_usage[sp1])
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
                if syn: S_G4 += 1
                else: N_G4 += 1
            else:
                if syn: S_nonG4 += 1
                else: N_nonG4 += 1
            X_adj.append([1 if is_g4 else 0, cai])
            y_adj.append(syn)
    naive_OR = (S_G4 / N_G4) / (S_nonG4 / N_nonG4) if N_G4 > 0 and N_nonG4 > 0 else float('nan')
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    clf.fit(X_adj, y_adj)
    adjusted_OR = math.exp(clf.coef_[0][0])
    pair_results[key] = {'naive_OR': naive_OR, 'adjusted_OR': adjusted_OR, 'n_genes': len(plist)}

n_naive_lt1 = sum(1 for v in pair_results.values() if v['naive_OR'] < 1)
n_adj_lt1 = sum(1 for v in pair_results.values() if v['adjusted_OR'] < 1)
cai_examples = {}
for p in pairs[:3]:
    cai = compute_cai(p['seq1'], codon_usage[p['species1']])
    cai_examples[p['pair_id']] = cai

result = {
    'species': species_results,
    'species_pairs': pair_results,
    'summary': {'n_pairs_naive_OR_lt_1': n_naive_lt1, 'n_pairs_adjusted_OR_lt_1': n_adj_lt1},
    'cai_examples': cai_examples
}
os.makedirs('/app/output', exist_ok=True)
with open('/app/output/result.json', 'w') as f:
    json.dump(result, f, indent=2)
