#!/bin/bash
set -euo pipefail
mkdir -p /app/output /app/work

for pop in S R1 R2; do
    minimap2 -ax map-ont /app/data/reference.fasta /app/data/reads/${pop}.fastq.gz | \
    samtools sort -o /app/work/${pop}.bam
    samtools index /app/work/${pop}.bam
done

python3 << 'PY'
import pysam, json
import primer3

ref = pysam.FastaFile('/app/data/reference.fasta')
ref_seq = ref.fetch('PPO2_CDS')
mut_idx = 292

def rev_comp(seq):
    comp = {'A':'T','T':'A','C':'G','G':'C','N':'N'}
    return ''.join(comp.get(b,b) for b in reversed(seq))

pop_data = {}
for pop in ['S','R1','R2']:
    bam = pysam.AlignmentFile(f'/app/work/{pop}.bam', 'rb')
    counts = {'A':0, 'C':0, 'G':0, 'T':0}
    for read in bam:
        if read.is_unmapped:
            continue
        for qpos, rpos in read.get_aligned_pairs():
            if rpos == mut_idx and qpos is not None:
                base = read.query_sequence[qpos].upper()
                if base in counts:
                    counts[base] += 1
                break
    total = sum(counts.values())
    if total == 0:
        freqs = {'WT_R98':0.0, 'R98L':0.0, 'R98Q':0.0}
    else:
        freqs = {
            'WT_R98': counts['G'] / total,
            'R98L': counts['T'] / total,
            'R98Q': counts['A'] / total
        }
    s = sum(freqs.values())
    if s > 0:
        freqs = {k: v/s for k,v in freqs.items()}
    carries_R98L = freqs['R98L'] > 0.05
    carries_R98Q = freqs['R98Q'] > 0.05
    resistant = carries_R98L or carries_R98Q
    pop_data[pop] = {
        'variant_frequencies': freqs,
        'carries_R98L': carries_R98L,
        'carries_R98Q': carries_R98Q,
        'resistant': resistant
    }

def tm(seq): return primer3.calc_tm(seq)
def gc(seq): return (seq.count('G')+seq.count('C'))/len(seq)*100
def ok(seq):
    import re
    return 58 <= tm(seq) <= 62 and 40 <= gc(seq) <= 60 and not re.search(r'(.)\1{3,}', seq)

fwd = None
for L in range(18, 26):
    start = mut_idx - L + 1
    if start < 0: continue
    seq = ref_seq[start:mut_idx] + 'A'
    if ok(seq):
        fwd = seq
        break
if fwd is None:
    fwd = ref_seq[mut_idx-19:mut_idx] + 'A'

rev = None
for r_start in range(mut_idx+20, 550):
    for L in range(18, 26):
        if r_start + L > len(ref_seq): break
        seq = rev_comp(ref_seq[r_start:r_start+L])
        if ok(seq):
            product_size = (r_start + L - 1) - (mut_idx - len(fwd) + 1) + 1
            if 150 <= product_size <= 300:
                rev = seq
                break
    if rev: break
if rev is None:
    rev = rev_comp(ref_seq[400:422])

product_size = (ref_seq.find(rev_comp(rev)) + len(rev) - 1) - (mut_idx - len(fwd) + 1) + 1

result = {
    'populations': pop_data,
    'diagnostic_primers': {
        'forward': fwd,
        'reverse': rev,
        'product_size': product_size,
        'target_allele': 'R98Q',
        'primer_with_3p_mutation': 'forward'
    }
}
with open('/app/output/result.json', 'w') as f:
    json.dump(result, f, indent=2)
print('Done')
PY
