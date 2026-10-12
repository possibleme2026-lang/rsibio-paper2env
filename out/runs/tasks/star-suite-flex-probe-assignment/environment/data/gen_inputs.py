import random
import os
from collections import defaultdict

random.seed(42)
BASES = 'ACGT'

def random_seq(n):
    return ''.join(random.choice(BASES) for _ in range(n))

def revcomp(s):
    comp = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
    return ''.join(comp[c] for c in reversed(s))

def mutate(s, n):
    s = list(s)
    for _ in range(n):
        i = random.randrange(len(s))
        orig = s[i]
        choices = [b for b in BASES if b != orig]
        s[i] = random.choice(choices)
    return ''.join(s)

def get_neighbors(bc):
    neighbors = []
    for i in range(len(bc)):
        for base in BASES:
            if base != bc[i]:
                neighbors.append(bc[:i] + base + bc[i+1:])
    return neighbors

samples = ['SAMPLE_A', 'SAMPLE_B', 'SAMPLE_C']
probes = []
for i in range(600):
    sample = samples[i // 200]
    pid = f"PRB_{i+1:04d}"
    seq = random_seq(40)
    probes.append([pid, sample, seq])

close_pairs = []
for _ in range(15):
    a, b = random.sample(range(600), 2)
    probes[b][2] = mutate(probes[a][2], 1)
    close_pairs.append((a, b))

cell_barcodes = list(set(random_seq(16) for _ in range(2200)))[:2000]
while len(cell_barcodes) < 2000:
    cell_barcodes.append(random_seq(16))
cell_barcodes = cell_barcodes[:2000]

neighbor_map = defaultdict(list)
for bc in cell_barcodes:
    for nb in get_neighbors(bc):
        neighbor_map[nb].append(bc)

with open('probes.tsv', 'w') as f:
    f.write('probe_id\tsample_id\tprobe_seq\n')
    for pid, sample, seq in probes:
        f.write(f'{pid}\t{sample}\t{seq}\n')

with open('cell_barcodes.txt', 'w') as f:
    for bc in cell_barcodes:
        f.write(f'{bc}\n')

valid_molecules = []
for _ in range(18000):
    probe_idx = random.randrange(600)
    bc = random.choice(cell_barcodes)
    umi = random_seq(12)
    valid_molecules.append((probe_idx, bc, umi))

corrected_molecules = []
for _ in range(1000):
    probe_idx = random.randrange(600)
    while True:
        orig_bc = random.choice(cell_barcodes)
        bc = mutate(orig_bc, 1)
        if bc not in cell_barcodes and len(neighbor_map.get(bc, [])) == 1:
            break
    umi = random_seq(12)
    corrected_molecules.append((probe_idx, bc, umi))

read_specs = []

for probe_idx, bc, umi in valid_molecules:
    orientation = 'fwd' if random.random() < 0.8 else 'rev'
    n_mut = random.choices([0,1,2], weights=[0.7,0.25,0.05])[0]
    read_specs.append(('valid', probe_idx, bc, umi, orientation, n_mut, None))

for _ in range(2000):
    probe_idx, bc, umi = random.choice(valid_molecules)
    orientation = 'fwd' if random.random() < 0.8 else 'rev'
    n_mut = random.choices([0,1,2], weights=[0.7,0.25,0.05])[0]
    read_specs.append(('valid', probe_idx, bc, umi, orientation, n_mut, None))

for probe_idx, bc, umi in corrected_molecules:
    orientation = 'fwd' if random.random() < 0.8 else 'rev'
    n_mut = random.choices([0,1,2], weights=[0.7,0.25,0.05])[0]
    read_specs.append(('corrected_bc', probe_idx, bc, umi, orientation, n_mut, None))

for _ in range(3000):
    probe_idx = random.randrange(600)
    while True:
        orig_bc = random.choice(cell_barcodes)
        bc = mutate(orig_bc, 2)
        if bc not in cell_barcodes and bc not in neighbor_map:
            break
    umi = random_seq(12)
    orientation = 'fwd' if random.random() < 0.8 else 'rev'
    n_mut = random.choices([0,1,2], weights=[0.7,0.25,0.05])[0]
    read_specs.append(('invalid_bc', probe_idx, bc, umi, orientation, n_mut, None))

for _ in range(3000):
    a, b = random.choice(close_pairs)
    seq_a = probes[a][2]
    seq_b = probes[b][2]
    diff_pos = [i for i in range(40) if seq_a[i] != seq_b[i]]
    if len(diff_pos) == 1:
        p = diff_pos[0]
        orig_a = seq_a[p]
        orig_b = seq_b[p]
        choices = [base for base in BASES if base != orig_a and base != orig_b]
        if choices:
            new_base = random.choice(choices)
            probe_seq = seq_a[:p] + new_base + seq_a[p+1:]
        else:
            probe_seq = mutate(seq_a, 1)
    else:
        probe_seq = mutate(seq_a, 1)
    bc = random.choice(cell_barcodes)
    umi = random_seq(12)
    read_specs.append(('ambiguous', None, bc, umi, 'fwd', 0, probe_seq))

for _ in range(3000):
    probe_seq = random_seq(40)
    bc = random.choice(cell_barcodes)
    umi = random_seq(12)
    read_specs.append(('unassigned', None, bc, umi, 'fwd', 0, probe_seq))

random.shuffle(read_specs)

with open('reads.fastq', 'w') as f:
    for i, spec in enumerate(read_specs):
        read_type, probe_idx, bc, umi, orientation, n_mut, custom_probe_seq = spec
        if custom_probe_seq is not None:
            probe_seq = custom_probe_seq
        else:
            probe_seq = probes[probe_idx][2]
            if orientation == 'rev':
                probe_seq = revcomp(probe_seq)
            if n_mut > 0:
                probe_seq = mutate(probe_seq, n_mut)
        transcript = random_seq(32)
        read_seq = bc + umi + probe_seq + transcript
        read_seq = read_seq[:100]
        qual = 'I' * len(read_seq)
        f.write(f'@read_{i+1}\n{read_seq}\n+\n{qual}\n')

print("Generated probes.tsv, cell_barcodes.txt, reads.fastq")
