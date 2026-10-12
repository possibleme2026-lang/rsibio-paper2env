#!/bin/bash
set -euo pipefail

mkdir -p /app/output

python3 - <<'PY'
import json
import numpy as np
from collections import defaultdict

probes = []
with open('/app/data/probes.tsv') as f:
    next(f)
    for line in f:
        pid, sample, seq = line.strip().split('\t')
        probes.append((pid, sample, seq))

cell_barcodes = []
with open('/app/data/cell_barcodes.txt') as f:
    for line in f:
        cell_barcodes.append(line.strip())

reads = []
with open('/app/data/reads.fastq') as f:
    while True:
        header = f.readline()
        if not header:
            break
        seq = f.readline().strip()
        f.readline()
        f.readline()
        reads.append(seq)

base_to_int = {'A':0, 'C':1, 'G':2, 'T':3, 'N':4}
def seq_to_arr(s):
    return np.array([base_to_int.get(c, 4) for c in s], dtype=np.int8)

def revcomp(s):
    comp = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
    return ''.join(comp.get(c, 'N') for c in reversed(s))

probe_ids = [p[0] for p in probes]
sample_ids = [p[1] for p in probes]
probe_arr = np.array([seq_to_arr(p[2]) for p in probes])
probe_rc_arr = np.array([seq_to_arr(revcomp(p[2])) for p in probes])
bc_arr = np.array([seq_to_arr(bc) for bc in cell_barcodes])

molecules = set()
assigned = 0
ambiguous = 0
unassigned = 0
invalid_bc = 0
corrected_bc = 0

for read in reads:
    bc = read[0:16]
    umi = read[16:28]
    probe_seq = read[28:68]
    
    bc_read_arr = seq_to_arr(bc)
    diffs = (bc_arr != bc_read_arr).sum(axis=1)
    min_d = diffs.min()
    if min_d == 0:
        idx = np.where(diffs == 0)[0]
        if len(idx) != 1:
            invalid_bc += 1
            continue
        corrected = cell_barcodes[idx[0]]
    elif min_d == 1:
        idx = np.where(diffs == 1)[0]
        if len(idx) != 1:
            invalid_bc += 1
            continue
        corrected = cell_barcodes[idx[0]]
        corrected_bc += 1
    else:
        invalid_bc += 1
        continue
    
    probe_read_arr = seq_to_arr(probe_seq)
    dist_fwd = (probe_arr != probe_read_arr).sum(axis=1)
    dist_rev = (probe_rc_arr != probe_read_arr).sum(axis=1)
    min_fwd = dist_fwd.min()
    min_rev = dist_rev.min()
    global_min = min(min_fwd, min_rev)
    
    if global_min > 2:
        unassigned += 1
        continue
    
    cands = set()
    if min_fwd == global_min:
        for i in np.where(dist_fwd == global_min)[0]:
            cands.add(i)
    if min_rev == global_min:
        for i in np.where(dist_rev == global_min)[0]:
            cands.add(i)
    
    if len(cands) != 1:
        ambiguous += 1
        continue
    
    idx = cands.pop()
    assigned += 1
    molecules.add((sample_ids[idx], probe_ids[idx], corrected, umi))

counts = defaultdict(int)
for (sample, probe, bc, umi) in molecules:
    counts[(probe, sample)] += 1

with open('/app/output/probe_counts.tsv', 'w') as f:
    f.write('probe_id\tsample_id\tumi_count\n')
    all_probes = sorted(set(p[0] for p in probes))
    all_samples = sorted(set(p[1] for p in probes))
    for pid in all_probes:
        for sid in all_samples:
            f.write(f'{pid}\t{sid}\t{counts.get((pid, sid), 0)}\n')

summary = {
    "total_reads": len(reads),
    "assigned_reads": assigned,
    "ambiguous_probe_reads": ambiguous,
    "unassigned_probe_reads": unassigned,
    "invalid_cell_barcode_reads": invalid_bc,
    "corrected_cell_barcodes": corrected_bc,
    "unique_umis_total": len(molecules)
}
with open('/app/output/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
PY
