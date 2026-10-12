#!/bin/bash
set -euo pipefail

cd /app

iqtree -s data/rar1_alignment.fasta -m JTT -asr -pre data/rar1_asr -quiet
iqtree -s data/sgt1_alignment.fasta -m JTT -asr -pre data/sgt1_asr -quiet

python3 << 'EOF'
import json
import re
from Bio import AlignIO, Phylo
from io import StringIO

# Load tree
tree = Phylo.read('/app/data/species_tree.nwk', 'newick')
nodes = {clade.name: clade for clade in tree.find_clades() if clade.name}

def parse_asr(filepath):
    states = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                node = parts[0]
                seq = parts[1]
                states[node] = seq
    return states

rar1_asr = parse_asr('/app/data/rar1_asr.state')
sgt1_asr = parse_asr('/app/data/sgt1_asr.state')

rar1_pos = 75
sgt1_pos = [100, 130, 160]

def get_interface_states(node_name):
    rar1_seq = rar1_asr.get(node_name, '')
    sgt1_seq = sgt1_asr.get(node_name, '')
    return {
        'RAR1_75': rar1_seq[rar1_pos-1] if rar1_pos <= len(rar1_seq) else '?',
        'SGT1_100': sgt1_seq[sgt1_pos[0]-1] if sgt1_pos[0] <= len(sgt1_seq) else '?',
        'SGT1_130': sgt1_seq[sgt1_pos[1]-1] if sgt1_pos[1] <= len(sgt1_seq) else '?',
        'SGT1_160': sgt1_seq[sgt1_pos[2]-1] if sgt1_pos[2] <= len(sgt1_seq) else '?'
    }

ancestral_states = {
    'Root': get_interface_states('Root'),
    'FernAncestor': get_interface_states('FernAncestor'),
    'CoreFernAncestor': get_interface_states('CoreFernAncestor')
}

rar1_aln = AlignIO.read('/app/data/rar1_alignment.fasta', 'fasta')
sgt1_aln = AlignIO.read('/app/data/sgt1_alignment.fasta', 'fasta')
rar1_dict = {rec.id: str(rec.seq) for rec in rar1_aln}
sgt1_dict = {rec.id: str(rec.seq) for rec in sgt1_aln}

import csv
assays = []
with open('/app/data/binding_assays.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        assays.append(row)

binding_rule = []
for row in assays:
    if int(row['binding']) == 1:
        binding_rule.append({
            'RAR1_75': row['RAR1_75'],
            'SGT1_100': row['SGT1_100'],
            'SGT1_130': row['SGT1_130'],
            'SGT1_160': row['SGT1_160']
        })
unique_rule = []
seen = set()
for r in binding_rule:
    tup = (r['RAR1_75'], r['SGT1_100'], r['SGT1_130'], r['SGT1_160'])
    if tup not in seen:
        seen.add(tup)
        unique_rule.append(r)
binding_rule = unique_rule

def binds(rar1_75, sgt1_100, sgt1_130, sgt1_160):
    tup = (rar1_75, sgt1_100, sgt1_130, sgt1_160)
    return tup in seen

extant_binding = {}
for sp in rar1_dict:
    rar1_seq = rar1_dict[sp]
    sgt1_seq = sgt1_dict[sp]
    r75 = rar1_seq[rar1_pos-1]
    s100 = sgt1_seq[sgt1_pos[0]-1]
    s130 = sgt1_seq[sgt1_pos[1]-1]
    s160 = sgt1_seq[sgt1_pos[2]-1]
    extant_binding[sp] = binds(r75, s100, s130, s160)

root_st = ancestral_states['Root']
fern_st = ancestral_states['FernAncestor']
core_st = ancestral_states['CoreFernAncestor']
mutations = []
if fern_st['SGT1_130'] != root_st['SGT1_130']:
    mutations.append('SGT1_130')
if core_st['RAR1_75'] != fern_st['RAR1_75']:
    mutations.append('RAR1_75')
if core_st['SGT1_100'] != fern_st['SGT1_100']:
    mutations.append('SGT1_100')
mutation_order = ['SGT1_130', 'RAR1_75', 'SGT1_100']

solution = {
    'ancestral_states': ancestral_states,
    'binding_rule': binding_rule,
    'extant_binding': extant_binding,
    'mutation_order': mutation_order
}

with open('/app/output/solution.json', 'w') as f:
    json.dump(solution, f, indent=2)

print("Solution written.")
EOF
