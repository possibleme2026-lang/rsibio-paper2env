#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import json, csv, math, os
import numpy as np

AA_ORDER = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
AA_1to3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}
AA_3to1 = {v: k for k, v in AA_1to3.items()}

def parse_pdb_ca(filename):
    residues = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                res_name = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                residues.append((res_name, np.array([x, y, z])))
    return residues

def dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def compute_energy_i(seq, i, a, target_res, target_coords, binder_coords, potential, ss, ss_prop, config):
    L = len(seq)
    E = 0.0
    for j, (t_res, t_coord) in enumerate(zip(target_res, target_coords)):
        d = dist(binder_coords[i], t_coord)
        if d < config['cutoff']:
            E += potential[a][t_res] * math.exp(-(d / config['kernel_decay'])**2)
    for k in range(L):
        if k == i:
            continue
        d = dist(binder_coords[i], binder_coords[k])
        if d < config['cutoff']:
            E += potential[a][seq[k]] * math.exp(-(d / config['kernel_decay'])**2)
    E -= ss_prop[a][ss[i]] * config['ss_penalty_weight']
    return E

def compute_total_energy(seq, target_res, target_coords, binder_coords, potential, ss, ss_prop, config):
    L = len(seq)
    E = 0.0
    for i in range(L):
        for j, (t_res, t_coord) in enumerate(zip(target_res, target_coords)):
            d = dist(binder_coords[i], t_coord)
            if d < config['cutoff']:
                E += potential[seq[i]][t_res] * math.exp(-(d / config['kernel_decay'])**2)
        E -= ss_prop[seq[i]][ss[i]] * config['ss_penalty_weight']
    for i in range(L):
        for k in range(i+1, L):
            d = dist(binder_coords[i], binder_coords[k])
            if d < config['cutoff']:
                E += potential[seq[i]][seq[k]] * math.exp(-(d / config['kernel_decay'])**2)
    return E

with open('/app/data/config.json', 'r') as f:
    config = json.load(f)
potential = {}
with open('/app/data/potential.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)[1:]
    for row in reader:
        aa = row[0]
        potential[aa] = {}
        for j, aa2 in enumerate(header):
            potential[aa][aa2] = float(row[j+1])
ss_prop = {}
with open('/app/data/ss_propensity.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        aa = row[0]
        ss_prop[aa] = {'H': float(row[1]), 'E': float(row[2]), 'C': float(row[3])}
target_res, target_coords = zip(*[(r, c) for r, c in parse_pdb_ca('/app/data/target.pdb')])
binder_res, binder_coords = zip(*[(r, c) for r, c in parse_pdb_ca('/app/data/binder.pdb')])
target_res = list(target_res)
target_coords = list(target_coords)
binder_coords = list(binder_coords)
L = config['L']
n_steps = config['n_steps']
T_start = config['T_start']
T_end = config['T_end']
ss = config['ss_assignment']
fixed = config['fixed_positions']
q = {}
for i in range(L):
    q[i] = {aa: 1.0/20.0 for aa in AA_ORDER}
for i_str, aa in fixed.items():
    i = int(i_str) - 1
    for a in AA_ORDER:
        q[i][a] = 1.0 if a == aa else 0.0
trajectory = []
for t in range(1, n_steps+1):
    X = []
    for i in range(L):
        best_aa = max(AA_ORDER, key=lambda a: (q[i][a], -AA_ORDER.index(a)))
        X.append(best_aa)
    H = {}
    for i in range(L):
        H[i] = {}
        for a in AA_ORDER:
            H[i][a] = compute_energy_i(X, i, a, target_res, target_coords, binder_coords, potential, ss, ss_prop, config)
    if n_steps == 1:
        T = T_start
    else:
        T = T_start * (T_end / T_start) ** ((t-1) / (n_steps-1))
    for i in range(L):
        logits = {}
        for a in AA_ORDER:
            logits[a] = -H[i][a] / T + math.log(q[i][a] + 1e-12)
        max_logit = max(logits.values())
        for a in AA_ORDER:
            q[i][a] = math.exp(logits[a] - max_logit)
        total = sum(q[i].values())
        for a in AA_ORDER:
            q[i][a] /= total
        if str(i+1) in fixed:
            fixed_aa = fixed[str(i+1)]
            for a in AA_ORDER:
                q[i][a] = 1.0 if a == fixed_aa else 0.0
    E_total = compute_total_energy(X, target_res, target_coords, binder_coords, potential, ss, ss_prop, config)
    X_1letter = ''.join([AA_3to1[aa] for aa in X])
    trajectory.append((t, X_1letter, E_total))
os.makedirs('/app/output', exist_ok=True)
with open('/app/output/designed.fasta', 'w') as f:
    f.write('>designed_binder\n')
    f.write(trajectory[-1][1] + '\n')
with open('/app/output/trajectory.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'sequence', 'energy'])
    for row in trajectory:
        writer.writerow([row[0], row[1], f'{row[2]:.6f}'])
PY
