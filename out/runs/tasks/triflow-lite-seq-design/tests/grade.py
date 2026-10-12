import os
import json
import csv
import math
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

def run_reference(config, target_res, target_coords, binder_coords, potential, ss_prop):
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
    gold_seq = X
    gold_energy = trajectory[-1][2]
    gold_per_pos = []
    for i in range(L):
        E_i = compute_energy_i(gold_seq, i, gold_seq[i], target_res, target_coords, binder_coords, potential, ss, ss_prop, config)
        gold_per_pos.append(E_i)
    return gold_seq, gold_energy, trajectory, gold_per_pos

def main():
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
    assert len(binder_coords) == config['L'], 'Binder length mismatch'
    gold_seq, gold_energy, gold_trajectory, gold_per_pos = run_reference(
        config, target_res, target_coords, binder_coords, potential, ss_prop)
    agent_fasta = '/app/output/designed.fasta'
    agent_traj = '/app/output/trajectory.csv'
    constraints = {}
    # 1. FASTA format
    try:
        with open(agent_fasta, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2, 'FASTA should have 2 lines'
        assert lines[0].startswith('>'), 'Header must start with >'
        seq_str = lines[1]
        assert len(seq_str) == config['L'], f'Sequence length should be {config["L"]}'
        assert all(c in AA_1to3 for c in seq_str), 'Invalid amino acid letters'
        constraints['fasta_format'] = True
    except Exception as e:
        constraints['fasta_format'] = False
        seq_str = None
        print(f'FASTA format error: {e}')
    # 2. Trajectory format
    traj_data = []
    try:
        with open(agent_traj, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ['step', 'sequence', 'energy'], f'Trajectory header mismatch: {header}'
            for row in reader:
                if len(row) == 0:
                    continue
                assert len(row) == 3, 'Trajectory row must have 3 columns'
                step = int(row[0])
                seq = row[1]
                energy = float(row[2])
                assert len(seq) == config['L'], 'Trajectory sequence length mismatch'
                traj_data.append((step, seq, energy))
        assert len(traj_data) == config['n_steps'], f'Trajectory should have {config["n_steps"]} rows'
        assert [r[0] for r in traj_data] == list(range(1, config['n_steps']+1)), 'Step numbers should be 1..n_steps'
        constraints['trajectory_format'] = True
    except Exception as e:
        constraints['trajectory_format'] = False
        print(f'Trajectory format error: {e}')
    # 3. Fixed positions
    try:
        agent_seq_3 = [AA_1to3[c] for c in seq_str]
        fixed = config['fixed_positions']
        for pos_str, aa in fixed.items():
            pos = int(pos_str) - 1
            assert agent_seq_3[pos] == aa, f'Position {pos_str} should be {aa}'
        constraints['fixed_positions'] = True
    except Exception as e:
        constraints['fixed_positions'] = False
        print(f'Fixed positions error: {e}')
    # 4. Sequence identity to gold
    try:
        gold_seq_1 = ''.join([AA_3to1[aa] for aa in gold_seq])
        identity = sum(1 for a, b in zip(seq_str, gold_seq_1) if a == b) / len(seq_str)
        constraints['seq_identity'] = identity >= 0.9
    except Exception as e:
        constraints['seq_identity'] = False
        print(f'Seq identity error: {e}')
    # 5. Energy threshold
    try:
        agent_seq_3 = [AA_1to3[c] for c in seq_str]
        agent_energy = compute_total_energy(agent_seq_3, target_res, target_coords, binder_coords, potential, config['ss_assignment'], ss_prop, config)
        constraints['energy_threshold'] = agent_energy <= gold_energy + 1.0
    except Exception as e:
        constraints['energy_threshold'] = False
        print(f'Energy threshold error: {e}')
    # 6. Per-position energy
    try:
        agent_per_pos = []
        for i in range(config['L']):
            E_i = compute_energy_i(agent_seq_3, i, agent_seq_3[i], target_res, target_coords, binder_coords, potential, config['ss_assignment'], ss_prop, config)
            agent_per_pos.append(E_i)
        max_agent = max(agent_per_pos)
        max_gold = max(gold_per_pos)
        constraints['per_pos_energy'] = max_agent <= max_gold + 1.0
    except Exception as e:
        constraints['per_pos_energy'] = False
        print(f'Per-position energy error: {e}')
    # 7. Hydrophobic fraction
    try:
        hydrophobics = set(['A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y'])
        frac = sum(1 for c in seq_str if c in hydrophobics) / len(seq_str)
        constraints['hydrophobic_fraction'] = 0.25 <= frac <= 0.75
    except Exception as e:
        constraints['hydrophobic_fraction'] = False
        print(f'Hydrophobic fraction error: {e}')
    # 8. Trajectory monotonicity
    try:
        energies = [r[2] for r in traj_data]
        monotonic = all(energies[i] >= energies[i+1] - 0.01 for i in range(len(energies)-1))
        constraints['trajectory_monotonic'] = monotonic
    except Exception as e:
        constraints['trajectory_monotonic'] = False
        print(f'Trajectory monotonic error: {e}')
    # 9. Final sequence match
    try:
        constraints['final_sequence_match'] = traj_data[-1][1] == seq_str
    except Exception as e:
        constraints['final_sequence_match'] = False
        print(f'Final sequence match error: {e}')
    # 10. Initial sequence
    try:
        constraints['initial_sequence'] = traj_data[0][1] == 'CAAAAAAAAAAC'
    except Exception as e:
        constraints['initial_sequence'] = False
        print(f'Initial sequence error: {e}')
    total = len(constraints)
    passed = sum(1 for v in constraints.values() if v)
    score = passed / total
    reward = score
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': score, 'reward': reward}, f)
    print(f'Constraints: {constraints}')
    print(f'Score: {score}')

if __name__ == '__main__':
    main()
