import numpy as np
import json
import csv

AA_1to3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}
AA_3to1 = {v: k for k, v in AA_1to3.items()}
AA_order = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

SS_PROP = {
    'ALA': [1.42, 0.83, 1.00], 'ARG': [0.98, 0.93, 1.00], 'ASN': [0.67, 0.89, 1.00],
    'ASP': [1.01, 0.54, 1.00], 'CYS': [0.70, 1.19, 1.00], 'GLN': [1.11, 1.10, 1.00],
    'GLU': [1.51, 0.37, 1.00], 'GLY': [0.57, 0.75, 1.00], 'HIS': [1.00, 0.87, 1.00],
    'ILE': [1.08, 1.60, 1.00], 'LEU': [1.21, 1.30, 1.00], 'LYS': [1.16, 0.74, 1.00],
    'MET': [1.45, 1.05, 1.00], 'PHE': [1.13, 1.38, 1.00], 'PRO': [0.57, 0.55, 1.00],
    'SER': [0.77, 0.75, 1.00], 'THR': [0.83, 1.19, 1.00], 'TRP': [1.08, 1.37, 1.00],
    'TYR': [0.69, 1.47, 1.00], 'VAL': [1.06, 1.70, 1.00]
}

HYDRO = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

def generate_potential(seed=42):
    np.random.seed(seed)
    n = len(AA_order)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            a1 = AA_order[i]
            a2 = AA_order[j]
            h1 = HYDRO[a1]
            h2 = HYDRO[a2]
            if h1 > 1.0 and h2 > 1.0:
                base = -1.0
            elif h1 < -1.0 and h2 < -1.0:
                pos1 = a1 in ['ARG', 'LYS']
                pos2 = a2 in ['ARG', 'LYS']
                neg1 = a1 in ['ASP', 'GLU']
                neg2 = a2 in ['ASP', 'GLU']
                if (pos1 and pos2) or (neg1 and neg2):
                    base = 1.0
                elif (pos1 and neg2) or (neg1 and pos2):
                    base = -1.5
                else:
                    base = 0.0
            else:
                base = 0.0
            noise = np.random.normal(0, 0.3)
            val = base + noise
            mat[i, j] = val
            mat[j, i] = val
    return mat

def compute_3d(a, b, c, bond, angle, dihedral):
    bc = c - b
    bc = bc / np.linalg.norm(bc)
    if abs(bc[2]) < 0.9:
        n = np.cross(bc, np.array([0,0,1]))
    else:
        n = np.cross(bc, np.array([0,1,0]))
    n = n / np.linalg.norm(n)
    m = np.cross(bc, n)
    angle_rad = np.radians(angle)
    dihedral_rad = np.radians(dihedral)
    d = c + bond * (np.cos(angle_rad)*bc + np.sin(angle_rad)*np.cos(dihedral_rad)*n + np.sin(angle_rad)*np.sin(dihedral_rad)*m)
    return d

def generate_backbone_nerf(seq_3, phi_psi, start_coords=None):
    atoms = []
    n = len(seq_3)
    if start_coords is None:
        N0 = np.array([0.0, 0.0, 0.0])
        CA0 = N0 + np.array([1.46, 0.0, 0.0])
        angle_NCA_C = np.radians(111.0)
        C0 = CA0 + 1.52 * np.array([np.cos(np.pi - angle_NCA_C), np.sin(np.pi - angle_NCA_C), 0.0])
        O0 = compute_3d(N0, CA0, C0, 1.23, 120.0, 180.0)
        CB0 = compute_3d(C0, N0, CA0, 1.53, 110.0, -122.5)
        start_coords = (N0, CA0, C0, O0, CB0)
    N_prev, CA_prev, C_prev, O_prev, CB_prev = start_coords
    atoms.append(('N', seq_3[0], 1, *N_prev))
    atoms.append(('CA', seq_3[0], 1, *CA_prev))
    atoms.append(('C', seq_3[0], 1, *C_prev))
    atoms.append(('O', seq_3[0], 1, *O_prev))
    atoms.append(('CB', seq_3[0], 1, *CB_prev))
    for i in range(1, n):
        phi, psi = phi_psi[i-1]
        N_i = compute_3d(N_prev, CA_prev, C_prev, 1.33, 116.0, psi)
        CA_i = compute_3d(CA_prev, C_prev, N_i, 1.46, 121.0, 180.0)
        C_i = compute_3d(C_prev, N_i, CA_i, 1.52, 111.0, phi)
        O_i = compute_3d(N_i, CA_i, C_i, 1.23, 120.0, 180.0)
        CB_i = compute_3d(C_i, N_i, CA_i, 1.53, 110.0, -122.5)
        atoms.append(('N', seq_3[i], i+1, *N_i))
        atoms.append(('CA', seq_3[i], i+1, *CA_i))
        atoms.append(('C', seq_3[i], i+1, *C_i))
        atoms.append(('O', seq_3[i], i+1, *O_i))
        atoms.append(('CB', seq_3[i], i+1, *CB_i))
        N_prev, CA_prev, C_prev = N_i, CA_i, C_i
    return atoms

def write_pdb(atoms, filename, chain='A'):
    with open(filename, 'w') as f:
        for serial, atom in enumerate(atoms, 1):
            name, resname, resid, x, y, z = atom
            f.write(f"ATOM  {serial:5d} {name:>4s} {resname:>3s} {chain}{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]:>2s}\n")
        f.write('END\n')

def main():
    target_seq = ['ALA'] * 80
    phi_psi_target = []
    for i in range(40):
        phi_psi_target.append((-57.0, -47.0))
    for i in range(40):
        phi_psi_target.append((-139.0, 135.0))
    target_atoms = generate_backbone_nerf(target_seq, phi_psi_target)
    write_pdb(target_atoms, 'target.pdb', 'A')
    binder_seq = ['ALA'] * 12
    phi_psi_binder = [(-57.0, -47.0)] * 12
    binder_atoms = generate_backbone_nerf(binder_seq, phi_psi_binder)
    target_ca = [(a[3], a[4], a[5]) for a in target_atoms if a[0] == 'CA' and 10 <= a[2] <= 30]
    centroid_target = np.mean(target_ca, axis=0)
    binder_ca = [(a[3], a[4], a[5]) for a in binder_atoms if a[0] == 'CA']
    centroid_binder = np.mean(binder_ca, axis=0)
    translation = centroid_target + np.array([10.0, 0.0, 0.0]) - centroid_binder
    new_binder_atoms = []
    for atom in binder_atoms:
        name, resname, resid, x, y, z = atom
        new_binder_atoms.append((name, resname, resid, x + translation[0], y + translation[1], z + translation[2]))
    write_pdb(new_binder_atoms, 'binder.pdb', 'B')
    pot = generate_potential()
    with open('potential.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + AA_order)
        for i, aa in enumerate(AA_order):
            writer.writerow([aa] + [f'{pot[i,j]:.4f}' for j in range(len(AA_order))])
    with open('ss_propensity.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['aa', 'H', 'E', 'C'])
        for aa in AA_order:
            writer.writerow([aa] + [f'{SS_PROP[aa][0]:.4f}', f'{SS_PROP[aa][1]:.4f}', f'{SS_PROP[aa][2]:.4f}'])
    config = {
        'L': 12,
        'seq_init': 'AAAAAAAAAAAA',
        'fixed_positions': {'1': 'CYS', '12': 'CYS'},
        'T_start': 2.0,
        'T_end': 0.1,
        'n_steps': 20,
        'cutoff': 10.0,
        'kernel_decay': 8.0,
        'ss_penalty_weight': 0.5,
        'ss_assignment': ['H'] * 12
    }
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    with open('aa_mapping.json', 'w') as f:
        json.dump(AA_1to3, f, indent=2)

if __name__ == '__main__':
    main()
