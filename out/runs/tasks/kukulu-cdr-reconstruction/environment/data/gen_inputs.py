import os, random, math
import numpy as np

SEED = 20260803
random.seed(SEED)
np.random.seed(SEED)

AA3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

def random_walk(n, step=3.8):
    pts = [np.array([0.,0.,0.])]
    for i in range(1, n):
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
        pts.append(pts[-1] + step * v)
    return pts

def add_atoms(ca_coords, resnames):
    n = len(ca_coords)
    residues = []
    for i in range(n):
        ca = ca_coords[i]
        if i < n-1:
            d = ca_coords[i+1] - ca
        else:
            d = ca - ca_coords[i-1]
        d = d / np.linalg.norm(d)
        perp = np.cross(d, [0,0,1])
        if np.linalg.norm(perp) < 0.1:
            perp = np.cross(d, [0,1,0])
        perp = perp / np.linalg.norm(perp)
        N = ca - 0.5*d + 0.3*perp
        C = ca + 0.5*d - 0.3*perp
        O = C + 0.6*perp
        CB = ca + 0.7*perp
        residues.append({'resname': resnames[i], 'N': N, 'CA': ca, 'C': C, 'O': O, 'CB': CB})
    return residues

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    comp_dir = os.path.join(base, 'complexes')
    frag_dir = os.path.join(base, 'fragments')
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(frag_dir, exist_ok=True)
    loop_lengths = [6,7,8,9,10,11,12,8]
    for idx, L in enumerate(loop_lengths):
        comp_id = f"complex_{idx+1:02d}"
        heavy_len = 120
        ca_h = random_walk(heavy_len)
        resnames_h = [random.choice(AA3) for _ in range(heavy_len)]
        ca_l = random_walk(100)
        resnames_l = [random.choice(AA3) for _ in range(100)]
        ca_a = random_walk(80)
        resnames_a = [random.choice(AA3) for _ in range(80)]
        ca_h = ca_h + np.array([0., 0., 0.])
        ca_l = ca_l + np.array([10., 10., 0.])
        ca_a = ca_a + np.array([5., 20., 10.])
        res_h = add_atoms(ca_h, resnames_h)
        res_l = add_atoms(ca_l, resnames_l)
        res_a = add_atoms(ca_a, resnames_a)
        masked_h = res_h[:94] + res_h[94+L:]
        out_path = os.path.join(comp_dir, f"{comp_id}.pdb")
        with open(out_path, 'w') as f:
            atom_serial = 1
            for i, res in enumerate(masked_h):
                if i < 94:
                    resnum = i + 1
                else:
                    resnum = i + 1 + L
                for atom_name in ['N', 'CA', 'C', 'O', 'CB']:
                    x, y, z = res[atom_name]
                    line = f"ATOM  {atom_serial:5d} {atom_name:^4s} {res['resname']:>3s} H{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n"
                    f.write(line)
                    atom_serial += 1
            for i, res in enumerate(res_l):
                resnum = i + 1
                for atom_name in ['N', 'CA', 'C', 'O', 'CB']:
                    x, y, z = res[atom_name]
                    line = f"ATOM  {atom_serial:5d} {atom_name:^4s} {res['resname']:>3s} L{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n"
                    f.write(line)
                    atom_serial += 1
            for i, res in enumerate(res_a):
                resnum = i + 1
                for atom_name in ['N', 'CA', 'C', 'O', 'CB']:
                    x, y, z = res[atom_name]
                    line = f"ATOM  {atom_serial:5d} {atom_name:^4s} {res['resname']:>3s} A{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n"
                    f.write(line)
                    atom_serial += 1
            f.write("END\n")
    for i in range(200):
        L = random.randint(6, 12)
        ca = random_walk(L)
        resnames = [random.choice(AA3) for _ in range(L)]
        res = add_atoms(ca, resnames)
        out_path = os.path.join(frag_dir, f"frag_{i+1:04d}.pdb")
        with open(out_path, 'w') as f:
            atom_serial = 1
            for j, r in enumerate(res):
                resnum = j + 1
                for atom_name in ['N', 'CA', 'C', 'O', 'CB']:
                    x, y, z = r[atom_name]
                    line = f"ATOM  {atom_serial:5d} {atom_name:^4s} {r['resname']:>3s} A{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n"
                    f.write(line)
                    atom_serial += 1
            f.write("END\n")
    print("Generated complexes and fragments.")

if __name__ == '__main__':
    main()
