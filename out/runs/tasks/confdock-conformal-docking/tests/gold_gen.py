import os, random
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

def generate_ligand(num_atoms, seed):
    rng = random.Random(seed)
    mol = Chem.RWMol()
    elements = ['C']*6 + ['N']*2 + ['O']*2 + ['S']
    for i in range(num_atoms):
        el = rng.choice(elements)
        atom = Chem.Atom(el)
        mol.AddAtom(atom)
    for i in range(1, num_atoms):
        mol.AddBond(i-1, i, Chem.BondType.SINGLE)
    _maxdeg = {'C': 4, 'N': 3, 'O': 2, 'S': 2}
    for _ in range(num_atoms // 6):
        i, j = rng.sample(range(num_atoms), 2)
        if mol.GetBondBetweenAtoms(i, j):
            continue
        ei = mol.GetAtomWithIdx(i).GetSymbol()
        ej = mol.GetAtomWithIdx(j).GetSymbol()
        if mol.GetAtomWithIdx(i).GetDegree() >= _maxdeg[ei]:
            continue
        if mol.GetAtomWithIdx(j).GetDegree() >= _maxdeg[ej]:
            continue
        mol.AddBond(i, j, Chem.BondType.SINGLE)
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    return mol

def generate_protein(ligand_coords, num_atoms, seed):
    rng = np.random.default_rng(seed)
    center = np.mean(ligand_coords, axis=0)
    protein_coords = []
    while len(protein_coords) < num_atoms:
        r = 15 * np.cbrt(rng.random())
        theta = 2 * np.pi * rng.random()
        phi = np.arccos(2 * rng.random() - 1)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        pos = center + np.array([x, y, z])
        if np.min(np.linalg.norm(ligand_coords - pos, axis=1)) > 2.5:
            protein_coords.append(pos)
    return np.array(protein_coords)

def write_pdb(coords, path, seed):
    rng = np.random.default_rng(seed)
    with open(path, 'w') as f:
        for i, (x, y, z) in enumerate(coords):
            el = rng.choice(['C', 'N', 'O'])
            b = rng.uniform(10, 50)
            f.write(f"ATOM  {i+1:5d} {el:>2s}   MOL     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}          {el:>2s}\n")
        f.write("END\n")

def perturb_pose(true_coords, method, elements, seed):
    rng = np.random.default_rng(seed)
    base_sigma = {'vina': 0.55, 'gnina': 0.45, 'diffdock': 0.75, 'equibind': 0.65}
    elem_mult = {'C': 1.0, 'N': 1.1, 'O': 1.2, 'S': 1.4}
    pred_coords = []
    for i, (coord, el) in enumerate(zip(true_coords, elements)):
        sigma = base_sigma[method] * elem_mult[el]
        noise = rng.normal(0, sigma, size=3)
        pred_coords.append(coord + noise)
    return np.array(pred_coords)

def compute_gnn_quantile(true_coords, pred_coords, method, elements, seed):
    rng = np.random.default_rng(seed)
    base_sigma = {'vina': 0.55, 'gnina': 0.45, 'diffdock': 0.75, 'equibind': 0.65}
    elem_mult = {'C': 1.0, 'N': 1.1, 'O': 1.2, 'S': 1.4}
    method_bias = {'vina': 0.4, 'gnina': 2.0, 'diffdock': 0.5, 'equibind': 1.8}
    elem_bias = {'C': 1.0, 'N': 0.8, 'O': 1.2, 'S': 0.6}
    quantiles = []
    for el in elements:
        sigma = base_sigma[method] * elem_mult[el]
        true_quantile = sigma * 1.8
        bias = method_bias[method] * elem_bias[el]
        noise = np.exp(rng.normal(0, 0.1))
        q = true_quantile * bias * noise
        quantiles.append(q)
    return np.array(quantiles)

def generate_test_set(output_dir, seed=12345, num_complexes=80):
    os.makedirs(os.path.join(output_dir, 'proteins'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'poses'), exist_ok=True)
    gnn_rows = []
    true_data = {}
    rng = random.Random(seed)
    for idx in range(num_complexes):
        complex_id = f"TEST{idx:04d}"
        num_atoms = rng.randint(20, 40)
        mol = generate_ligand(num_atoms, seed + idx)
        conf = mol.GetConformer()
        true_coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(num_atoms)])
        elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(num_atoms)]
        protein_coords = generate_protein(true_coords, num_atoms=300, seed=seed+idx+1000)
        write_pdb(protein_coords, os.path.join(output_dir, 'proteins', f"{complex_id}.pdb"), seed+idx+2000)
        for method in ['vina', 'gnina', 'diffdock', 'equibind']:
            m_hash = sum(ord(c) for c in method)
            pred_coords = perturb_pose(true_coords, method, elements, seed+idx+m_hash)
            pred_mol = Chem.Mol(mol)
            pred_conf = Chem.Conformer(num_atoms)
            for i in range(num_atoms):
                pred_conf.SetAtomPosition(i, pred_coords[i])
            pred_mol.RemoveAllConformers()
            pred_mol.AddConformer(pred_conf)
            Chem.MolToMolFile(pred_mol, os.path.join(output_dir, 'poses', f"{complex_id}_{method}.sdf"))
            qs = compute_gnn_quantile(true_coords, pred_coords, method, elements, seed+idx+m_hash+1000)
            for atom_idx, q in enumerate(qs):
                gnn_rows.append({
                    'complex_id': complex_id,
                    'docking_method': method,
                    'atom_id': atom_idx + 1,
                    'gnn_quantile': q
                })
                true_data[(complex_id, method, atom_idx+1)] = {
                    'true_coords': true_coords[atom_idx],
                    'pred_coords': pred_coords[atom_idx],
                    'gnn_quantile': q,
                    'element': elements[atom_idx]
                }
    df = pd.DataFrame(gnn_rows)
    df.to_csv(os.path.join(output_dir, 'gnn_quantiles.csv'), index=False)
    return true_data
