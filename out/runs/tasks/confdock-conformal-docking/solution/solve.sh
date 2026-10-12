#!/bin/bash
set -euo pipefail
mkdir -p /app/solution
cat > /app/solution/predict.py << 'EOF'
import argparse, os, sys
import numpy as np
import pandas as pd
from rdkit import Chem

def read_sdf_coords(path):
    mol = Chem.MolFromMolFile(path, removeHs=False)
    conf = mol.GetConformer()
    coords = []
    elements = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
        elements.append(mol.GetAtomWithIdx(i).GetSymbol())
    return np.array(coords), elements

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    calib_dir = '/app/data/calibration'
    test_dir = args.input
    
    calib_gnn = pd.read_csv(os.path.join(calib_dir, 'gnn_quantiles.csv'))
    
    scores = []
    groups = []
    for _, row in calib_gnn.iterrows():
        cid = row['complex_id']
        method = row['docking_method']
        atom_id = row['atom_id']
        gnn_q = row['gnn_quantile']
        true_coords, elements = read_sdf_coords(os.path.join(calib_dir, 'ligands', f"{cid}_true.sdf"))
        pred_coords, _ = read_sdf_coords(os.path.join(calib_dir, 'poses', f"{cid}_{method}.sdf"))
        true_pos = true_coords[atom_id-1]
        pred_pos = pred_coords[atom_id-1]
        error = np.linalg.norm(true_pos - pred_pos)
        score = error / gnn_q
        scores.append(score)
        groups.append((method, elements[atom_id-1]))
    scores = np.array(scores)
    groups = np.array(groups, dtype=object)
    
    alpha = 0.1
    q_groups = {}
    unique_groups = set(groups)
    for g in unique_groups:
        mask = np.array([tuple(x) == g for x in groups])
        n = mask.sum()
        if n >= 10:
            sorted_scores = np.sort(scores[mask])
            idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
            if idx >= n:
                idx = n - 1
            q_groups[g] = sorted_scores[idx]
        else:
            q_groups[g] = None
    
    q_method = {}
    for method in ['vina', 'gnina', 'diffdock', 'equibind']:
        mask = np.array([x[0] == method for x in groups])
        n = mask.sum()
        sorted_scores = np.sort(scores[mask])
        idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
        if idx >= n:
            idx = n - 1
        q_method[method] = sorted_scores[idx]
    
    sorted_scores = np.sort(scores)
    n = len(scores)
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    if idx >= n:
        idx = n - 1
    q_global = sorted_scores[idx]
    
    test_gnn = pd.read_csv(os.path.join(test_dir, 'gnn_quantiles.csv'))
    results = []
    for _, row in test_gnn.iterrows():
        cid = row['complex_id']
        method = row['docking_method']
        atom_id = row['atom_id']
        gnn_q = row['gnn_quantile']
        pred_coords, elements = read_sdf_coords(os.path.join(test_dir, 'poses', f"{cid}_{method}.sdf"))
        el = elements[atom_id-1]
        g = (method, el)
        if g in q_groups and q_groups[g] is not None:
            q = q_groups[g]
        elif method in q_method:
            q = q_method[method]
        else:
            q = q_global
        radius = q * gnn_q
        results.append({
            'complex_id': cid,
            'docking_method': method,
            'atom_id': atom_id,
            'radius': radius
        })
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} predictions to {args.output}")

if __name__ == '__main__':
    main()
EOF
chmod +x /app/solution/predict.py
echo "Reference solution written to /app/solution/predict.py"
