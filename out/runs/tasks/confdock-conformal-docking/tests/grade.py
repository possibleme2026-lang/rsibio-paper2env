import os, sys, subprocess, json, math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.append('/tests')
import gold_gen

def read_sdf_coords(path):
    from rdkit import Chem
    mol = Chem.MolFromMolFile(path, removeHs=False)
    if mol is None:
        raise ValueError(f"Could not read {path}")
    conf = mol.GetConformer()
    coords = []
    elements = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
        elements.append(mol.GetAtomWithIdx(i).GetSymbol())
    return np.array(coords), elements

def main():
    test_dir = '/tmp/hidden_test'
    os.makedirs(test_dir, exist_ok=True)
    true_data = gold_gen.generate_test_set(test_dir, seed=12345, num_complexes=80)
    
    output_csv = '/tmp/agent_predictions.csv'
    cmd = ['python', '/app/solution/predict.py', '--input', test_dir, '--output', output_csv]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("Agent script failed:", proc.stderr)
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump({"score": 0.0, "reward": 0.0}, f)
        return
    
    if not os.path.exists(output_csv):
        print("Output CSV not found")
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump({"score": 0.0, "reward": 0.0}, f)
        return
    
    df = pd.read_csv(output_csv)
    expected_cols = ['complex_id', 'docking_method', 'atom_id', 'radius']
    
    c1 = list(df.columns) == expected_cols
    if c1:
        expected_keys = set(true_data.keys())
        agent_keys = set(zip(df['complex_id'], df['docking_method'], df['atom_id']))
        c1 = (agent_keys == expected_keys)
    
    c2 = False
    if c1:
        c2 = np.all(np.isfinite(df['radius'])) and np.all(df['radius'] > 0) and np.all(df['radius'] <= 10.0)
    
    coverage = 0.0
    method_coverages = {}
    mean_radius = 0.0
    baseline_mean_radius = 0.0
    corrs = {}
    if c1:
        df_indexed = df.set_index(['complex_id', 'docking_method', 'atom_id'])
        covered = []
        for key, row in df_indexed.iterrows():
            true_info = true_data.get(key)
            if true_info is None:
                continue
            dist = np.linalg.norm(true_info['true_coords'] - true_info['pred_coords'])
            covered.append(dist <= row['radius'])
        coverage = np.mean(covered) if covered else 0.0
        
        for method in ['vina', 'gnina', 'diffdock', 'equibind']:
            method_covered = []
            for key, row in df_indexed.iterrows():
                if key[1] != method:
                    continue
                true_info = true_data.get(key)
                if true_info is None:
                    continue
                dist = np.linalg.norm(true_info['true_coords'] - true_info['pred_coords'])
                method_covered.append(dist <= row['radius'])
            method_coverages[method] = np.mean(method_covered) if method_covered else 0.0
        
        calib_dir = '/app/data/calibration'
        calib_gnn = pd.read_csv(os.path.join(calib_dir, 'gnn_quantiles.csv'))
        scores = []
        for _, row in calib_gnn.iterrows():
            cid = row['complex_id']
            method = row['docking_method']
            atom_id = row['atom_id']
            gnn_q = row['gnn_quantile']
            true_coords, _ = read_sdf_coords(os.path.join(calib_dir, 'ligands', f"{cid}_true.sdf"))
            pred_coords, _ = read_sdf_coords(os.path.join(calib_dir, 'poses', f"{cid}_{method}.sdf"))
            true_pos = true_coords[atom_id-1]
            pred_pos = pred_coords[atom_id-1]
            error = np.linalg.norm(true_pos - pred_pos)
            scores.append(error / gnn_q)
        scores = np.array(scores)
        q_global = np.quantile(scores, 0.9)
        baseline_radii = []
        for key in true_data:
            gnn_q = true_data[key]['gnn_quantile']
            baseline_radii.append(q_global * gnn_q)
        baseline_mean_radius = np.mean(baseline_radii) if baseline_radii else 0.0
        
        mean_radius = df['radius'].mean()
        
        for method in ['vina', 'gnina', 'diffdock', 'equibind']:
            method_df = df[df['docking_method'] == method]
            if len(method_df) < 2:
                corrs[method] = 0.0
                continue
            gnn_qs = []
            radii = []
            for _, row in method_df.iterrows():
                key = (row['complex_id'], row['docking_method'], row['atom_id'])
                gnn_qs.append(true_data[key]['gnn_quantile'])
                radii.append(row['radius'])
            gnn_qs = np.array(gnn_qs)
            radii = np.array(radii)
            if np.std(gnn_qs) == 0 or np.std(radii) == 0:
                corrs[method] = 0.0
            else:
                corrs[method] = np.corrcoef(gnn_qs, radii)[0, 1]
    
    c3 = coverage >= 0.90 if c1 else False
    c4 = all(cov >= 0.85 for cov in method_coverages.values()) if c1 else False
    c5 = (mean_radius <= 0.70 * baseline_mean_radius) if c1 and baseline_mean_radius > 0 else False
    c6 = all(corr >= 0.5 for corr in corrs.values()) if c1 else False
    
    constraints = [c1, c2, c3, c4, c5, c6]
    score = sum(constraints) / len(constraints)
    
    print(f"Coverage: {coverage:.3f}")
    print(f"Method coverages: {method_coverages}")
    print(f"Mean radius: {mean_radius:.3f}, baseline: {baseline_mean_radius:.3f}, reduction: {1 - mean_radius/baseline_mean_radius:.3f}")
    print(f"Correlations: {corrs}")
    print(f"Constraints: {constraints}")
    
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({"score": score, "reward": score}, f)

if __name__ == '__main__':
    main()
