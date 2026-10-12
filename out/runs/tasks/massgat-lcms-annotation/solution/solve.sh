#!/bin/bash
set -euo pipefail
cat > /app/predict.py << 'PYEOF'
import csv, sys, math
from collections import defaultdict
import networkx as nx

ADDUCTS = {'M+H': 1.007276, 'M+Na': 22.989218, 'M+NH4': 18.033823, 'M+K': 38.963158}
ISOTOPES = {'iso0': 0.0, 'iso1': 1.003355, 'iso2': 2.006710}
ION_TYPES = [(ad, iso, ADDUCTS[ad] + ISOTOPES[iso]) for ad in ADDUCTS for iso in ISOTOPES]

def read_features(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'feature_id': int(r['feature_id']),
                'mz': float(r['mz']),
                'rt': float(r['rt']),
                'intensity': float(r['intensity'])
            })
    return rows

def build_graph(features, ppm=5.0, rt_tol=0.1):
    G = nx.Graph()
    for f in features:
        G.add_node(f['feature_id'])
    n = len(features)
    for i in range(n):
        for j in range(i+1, n):
            f1 = features[i]
            f2 = features[j]
            if abs(f1['rt'] - f2['rt']) > rt_tol:
                continue
            mz1 = f1['mz']
            mz2 = f2['mz']
            delta = abs(mz2 - mz1)
            matched = False
            for (ad1, iso1, shift1) in ION_TYPES:
                for (ad2, iso2, shift2) in ION_TYPES:
                    expected = abs(shift2 - shift1)
                    if expected == 0:
                        continue
                    if abs(delta - expected) <= ppm * 1e-6 * max(mz1, mz2):
                        matched = True
                        break
                if matched:
                    break
            if matched:
                G.add_edge(f1['feature_id'], f2['feature_id'])
    return G

def infer_component(feats):
    candidates = []
    for f in feats:
        for ad, iso, shift in ION_TYPES:
            M = f['mz'] - shift
            candidates.append((M, f['feature_id'], ad, iso))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    best_count = 0
    best_M = None
    for i in range(len(candidates)):
        M0 = candidates[i][0]
        members = []
        for cand in candidates:
            if abs(cand[0] - M0) <= 5e-6 * M0:
                members.append(cand)
        unique_fids = set(m[1] for m in members)
        if len(unique_fids) > best_count:
            best_count = len(unique_fids)
            best_M = M0
    if best_count < 2:
        return None
    assignments = {}
    for f in feats:
        fid = f['feature_id']
        best_match = None
        best_ppm = float('inf')
        for ad, iso, shift in ION_TYPES:
            M_impl = f['mz'] - shift
            ppm_err = abs(M_impl - best_M) / best_M * 1e6
            if ppm_err <= 5.0 and ppm_err < best_ppm:
                best_ppm = ppm_err
                best_match = (ad, iso)
        if best_match:
            ad, iso = best_match
            assignments[fid] = f'{ad}_{iso}'
    if len(assignments) >= 2:
        return best_M, assignments
    return None

def main():
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    features = read_features(input_csv)
    G = build_graph(features)
    components = list(nx.connected_components(G))
    output = []
    compound_id = 1
    for comp in components:
        feats = [f for f in features if f['feature_id'] in comp]
        res = infer_component(feats)
        if res:
            M, assignments = res
            for f in feats:
                fid = f['feature_id']
                if fid in assignments:
                    output.append({'feature_id': fid, 'is_valid': 1, 'compound_id': compound_id, 'ion_type': assignments[fid]})
                else:
                    output.append({'feature_id': fid, 'is_valid': 0, 'compound_id': -1, 'ion_type': 'noise'})
            compound_id += 1
        else:
            for f in feats:
                output.append({'feature_id': f['feature_id'], 'is_valid': 0, 'compound_id': -1, 'ion_type': 'noise'})
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['feature_id','is_valid','compound_id','ion_type'])
        writer.writeheader()
        for row in output:
            writer.writerow(row)

if __name__ == '__main__':
    main()
PYEOF
python /app/predict.py /app/data/sample_features.csv /app/predictions.csv
