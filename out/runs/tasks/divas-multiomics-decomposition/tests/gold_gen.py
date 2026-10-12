#!/usr/bin/env python3
import numpy as np

SEED = 20260112
N = 60
T_FEAT = 200
P_FEAT = 150
M_FEAT = 120

def get_true_components():
    rng = np.random.default_rng(SEED)
    components = []
    for i in range(3):
        components.append({'type': 'joint', 'modalities': ['transcriptomics', 'proteomics', 'metabolomics']})
    for i in range(2):
        components.append({'type': 'partial', 'modalities': ['transcriptomics', 'proteomics']})
    for i in range(2):
        components.append({'type': 'partial', 'modalities': ['transcriptomics', 'metabolomics']})
    for i in range(1):
        components.append({'type': 'partial', 'modalities': ['proteomics', 'metabolomics']})
    for i in range(2):
        components.append({'type': 'individual', 'modalities': ['transcriptomics']})
    for i in range(2):
        components.append({'type': 'individual', 'modalities': ['proteomics']})
    for i in range(1):
        components.append({'type': 'individual', 'modalities': ['metabolomics']})
    for i, comp in enumerate(components):
        comp['id'] = f'true_{i+1:02d}'
        comp['scores'] = rng.normal(0, 1, N)
        comp['scores'] = (comp['scores'] - comp['scores'].mean()) / comp['scores'].std()
        comp['loadings'] = {}
        for mod in comp['modalities']:
            n_feat = {'transcriptomics': T_FEAT, 'proteomics': P_FEAT, 'metabolomics': M_FEAT}[mod]
            load = rng.normal(0, 1, n_feat)
            load = load / np.linalg.norm(load)
            comp['loadings'][mod] = load
        if comp['type'] == 'joint':
            comp['strength'] = 2.0
        elif comp['type'] == 'partial':
            comp['strength'] = 1.5
        else:
            comp['strength'] = 1.0
    return components

if __name__ == '__main__':
    import os
    import pandas as pd
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='.')
    args = parser.parse_args()
    components = get_true_components()
    rng = np.random.default_rng(SEED + 1)
    data = {}
    for mod, n_feat in [('transcriptomics', T_FEAT), ('proteomics', P_FEAT), ('metabolomics', M_FEAT)]:
        X = np.zeros((N, n_feat))
        for comp in components:
            if mod in comp['modalities']:
                X += comp['strength'] * np.outer(comp['scores'], comp['loadings'][mod])
        X += rng.normal(0, 1.0, (N, n_feat))
        X = X - X.mean(axis=0)
        data[mod] = X
    os.makedirs(args.output, exist_ok=True)
    sample_ids = [f'S{i+1:02d}' for i in range(N)]
    for mod, n_feat in [('transcriptomics', T_FEAT), ('proteomics', P_FEAT), ('metabolomics', M_FEAT)]:
        if mod == 'transcriptomics':
            cols = [f'gene_{i+1}' for i in range(n_feat)]
        elif mod == 'proteomics':
            cols = [f'protein_{i+1}' for i in range(n_feat)]
        else:
            cols = [f'metabolite_{i+1}' for i in range(n_feat)]
        df = pd.DataFrame(data[mod], columns=cols)
        df.insert(0, 'sample_id', sample_ids)
        df.to_csv(os.path.join(args.output, f'{mod}.csv'), index=False)
    pd.DataFrame({'sample_id': sample_ids}).to_csv(os.path.join(args.output, 'samples.csv'), index=False)
