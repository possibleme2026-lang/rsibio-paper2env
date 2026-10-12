#!/bin/bash
set -euo pipefail
python3 <<'EOF'
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_PATH = '/app/data/benchmark_data.csv'
OUT_PATH = '/app/output/benchmark_report.json'
METHODS = ['alphafold2', 'esmfold', 'rosetta', 'foldx', 'evfold', 'sparks']
STATES = ['p_3', 'p_4b_plus', 'p_mixed', 'p_unfolded']
POSITIONS = ['26', '33', '45']

df = pd.read_csv(DATA_PATH)

def compute_metrics(df, method, position=None):
    if position is not None:
        df = df[df['position'] == int(position)]
    pred_cols = [f'{method}_{s}' for s in STATES]
    sub = df.dropna(subset=pred_cols)
    if len(sub) == 0:
        return None
    rmse = {}
    spearman = {}
    for s in STATES:
        meas = sub[f'measured_{s}'].values
        pred = sub[f'{method}_{s}'].values
        rmse[s] = float(np.sqrt(np.mean((pred - meas)**2)))
        if len(np.unique(meas)) < 2 or len(np.unique(pred)) < 2:
            spearman[s] = float('nan')
        else:
            rho, _ = spearmanr(meas, pred)
            spearman[s] = float(rho)
    return {'rmse': rmse, 'spearman': spearman}

overall = {}
for method in METHODS:
    overall[method] = compute_metrics(df, method)

per_pos = {}
for pos in POSITIONS:
    per_pos[pos] = {}
    for method in METHODS:
        per_pos[pos][method] = compute_metrics(df, method, pos)

best = {}
for pos in POSITIONS:
    avg_rmse = {}
    avg_spearman = {}
    for method in METHODS:
        m = per_pos[pos][method]
        if m is None:
            continue
        avg_rmse[method] = np.mean(list(m['rmse'].values()))
        avg_spearman[method] = np.nanmean(list(m['spearman'].values()))
    best_rmse = sorted(avg_rmse.items(), key=lambda x: (x[1], x[0]))[0][0]
    best_spearman = sorted(avg_spearman.items(), key=lambda x: (-x[1], x[0]))[0][0]
    best[pos] = {'best_rmse_method': best_rmse, 'best_spearman_method': best_spearman}

result = {
    'overall_metrics': overall,
    'per_position_metrics': per_pos,
    'best_methods': best
}
os.makedirs('/app/output', exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(result, f, indent=2)
print('Report written to', OUT_PATH)
EOF
