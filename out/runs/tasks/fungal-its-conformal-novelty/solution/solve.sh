#!/bin/bash
set -euo pipefail
python3 << 'EOF'
import pandas as pd
import numpy as np
import json
import os

cal = pd.read_csv('/app/data/calibration_scores.tsv', sep='\t')
val = pd.read_csv('/app/data/validation_scores.tsv', sep='\t')
query = pd.read_csv('/app/data/query_scores.tsv', sep='\t')
soil = pd.read_csv('/app/data/soil_asvs.tsv', sep='\t')

os.makedirs('/app/output', exist_ok=True)

alpha = 0.05
thresholds = {}
cal_scores = {}
for (region, method), group in cal.groupby(['region', 'method']):
    scores = np.sort(group['score'].values)
    n = len(scores)
    r = int(np.floor(alpha * (n + 1)))
    t = scores[r - 1] if r > 0 else scores[0] - 1e-6
    thresholds[f'{region}|{method}'] = t
    cal_scores[(region, method)] = scores

def compute_p(s, cal_scores):
    C = np.sum(cal_scores <= s)
    return (1 + C) / (len(cal_scores) + 1)

query['p_value'] = 0.0
query['novel_flag'] = 0
for (region, method), group in query.groupby(['region', 'method']):
    scores = cal_scores[(region, method)]
    for idx, row in group.iterrows():
        p = compute_p(row['score'], scores)
        query.at[idx, 'p_value'] = round(p, 6)
        query.at[idx, 'novel_flag'] = 1 if p <= alpha else 0
query = query.sort_values(['query_id', 'region', 'method'])
query[['query_id', 'region', 'method', 'p_value', 'novel_flag']].to_csv('/app/output/novelty_flags.tsv', sep='\t', index=False)

with open('/app/output/thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)

coverage = {}
for (region, method), group in val.groupby(['region', 'method']):
    scores = cal_scores[(region, method)]
    flags = [1 if compute_p(s, scores) <= alpha else 0 for s in group['score']]
    coverage[f'{region}|{method}'] = round(np.mean(flags), 4)
with open('/app/output/coverage_report.json', 'w') as f:
    json.dump(coverage, f, indent=2)

key = 'ITS1|alignment'
soil['p_value'] = 0.0
soil['novel_flag'] = 0
for idx, row in soil.iterrows():
    if row['region'] == 'ITS1' and row['method'] == 'alignment':
        p = compute_p(row['score'], cal_scores[('ITS1', 'alignment')])
        soil.at[idx, 'p_value'] = round(p, 6)
        soil.at[idx, 'novel_flag'] = 1 if p <= alpha else 0
soil_flags = soil[['asv_id', 'region', 'method', 'p_value', 'novel_flag', 'abundance']].copy()
soil_flags = soil_flags.sort_values('asv_id')
soil_flags.to_csv('/app/output/soil_flags.tsv', sep='\t', index=False)

all_asvs = soil_flags['novel_flag'] == 1
abundant = soil_flags['abundance'] >= 10
soil_summary = {
    'fraction_novel_all': round(all_asvs.mean(), 4),
    'fraction_novel_abundant': round((all_asvs & abundant).sum() / abundant.sum(), 4)
}
with open('/app/output/soil_summary.json', 'w') as f:
    json.dump(soil_summary, f, indent=2)
print('Solution completed.')
EOF
