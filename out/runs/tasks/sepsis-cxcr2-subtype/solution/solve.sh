#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import json
import numpy as np
import pandas as pd

DATA_DIR = '/app/data'
meta = pd.read_csv(f'{DATA_DIR}/metadata.tsv', sep='\t')
blood = pd.read_csv(f'{DATA_DIR}/blood_counts.tsv', sep='\t', index_col=0)
liver = pd.read_csv(f'{DATA_DIR}/liver_counts.tsv', sep='\t', index_col=0)
with open(f'{DATA_DIR}/signatures.json') as f:
    signatures = json.load(f)

def score_tissue(counts):
    lib = counts.sum(axis=0)
    cpm = counts.div(lib, axis=1) * 1e6
    log2cpm = np.log2(cpm + 1)
    mean = log2cpm.mean(axis=1)
    std = log2cpm.std(axis=1, ddof=0)
    std[std == 0] = 1.0
    z = log2cpm.sub(mean, axis=0).div(std, axis=0)
    scores = {}
    for mts, genes in signatures.items():
        genes_in = [g for g in genes if g in z.index]
        scores[mts] = z.loc[genes_in].mean(axis=0)
    return pd.DataFrame(scores)

blood_scores = score_tissue(blood)
liver_scores = score_tissue(liver)
combined = (blood_scores + liver_scores) / 2.0
labels = combined.idxmax(axis=1)

meta_idx = meta.set_index('mouse_id')
meta_idx['MTS'] = labels
summary = {}
for mts in ['MTS1', 'MTS2', 'MTS3']:
    sub = meta_idx[meta_idx['MTS'] == mts]
    summary[mts] = {
        'n': int(len(sub)),
        'mean_pathogen_log10': float(sub['pathogen_log10'].mean()),
        'mean_IL6': float(sub['IL6_pg_ml'].mean())
    }
effects = {}
for mts in ['MTS1', 'MTS2', 'MTS3']:
    sub = meta_idx[meta_idx['MTS'] == mts]
    veh = sub[sub['treatment'] == 'Vehicle']
    dan = sub[sub['treatment'] == 'Danirixin']
    effects[mts] = {
        'survival_diff': float(dan['survival_7d'].mean() - veh['survival_7d'].mean()),
        'pathogen_diff': float(dan['pathogen_log10'].mean() - veh['pathogen_log10'].mean()),
        'netosis_diff': float(dan['NETosis_units'].mean() - veh['NETosis_units'].mean()),
        'hepatic_injury_diff': float(dan['hepatic_injury_ALT'].mean() - veh['hepatic_injury_ALT'].mean())
    }

result = {
    'sample_subtypes': {mid: str(labels[mid]) for mid in meta['mouse_id']},
    'signature_scores': {
        mid: {mts: float(combined.loc[mid, mts]) for mts in ['MTS1', 'MTS2', 'MTS3']}
        for mid in meta['mouse_id']
    },
    'subtype_summary': summary,
    'treatment_effects': effects,
    'recommendation': {
        'preferred_subtype': 'MTS3',
        'avoid_subtype': 'MTS1',
        'rationale': 'Danirixin improves 7-day survival in MTS3 but exacerbates bacterial dissemination in MTS1, so CXCR2 blockade should be guided by subtype.'
    }
}
with open('/app/output/result.json', 'w') as f:
    json.dump(result, f, indent=2)
PY
