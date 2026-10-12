#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python << 'PY'
import pandas as pd, numpy as np, json, os

in_dir = '/app/data'
out_dir = '/app/output'
psm = pd.read_csv(f'{in_dir}/psm_table.tsv', sep='\t')
feat = pd.read_csv(f'{in_dir}/feature_table.tsv', sep='\t')
map_df = pd.read_csv(f'{in_dir}/peptide_protein_map.tsv', sep='\t')
unique_peps = set(map_df[map_df.mapping_type == 'unique'].peptide_sequence)

psm_nonmbr = psm[(psm.is_mbr == 0) & (psm.peptide_sequence.isin(unique_peps))]
spc = psm_nonmbr.groupby(['run', 'protein_id']).size().reset_index(name='spc')
spc.sort_values(['run', 'protein_id']).to_csv(f'{out_dir}/protein_spc.tsv', sep='\t', index=False)

psm_auc = psm_nonmbr[['run', 'peptide_sequence', 'charge', 'protein_id']].drop_duplicates()
merged = psm_auc.merge(feat, on=['run', 'peptide_sequence', 'charge'], how='inner')
merged = merged.drop_duplicates(subset=['run', 'feature_id'])
auc_nombr = merged.groupby(['run', 'protein_id'])['auc'].sum().reset_index()
auc_nombr.sort_values(['run', 'protein_id']).to_csv(f'{out_dir}/protein_auc_nombr.tsv', sep='\t', index=False)

psm_mbr = psm[psm.peptide_sequence.isin(unique_peps)][['run', 'peptide_sequence', 'charge', 'protein_id']].drop_duplicates()
merged_mbr = psm_mbr.merge(feat, on=['run', 'peptide_sequence', 'charge'], how='inner')
merged_mbr = merged_mbr.drop_duplicates(subset=['run', 'feature_id'])
auc_mbr = merged_mbr.groupby(['run', 'protein_id'])['auc'].sum().reset_index()
auc_mbr.sort_values(['run', 'protein_id']).to_csv(f'{out_dir}/protein_auc_mbr.tsv', sep='\t', index=False)

mbr_prots = psm[psm.is_mbr == 1].groupby(['run', 'protein_id']).size().reset_index(name='mbr_psm_count')
nonmbr_pairs = set(psm[psm.is_mbr == 0][['run', 'protein_id']].itertuples(index=False, name=None))
mbr_prots['has_nonmbr'] = mbr_prots.apply(lambda r: (r['run'], r['protein_id']) in nonmbr_pairs, axis=1)
fp = mbr_prots[~mbr_prots.has_nonmbr][['run', 'protein_id', 'mbr_psm_count']]
fp.sort_values(['run', 'protein_id']).to_csv(f'{out_dir}/mbr_false_positive_proteins.tsv', sep='\t', index=False)

def dyn_range(df, col):
    vals = df[df[col] > 0][col]
    if len(vals) < 2:
        return 0.0
    return float(np.log10(vals.max() / vals.min()))

report = {
    'dynamic_range_auc': dyn_range(auc_nombr, 'auc'),
    'dynamic_range_spc': dyn_range(spc, 'spc'),
    'quantified_proteins_auc': int(len(auc_nombr)),
    'quantified_proteins_spc': int(len(spc)),
    'mbr_false_positive_proteins': int(len(fp)),
    'mbr_false_positive_psms': int(fp['mbr_psm_count'].sum()) if len(fp) > 0 else 0
}
with open(f'{out_dir}/report.json', 'w') as f:
    json.dump(report, f, indent=2)
print('Done')
PY
