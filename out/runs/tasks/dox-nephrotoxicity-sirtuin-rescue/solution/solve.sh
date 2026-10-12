#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python << 'EOF'
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

DATA_DIR = '/app/data'
OUTPUT_FILE = '/app/output/results.json'

rna = pd.read_csv(f'{DATA_DIR}/rna_counts.csv', index_col=0)
prot = pd.read_csv(f'{DATA_DIR}/proteomics.csv', index_col=0)
seahorse = pd.read_csv(f'{DATA_DIR}/seahorse_ocr.csv')
sim = pd.read_csv(f'{DATA_DIR}/sim_morphology.csv')
sod2 = pd.read_csv(f'{DATA_DIR}/sod2_acetyl.csv')
with open(f'{DATA_DIR}/gene_sets.json') as f:
    gene_sets = json.load(f)

samples = rna.columns.tolist()
groups = [s.split('_')[0] for s in samples]
group_indices = {g: [i for i, s in enumerate(samples) if s.startswith(g)] for g in set(groups)}
group_names = ['Control', 'DOX', 'DOX+RSV', 'DOX+NMN']

total_counts = rna.sum(axis=0)
cpm = rna.div(total_counts, axis=1) * 1e6
keep_genes = (cpm >= 1).sum(axis=1) >= 3
rna_filtered = rna.loc[keep_genes]
cpm_filtered = cpm.loc[keep_genes]
rna_genes_after_filter = int(rna_filtered.shape[0])

keep_prot = (prot > 0).sum(axis=1) >= 3
prot_filtered = prot.loc[keep_prot]
proteins_after_filter = int(prot_filtered.shape[0])

log_cpm = np.log2(cpm_filtered + 1)
dox_idx = group_indices['DOX']
ctrl_idx = group_indices['Control']

log2fc = log_cpm.iloc[:, dox_idx].mean(axis=1) - log_cpm.iloc[:, ctrl_idx].mean(axis=1)
pvals = []
for gene in log_cpm.index:
    dox_vals = log_cpm.loc[gene, log_cpm.columns[dox_idx]]
    ctrl_vals = log_cpm.loc[gene, log_cpm.columns[ctrl_idx]]
    t, p = stats.ttest_ind(dox_vals, ctrl_vals, equal_var=False)
    pvals.append(p)
pvals = np.array(pvals)
_, fdr, _, _ = multipletests(pvals, method='fdr_bh')
sig = (fdr < 0.05) & (np.abs(log2fc) > 1.0)
sig_genes = log2fc[sig].index.tolist()

oxphos_set = set(gene_sets['OXPHOS'])
inflam_set = set(gene_sets['Inflammation'])
down_oxphos = [g for g in sig_genes if g in oxphos_set and log2fc[g] < 0]
up_inflam = [g for g in sig_genes if g in inflam_set and log2fc[g] > 0]
num_down_OXPHOS = len(down_oxphos)
num_up_inflammation = len(up_inflam)

sig_log2fc = log2fc[sig]
top10_down = sig_log2fc.sort_values(ascending=True).head(10).index.tolist()
top10_up = sig_log2fc.sort_values(ascending=False).head(10).index.tolist()

z_scores = log_cpm.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0, axis=1)
pathway_scores = {}
for pathway in ['OXPHOS', 'FAO']:
    genes = [g for g in gene_sets[pathway] if g in z_scores.index]
    if len(genes) == 0:
        pathway_scores[pathway] = {g: 0.0 for g in group_names}
        continue
    sample_scores = z_scores.loc[genes].mean(axis=0)
    grp_means = {}
    for g in group_names:
        idx = group_indices[g]
        grp_means[g] = float(sample_scores.iloc[idx].mean())
    pathway_scores[pathway] = grp_means

basal_window = (seahorse['time'] >= 0) & (seahorse['time'] <= 18)
oligo_window = (seahorse['time'] >= 21) & (seahorse['time'] <= 38)
fccp_window = (seahorse['time'] >= 42) & (seahorse['time'] <= 58)
non_mito_window = (seahorse['time'] >= 63) & (seahorse['time'] <= 90)

well_metrics = {}
for well in seahorse['well'].unique():
    well_data = seahorse[seahorse['well'] == well]
    non_mito = well_data[non_mito_window]['OCR'].mean()
    basal = well_data[basal_window]['OCR'].mean() - non_mito
    oligo = well_data[oligo_window]['OCR'].mean() - non_mito
    maximal = well_data[fccp_window]['OCR'].mean() - non_mito
    spare = maximal - basal
    atp = basal - oligo
    well_metrics[well] = {'basal': float(basal), 'maximal': float(maximal), 'spare': float(spare), 'ATP': float(atp), 'non_mito': float(non_mito)}

seahorse_metrics = {}
for metric in ['basal', 'maximal', 'spare', 'ATP', 'non_mito']:
    seahorse_metrics[metric] = {}
    for g in group_names:
        wells_g = seahorse[seahorse['group'] == g]['well'].unique()
        vals = [well_metrics[w][metric] for w in wells_g]
        seahorse_metrics[metric][g] = float(np.mean(vals))

sim['fragmented'] = (sim['aspect_ratio'] < 2.0) & (sim['form_factor'] < 0.5)
sim_percent = {}
for g in group_names:
    grp_data = sim[sim['group'] == g]
    sim_percent[g] = float(100 * grp_data['fragmented'].mean())

sod2['ratio'] = sod2['acetylated_SOD2'] / sod2['total_SOD2']
sod2_ratio = {}
for g in group_names:
    grp_data = sod2[sod2['group'] == g]
    sod2_ratio[g] = float(grp_data['ratio'].mean())

metrics = {
    'OXPHOS': (pathway_scores['OXPHOS'], True),
    'FAO': (pathway_scores['FAO'], True),
    'basal': (seahorse_metrics['basal'], True),
    'maximal': (seahorse_metrics['maximal'], True),
    'spare': (seahorse_metrics['spare'], True),
    'ATP': (seahorse_metrics['ATP'], True),
    'percent_non_fragmented': ({g: 100 - sim_percent[g] for g in group_names}, True),
    'sod2_acetylation': (sod2_ratio, False)
}
rescue_scores = {}
for treatment in ['DOX+RSV', 'DOX+NMN']:
    score = 0.0
    for name, (values, higher_better) in metrics.items():
        ctrl = values['Control']
        dox = values['DOX']
        treat = values[treatment]
        if higher_better:
            if abs(ctrl - dox) > 1e-9:
                score += (treat - dox) / (ctrl - dox)
        else:
            if abs(dox - ctrl) > 1e-9:
                score += (dox - treat) / (dox - ctrl)
    rescue_scores[treatment] = float(score)

output = {
    'qc': {'rna_genes_after_filter': rna_genes_after_filter, 'proteins_after_filter': proteins_after_filter},
    'differential_expression': {'num_down_OXPHOS': num_down_OXPHOS, 'num_up_inflammation': num_up_inflammation, 'top10_down': top10_down, 'top10_up': top10_up},
    'pathway_scores': {'OXPHOS': pathway_scores['OXPHOS'], 'FAO': pathway_scores['FAO']},
    'seahorse': seahorse_metrics,
    'sim_morphology': {'percent_fragmented': sim_percent},
    'sod2_acetylation': {'ratio': sod2_ratio},
    'rescue_scores': rescue_scores
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
EOF
