#!/bin/bash
set -euo pipefail
mkdir -p /app/results
python - <<'PY'
import json, os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import mannwhitneyu, pearsonr
from statsmodels.stats.multitest import multipletests

data_dir = '/app/data'
counts = sio.mmread(os.path.join(data_dir, 'scrna', 'counts.mtx')).tocsr()
genes = pd.read_csv(os.path.join(data_dir, 'scrna', 'genes.tsv'), header=None)[0].tolist()
barcodes = pd.read_csv(os.path.join(data_dir, 'scrna', 'barcodes.tsv'), header=None)[0].tolist()
meta = pd.read_csv(os.path.join(data_dir, 'scrna', 'cell_metadata.tsv'), sep='\t')
clin = pd.read_csv(os.path.join(data_dir, 'clinical', 'participants.tsv'), sep='\t')
with open(os.path.join(data_dir, 'genesets', 'gng4_tfh_signature.txt')) as f:
    gng4_sig = [l.strip() for l in f if l.strip()]
with open(os.path.join(data_dir, 'genesets', 'inflammatory.txt')) as f:
    infl = [l.strip() for l in f if l.strip()]
with open(os.path.join(data_dir, 'genesets', 'bcl6_regulon.txt')) as f:
    bcl6_reg = [l.strip() for l in f if l.strip()]
with open(os.path.join(data_dir, 'genesets', 'dz_markers.txt')) as f:
    dz = [l.strip() for l in f if l.strip()]
with open(os.path.join(data_dir, 'genesets', 'lz_markers.txt')) as f:
    lz = [l.strip() for l in f if l.strip()]
with open(os.path.join(data_dir, 'genesets', 'myc_targets.txt')) as f:
    myc = [l.strip() for l in f if l.strip()]
with open(os.path.join(data_dir, 'genesets', 'mtorc1_targets.txt')) as f:
    mtor = [l.strip() for l in f if l.strip()]
bcr = pd.read_csv(os.path.join(data_dir, 'bcr', 'airr_repertoire.tsv'), sep='\t')
lib = np.array(counts.sum(axis=0)).flatten()
lib[lib==0]=1
counts_norm = counts.multiply(10000.0/lib).tocsr()
counts_norm.data = np.log1p(counts_norm.data)
gene_to_idx = {g:i for i,g in enumerate(genes)}
barcode_to_idx = {b:i for i,b in enumerate(barcodes)}
donors = clin['donor_id'].tolist()
titers = dict(zip(clin['donor_id'], clin['end_of_study_titer_mIU_mL']))
groups = {d: ('High' if titers[d]>=100 else 'Low') for d in donors}
tfh = meta[meta['cell_type']=='Tfh'].copy()
tfh_idx = [barcode_to_idx[b] for b in tfh['cell_id']]
gcb = meta[meta['cell_type']=='GC_B'].copy()
gcb_idx = [barcode_to_idx[b] for b in gcb['cell_id']]
def zscore(cell_idx, gene_list):
    sub = counts_norm[[gene_to_idx[g] for g in gene_list], :][:, cell_idx]
    arr = sub.toarray()
    m = arr.mean(axis=1, keepdims=True)
    s = arr.std(axis=1, keepdims=True)
    s[s==0]=1
    return (arr-m)/s
gng4_expr = counts_norm[gene_to_idx['GNG4'], tfh_idx].toarray().flatten()
gng4_plus = np.zeros(len(tfh_idx), dtype=bool)
for d in donors:
    dm = (tfh['donor_id']==d).values
    if dm.sum()==0: continue
    th = np.percentile(gng4_expr[dm], 60)
    gng4_plus[dm] = gng4_expr[dm] > th
tfh['is_gng4_plus'] = gng4_plus
def mean_zscore(gene_list):
    z = zscore(tfh_idx, gene_list)
    cell_score = z.mean(axis=0)
    res = {}
    for d in donors:
        mask = (tfh['donor_id']==d).values & gng4_plus
        res[d] = float(cell_score[mask].mean()) if mask.sum()>0 else 0.0
    return res
gng4_score = mean_zscore(gng4_sig)
infl_score = mean_zscore(infl)
bcl6_act = mean_zscore(bcl6_reg)
bcl6_expr = counts_norm[gene_to_idx['BCL6'], tfh_idx].toarray().flatten()
reg_targets = [g for g in bcl6_reg if g!='BCL6']
z_t = zscore(tfh_idx, reg_targets)
target_score = z_t.mean(axis=0)
bcl6_spec = {}
for d in donors:
    mask = (tfh['donor_id']==d).values & gng4_plus
    if mask.sum()>2:
        bcl6_spec[d] = float(pearsonr(bcl6_expr[mask], target_score[mask])[0])
    else:
        bcl6_spec[d] = 0.0
def gcb_score(gene_list):
    z = zscore(gcb_idx, gene_list)
    cs = z.mean(axis=0)
    res = {}
    for d in donors:
        mask = (gcb['donor_id']==d).values
        res[d] = float(cs[mask].mean()) if mask.sum()>0 else 0.0
    return res
dz_s = gcb_score(dz)
lz_s = gcb_score(lz)
myc_s = gcb_score(myc)
mtor_s = gcb_score(mtor)
bcr['donor_id'] = bcr['cell_id'].str.split('_').str[0]
mean_shm = bcr.groupby('donor_id')['somatic_hypermutation_rate'].mean().to_dict()
div = {}
for d, grp in bcr.groupby('donor_id'):
    cnt = grp['clonotype_id'].value_counts()
    p = cnt/cnt.sum()
    div[d] = float(-np.sum(p*np.log(p)))
clono_donors = bcr.groupby('clonotype_id')['donor_id'].nunique()
public_clonos = set(clono_donors[clono_donors>=2].index)
pub_count = {}
for d in donors:
    donor_clonos = set(bcr[bcr['donor_id']==d]['clonotype_id'])
    pub_count[d] = len(donor_clonos & public_clonos)
gng4_tfh_idx = [tfh_idx[i] for i in range(len(tfh_idx)) if gng4_plus[i]]
gng4_tfh_donor = [tfh['donor_id'].values[i] for i in range(len(tfh_idx)) if gng4_plus[i]]
gng4_counts = counts_norm[:, gng4_tfh_idx]
high_mask = np.array([groups[d]=='High' for d in gng4_tfh_donor])
low_mask = ~high_mask
de = []
for i,g in enumerate(genes):
    expr = gng4_counts[i,:].toarray().flatten()
    mh = expr[high_mask].mean()
    ml = expr[low_mask].mean()
    log2fc = np.log2((mh+1)/(ml+1))
    stat,p = mannwhitneyu(expr[high_mask], expr[low_mask], alternative='two-sided')
    de.append({'gene':g, 'mean_high':mh, 'mean_low':ml, 'log2_fold_change':log2fc, 'p_value':p})
de_df = pd.DataFrame(de)
_, padj, _, _ = multipletests(de_df['p_value'], method='fdr_bh')
de_df['padj'] = padj
de_df['direction'] = np.where(de_df['log2_fold_change']>0, 'up_in_high', 'up_in_low')
de_df = de_df.sort_values('p_value')
de_df.to_csv('/app/results/de_genes.csv', index=False)
titer_vals = [titers[d] for d in donors]
cor_gng4 = float(pearsonr([gng4_score[d] for d in donors], titer_vals)[0])
cor_shm = float(pearsonr([mean_shm[d] for d in donors], titer_vals)[0])
cor_infl = float(pearsonr([infl_score[d] for d in donors], titer_vals)[0])
out = {'donors': {}, 'correlations': {'gng4_score_vs_titer': cor_gng4, 'shm_vs_titer': cor_shm, 'inflammatory_vs_titer': cor_infl}}
for d in donors:
    out['donors'][d] = {
        'responder_group': groups[d],
        'end_of_study_titer_mIU_mL': float(titers[d]),
        'n_tfh': int((tfh['donor_id']==d).sum()),
        'n_gng4_tfh': int(((tfh['donor_id']==d) & gng4_plus).sum()),
        'gng4_tfh_signature_score': gng4_score[d],
        'inflammatory_score': infl_score[d],
        'bcl6_regulon_activity': bcl6_act[d],
        'bcl6_regulon_specificity': bcl6_spec[d],
        'dz_score': dz_s[d],
        'lz_score': lz_s[d],
        'myc_score': myc_s[d],
        'mtorc1_score': mtor_s[d],
        'mean_shm': mean_shm[d],
        'clonotype_diversity': div[d],
        'public_clonotypes': pub_count[d]
    }
with open('/app/results/analysis.json', 'w') as f:
    json.dump(out, f, indent=2)
PY
