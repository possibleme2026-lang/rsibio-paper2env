import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
from scipy.stats import mannwhitneyu, pearsonr
from statsmodels.stats.multitest import multipletests
import os

SEED = 42
TFH_MARKERS = ['CXCR5', 'PDCD1', 'BCL6', 'ICOS', 'IL21', 'TOX2', 'SH2D1A', 'CD200', 'BTLA', 'CXCL13', 'GNG4']
GNG4_SIGNATURE = ['GNG4', 'CXCL13', 'ICOS', 'PDCD1', 'BCL6', 'IL21', 'TOX2', 'SH2D1A', 'CD200', 'BTLA']
INFLAMMATORY = ['IFNG', 'TNF', 'IL17A', 'IL17F', 'STAT1', 'NFKB1', 'RELA', 'IL6', 'CXCL10', 'CCL5', 'GZMB', 'PRF1']
BCL6_REGULON = ['BCL6', 'CXCR4', 'GCSAM', 'AICDA', 'MKI67', 'RGS13', 'STMN1', 'TUBA1B', 'HMGB2', 'CD83', 'CD86']
DZ_MARKERS = ['CXCR4', 'GCSAM', 'MKI67', 'AICDA', 'RGS13', 'STMN1', 'TUBA1B', 'HMGB2']
LZ_MARKERS = ['CD83', 'CD86', 'CR2', 'FCER2', 'CD40', 'CD74', 'HLA-DRA', 'HLA-DRB1']
MYC_TARGETS = ['MYC', 'E2F1', 'CDK4', 'CCND2', 'TERT', 'ODC1', 'SRM', 'LDHA', 'ENO1', 'PKM']
MTORC1_TARGETS = ['RPS6KB1', 'EIF4EBP1', 'SLC7A5', 'SLC3A2', 'MYC', 'HIF1A', 'VEGFA', 'PPARGC1A', 'TFEB']
B_CELL_MARKERS = ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'CD38', 'CD27', 'IGHM', 'IGHD', 'IGHA1', 'IGHG1']
T_CELL_MARKERS = ['CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'TRAC']
ALL_MARKERS = list(set(TFH_MARKERS + GNG4_SIGNATURE + INFLAMMATORY + BCL6_REGULON + DZ_MARKERS + LZ_MARKERS + MYC_TARGETS + MTORC1_TARGETS + B_CELL_MARKERS + T_CELL_MARKERS))
N_GENES = 300
GENE_NAMES = ALL_MARKERS + [f'GENE{i:04d}' for i in range(N_GENES - len(ALL_MARKERS))]
GENE_NAMES = GENE_NAMES[:N_GENES]
DONORS = [f'D{i:02d}' for i in range(1, 11)]
HIGH = DONORS[:6]
LOW = DONORS[6:]
GROUPS = {d: 'High' for d in HIGH}
GROUPS.update({d: 'Low' for d in LOW})
TITERS = {'D01': 1200, 'D02': 800, 'D03': 1500, 'D04': 300, 'D05': 2000, 'D06': 1000, 'D07': 10, 'D08': 50, 'D09': 5, 'D10': 80}
VACC = {d: 'de_novo' if i%2==0 else 'booster' for i,d in enumerate(DONORS)}
CELL_TYPES = ['Tfh', 'GC_B', 'T_cell', 'B_cell', 'other']
CELL_COUNTS = {'Tfh': 300, 'GC_B': 200, 'T_cell': 100, 'B_cell': 100, 'other': 100}

def generate_data(outdir, seed=SEED):
    rng = np.random.default_rng(seed)
    n_cells = sum(CELL_COUNTS.values()) * len(DONORS)
    n_genes = len(GENE_NAMES)
    mu = rng.lognormal(mean=1.0, sigma=1.0, size=n_genes)
    for g in ALL_MARKERS:
        idx = GENE_NAMES.index(g)
        mu[idx] = rng.uniform(10, 50)
    cell_meta = []
    counts = np.zeros((n_cells, n_genes), dtype=int)
    cell_idx = 0
    for donor in DONORS:
        group = GROUPS[donor]
        for ct in CELL_TYPES:
            n = CELL_COUNTS[ct]
            for i in range(n):
                cell_id = f'{donor}_{ct}_{i:04d}'
                s = rng.lognormal(mean=0, sigma=0.3)
                lam = mu * s
                if ct == 'Tfh':
                    for g in TFH_MARKERS:
                        lam[GENE_NAMES.index(g)] *= 5.0
                elif ct in ['GC_B', 'B_cell']:
                    for g in B_CELL_MARKERS:
                        lam[GENE_NAMES.index(g)] *= 5.0
                elif ct == 'T_cell':
                    for g in T_CELL_MARKERS:
                        lam[GENE_NAMES.index(g)] *= 5.0
                is_gng4 = False
                if ct == 'Tfh':
                    is_gng4 = rng.random() < 0.4
                    if is_gng4:
                        for g in GNG4_SIGNATURE:
                            lam[GENE_NAMES.index(g)] *= 3.0
                        if group == 'High':
                            for g in GNG4_SIGNATURE:
                                lam[GENE_NAMES.index(g)] *= 1.5
                        else:
                            for g in INFLAMMATORY:
                                lam[GENE_NAMES.index(g)] *= 3.0
                if ct == 'GC_B':
                    if group == 'High':
                        for g in DZ_MARKERS + LZ_MARKERS:
                            lam[GENE_NAMES.index(g)] *= 2.0
                        for g in MYC_TARGETS + MTORC1_TARGETS:
                            lam[GENE_NAMES.index(g)] *= 2.0
                    else:
                        for g in DZ_MARKERS + LZ_MARKERS:
                            lam[GENE_NAMES.index(g)] *= 1.0
                        for g in MYC_TARGETS + MTORC1_TARGETS:
                            lam[GENE_NAMES.index(g)] *= 1.0
                if ct == 'Tfh' and is_gng4:
                    if group == 'High':
                        eff = rng.lognormal(mean=0, sigma=0.5)
                        for g in BCL6_REGULON:
                            lam[GENE_NAMES.index(g)] *= eff
                    else:
                        eff_bcl6 = rng.lognormal(mean=0, sigma=0.8)
                        lam[GENE_NAMES.index('BCL6')] *= eff_bcl6
                        eff_targets = rng.lognormal(mean=0, sigma=0.8)
                        for g in BCL6_REGULON:
                            if g != 'BCL6':
                                lam[GENE_NAMES.index(g)] *= eff_targets
                counts[cell_idx] = rng.negative_binomial(n=2, p=2/(2+lam))
                cell_meta.append({'cell_id': cell_id, 'sample_id': f'{donor}_biopsy', 'donor_id': donor, 'cell_type': ct})
                cell_idx += 1
    os.makedirs(os.path.join(outdir, 'scrna'), exist_ok=True)
    pd.Series(GENE_NAMES).to_csv(os.path.join(outdir, 'scrna', 'genes.tsv'), index=False, header=False)
    pd.Series([m['cell_id'] for m in cell_meta]).to_csv(os.path.join(outdir, 'scrna', 'barcodes.tsv'), index=False, header=False)
    sio.mmwrite(os.path.join(outdir, 'scrna', 'counts.mtx'), sp.csr_matrix(counts.T))
    pd.DataFrame(cell_meta).to_csv(os.path.join(outdir, 'scrna', 'cell_metadata.tsv'), sep='\t', index=False)
    os.makedirs(os.path.join(outdir, 'clinical'), exist_ok=True)
    clin = pd.DataFrame({'donor_id': DONORS, 'vaccination_type': [VACC[d] for d in DONORS], 'end_of_study_titer_mIU_mL': [TITERS[d] for d in DONORS]})
    clin.to_csv(os.path.join(outdir, 'clinical', 'participants.tsv'), sep='\t', index=False)
    genesets = {'tfh_markers': TFH_MARKERS, 'gng4_tfh_signature': GNG4_SIGNATURE, 'inflammatory': INFLAMMATORY, 'bcl6_regulon': BCL6_REGULON, 'dz_markers': DZ_MARKERS, 'lz_markers': LZ_MARKERS, 'myc_targets': MYC_TARGETS, 'mtorc1_targets': MTORC1_TARGETS}
    os.makedirs(os.path.join(outdir, 'genesets'), exist_ok=True)
    for name, genes in genesets.items():
        with open(os.path.join(outdir, 'genesets', f'{name}.txt'), 'w') as f:
            f.write('\n'.join(genes))
    os.makedirs(os.path.join(outdir, 'bcr'), exist_ok=True)
    public_clonotypes = [f'PUBLIC_{i:03d}' for i in range(20)]
    donor_public = {}
    for donor in DONORS:
        n = rng.integers(5, 11)
        donor_public[donor] = list(rng.choice(public_clonotypes, size=n, replace=False))
    records = []
    for donor in DONORS:
        group = GROUPS[donor]
        base_shm = 0.05 if group == 'High' else 0.02
        clonotypes = donor_public[donor].copy()
        n_private = rng.integers(50, 100)
        clonotypes += [f'PRIVATE_{donor}_{i:03d}' for i in range(n_private)]
        n_cells = CELL_COUNTS['GC_B'] + CELL_COUNTS['B_cell']
        probs = rng.dirichlet(np.ones(len(clonotypes)) * 0.5)
        cell_clono = rng.choice(clonotypes, size=n_cells, p=probs)
        for i, clono in enumerate(cell_clono):
            cell_id = f'{donor}_BCR_{i:04d}'
            n_contigs = rng.choice([1, 2], p=[0.7, 0.3])
            for j in range(n_contigs):
                contig_id = f'{cell_id}_{j}'
                v = f'IGHV{rng.integers(1,7)}-{rng.integers(1,10)}'
                d = f'IGHD{rng.integers(1,7)}-{rng.integers(1,10)}'
                j = f'IGHJ{rng.integers(1,7)}'
                cdr3_nt = ''.join(rng.choice(list('ACGT'), size=rng.integers(36, 54)))
                cdr3_aa = ''.join(rng.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=len(cdr3_nt)//3))
                shm = max(0.0, base_shm + rng.normal(0, 0.005))
                isotype = rng.choice(['IGHM', 'IGHD', 'IGHA1', 'IGHG1', 'IGHG2'])
                records.append({'cell_id': cell_id, 'contig_id': contig_id, 'v_call': v, 'd_call': d, 'j_call': j, 'cdr3_nt': cdr3_nt, 'cdr3_aa': cdr3_aa, 'somatic_hypermutation_rate': shm, 'isotype': isotype, 'clonotype_id': clono})
    pd.DataFrame(records).to_csv(os.path.join(outdir, 'bcr', 'airr_repertoire.tsv'), sep='\t', index=False)
    os.makedirs(os.path.join(outdir, 'tcr'), exist_ok=True)
    tcr_records = []
    for donor in DONORS:
        n_tfh = CELL_COUNTS['Tfh']
        for i in range(n_tfh):
            cell_id = f'{donor}_Tfh_{i:04d}'
            n_contigs = rng.choice([1, 2], p=[0.8, 0.2])
            for j in range(n_contigs):
                tcr_records.append({'cell_id': cell_id, 'contig_id': f'{cell_id}_{j}', 'v_call': f'TRAV{rng.integers(1,30)}', 'd_call': f'TRAD{rng.integers(1,5)}', 'j_call': f'TRAJ{rng.integers(1,50)}', 'cdr3_nt': ''.join(rng.choice(list('ACGT'), size=rng.integers(30, 45))), 'cdr3_aa': ''.join(rng.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=12)), 'clonotype_id': f'TCR_{donor}_{rng.integers(0, 100)}'})
    pd.DataFrame(tcr_records).to_csv(os.path.join(outdir, 'tcr', 'airr_repertoire.tsv'), sep='\t', index=False)

def reference_analysis(data_dir):
    counts = sio.mmread(os.path.join(data_dir, 'scrna', 'counts.mtx')).tocsr()
    genes = pd.read_csv(os.path.join(data_dir, 'scrna', 'genes.tsv'), header=None)[0].tolist()
    barcodes = pd.read_csv(os.path.join(data_dir, 'scrna', 'barcodes.tsv'), header=None)[0].tolist()
    meta = pd.read_csv(os.path.join(data_dir, 'scrna', 'cell_metadata.tsv'), sep='\t')
    clin = pd.read_csv(os.path.join(data_dir, 'clinical', 'participants.tsv'), sep='\t')
    gs = {}
    for f in os.listdir(os.path.join(data_dir, 'genesets')):
        name = f.replace('.txt', '')
        with open(os.path.join(data_dir, 'genesets', f)) as fh:
            gs[name] = [line.strip() for line in fh if line.strip()]
    bcr = pd.read_csv(os.path.join(data_dir, 'bcr', 'airr_repertoire.tsv'), sep='\t')
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    barcode_to_idx = {b: i for i, b in enumerate(barcodes)}
    lib_size = np.array(counts.sum(axis=0)).flatten()
    lib_size[lib_size == 0] = 1
    scale = 10000.0 / lib_size
    counts_norm = counts.multiply(scale).tocsr()
    counts_norm.data = np.log1p(counts_norm.data)
    donors = clin['donor_id'].tolist()
    titers = dict(zip(clin['donor_id'], clin['end_of_study_titer_mIU_mL']))
    groups = {d: ('High' if titers[d] >= 100 else 'Low') for d in donors}
    tfh_mask = meta['cell_type'] == 'Tfh'
    tfh_meta = meta[tfh_mask].copy()
    tfh_indices = [barcode_to_idx[b] for b in tfh_meta['cell_id']]
    gcb_mask = meta['cell_type'] == 'GC_B'
    gcb_meta = meta[gcb_mask].copy()
    gcb_indices = [barcode_to_idx[b] for b in gcb_meta['cell_id']]
    def zscore_genes(cell_indices, gene_list):
        sub = counts_norm[[gene_to_idx[g] for g in gene_list], :][:, cell_indices]
        arr = sub.toarray()
        means = arr.mean(axis=1, keepdims=True)
        stds = arr.std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        return (arr - means) / stds
    gng4_idx = gene_to_idx['GNG4']
    gng4_expr = counts_norm[gng4_idx, tfh_indices].toarray().flatten()
    gng4_plus = np.zeros(len(tfh_indices), dtype=bool)
    for d in donors:
        dmask = (tfh_meta['donor_id'] == d).values
        if dmask.sum() == 0:
            continue
        thresh = np.percentile(gng4_expr[dmask], 60)
        gng4_plus[dmask] = gng4_expr[dmask] > thresh
    tfh_meta['is_gng4_plus'] = gng4_plus
    def compute_mean_zscore(gene_list):
        z = zscore_genes(tfh_indices, gene_list)
        cell_score = z.mean(axis=0)
        res = {}
        for d in donors:
            mask = (tfh_meta['donor_id'] == d).values & gng4_plus
            res[d] = float(cell_score[mask].mean()) if mask.sum() > 0 else 0.0
        return res
    gng4_scores = compute_mean_zscore(gs['gng4_tfh_signature'])
    infl_scores = compute_mean_zscore(gs['inflammatory'])
    bcl6_act = compute_mean_zscore(gs['bcl6_regulon'])
    bcl6_expr = counts_norm[gene_to_idx['BCL6'], tfh_indices].toarray().flatten()
    regulon_targets = [g for g in gs['bcl6_regulon'] if g != 'BCL6']
    z_targets = zscore_genes(tfh_indices, regulon_targets)
    target_score = z_targets.mean(axis=0)
    bcl6_spec = {}
    for d in donors:
        mask = (tfh_meta['donor_id'] == d).values & gng4_plus
        if mask.sum() > 2:
            r, _ = pearsonr(bcl6_expr[mask], target_score[mask])
            bcl6_spec[d] = float(r)
        else:
            bcl6_spec[d] = 0.0
    def compute_gcb_score(gene_list):
        z = zscore_genes(gcb_indices, gene_list)
        cell_score = z.mean(axis=0)
        res = {}
        for d in donors:
            mask = (gcb_meta['donor_id'] == d).values
            res[d] = float(cell_score[mask].mean()) if mask.sum() > 0 else 0.0
        return res
    dz_scores = compute_gcb_score(gs['dz_markers'])
    lz_scores = compute_gcb_score(gs['lz_markers'])
    myc_scores = compute_gcb_score(gs['myc_targets'])
    mtorc1_scores = compute_gcb_score(gs['mtorc1_targets'])
    bcr['donor_id'] = bcr['cell_id'].str.split('_').str[0]
    mean_shm = bcr.groupby('donor_id')['somatic_hypermutation_rate'].mean().to_dict()
    diversity = {}
    for d, grp in bcr.groupby('donor_id'):
        cnt = grp['clonotype_id'].value_counts()
        p = cnt / cnt.sum()
        diversity[d] = float(-np.sum(p * np.log(p)))
    clono_donors = bcr.groupby('clonotype_id')['donor_id'].nunique()
    public_clonos = set(clono_donors[clono_donors >= 2].index)
    public_count = {}
    for d in donors:
        donor_clonos = set(bcr[bcr['donor_id'] == d]['clonotype_id'])
        public_count[d] = len(donor_clonos & public_clonos)
    gng4_tfh_indices = [tfh_indices[i] for i in range(len(tfh_indices)) if gng4_plus[i]]
    gng4_tfh_donor = [tfh_meta['donor_id'].values[i] for i in range(len(tfh_indices)) if gng4_plus[i]]
    gng4_counts = counts_norm[:, gng4_tfh_indices]
    high_mask = np.array([groups[d] == 'High' for d in gng4_tfh_donor])
    low_mask = ~high_mask
    de_results = []
    for i, gene in enumerate(genes):
        expr = gng4_counts[i, :].toarray().flatten()
        mh = expr[high_mask].mean()
        ml = expr[low_mask].mean()
        log2fc = np.log2((mh + 1) / (ml + 1))
        stat, pval = mannwhitneyu(expr[high_mask], expr[low_mask], alternative='two-sided')
        de_results.append({'gene': gene, 'mean_high': mh, 'mean_low': ml, 'log2_fold_change': log2fc, 'p_value': pval})
    de_df = pd.DataFrame(de_results)
    _, padj, _, _ = multipletests(de_df['p_value'], method='fdr_bh')
    de_df['padj'] = padj
    de_df['direction'] = np.where(de_df['log2_fold_change'] > 0, 'up_in_high', 'up_in_low')
    de_df = de_df.sort_values('p_value')
    titer_vals = [titers[d] for d in donors]
    cor_gng4, _ = pearsonr([gng4_scores[d] for d in donors], titer_vals)
    cor_shm, _ = pearsonr([mean_shm[d] for d in donors], titer_vals)
    cor_infl, _ = pearsonr([infl_scores[d] for d in donors], titer_vals)
    ref = {'donors': {}, 'correlations': {'gng4_score_vs_titer': float(cor_gng4), 'shm_vs_titer': float(cor_shm), 'inflammatory_vs_titer': float(cor_infl)}, 'de_genes': de_df.to_dict(orient='records')}
    for d in donors:
        ref['donors'][d] = {
            'responder_group': groups[d],
            'end_of_study_titer_mIU_mL': float(titers[d]),
            'n_tfh': int((tfh_meta['donor_id'] == d).sum()),
            'n_gng4_tfh': int(((tfh_meta['donor_id'] == d) & gng4_plus).sum()),
            'gng4_tfh_signature_score': float(gng4_scores[d]),
            'inflammatory_score': float(infl_scores[d]),
            'bcl6_regulon_activity': float(bcl6_act[d]),
            'bcl6_regulon_specificity': float(bcl6_spec[d]),
            'dz_score': float(dz_scores[d]),
            'lz_score': float(lz_scores[d]),
            'myc_score': float(myc_scores[d]),
            'mtorc1_score': float(mtorc1_scores[d]),
            'mean_shm': float(mean_shm[d]),
            'clonotype_diversity': float(diversity[d]),
            'public_clonotypes': int(public_count[d])
        }
    return ref
