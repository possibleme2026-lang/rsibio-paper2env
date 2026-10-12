#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.stats import ttest_ind

DATA = '/app/data'
OUT = '/app/output'
REWARD_PATH = '/logs/verifier/reward.json'
REPORT_PATH = '/logs/verifier/report.json'

def bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out

def load_csv(path, required_cols):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if not all(c in df.columns for c in required_cols):
        return None
    return df

def normalize_log2(mat):
    mat = mat.tocsr()
    totals = np.asarray(mat.sum(axis=0)).ravel().astype(float)
    totals[totals == 0] = 1.0
    scale = 1e4 / totals
    mat_norm = mat @ sparse.diags(scale)
    mat_norm = mat_norm.tocsr()
    mat_norm.data = np.log2(1.0 + mat_norm.data)
    return mat_norm

def pearson_corr(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A_mean = A.mean(axis=1, keepdims=True)
    B_mean = B.mean(axis=1, keepdims=True)
    A_c = A - A_mean
    B_c = B - B_mean
    A_norm = np.sqrt((A_c ** 2).sum(axis=1, keepdims=True))
    B_norm = np.sqrt((B_c ** 2).sum(axis=1, keepdims=True))
    A_norm[A_norm == 0] = 1.0
    B_norm[B_norm == 0] = 1.0
    return (A_c @ B_c.T) / (A_norm * B_norm.T)

def main():
    ref_mat = mmread(os.path.join(DATA, 'reference_counts.mtx')).tocsr()
    cult_mat = mmread(os.path.join(DATA, 'cultured_counts.mtx')).tocsr()
    genes = pd.read_csv(os.path.join(DATA, 'genes.txt'), header=None)[0].astype(str).tolist()
    ref_cells = pd.read_csv(os.path.join(DATA, 'reference_cells.csv'))
    cult_cells = pd.read_csv(os.path.join(DATA, 'cultured_cells.csv'))

    ref_totals = np.asarray(ref_mat.sum(axis=1)).ravel()
    keep = ref_totals >= 10
    n_genes_filtered = int(keep.sum())
    ref_filt = ref_mat[keep, :]
    cult_filt = cult_mat[keep, :]
    genes_filt = [g for g, k in zip(genes, keep) if k]

    ref_log = normalize_log2(ref_filt)
    cult_log = normalize_log2(cult_filt)

    ref_types = ref_cells['cell_type'].values
    unique_types = sorted(ref_cells['cell_type'].unique())
    centroids = []
    for ct in unique_types:
        idx = np.where(ref_types == ct)[0]
        centroids.append(np.asarray(ref_log[:, idx].mean(axis=1)).ravel())
    cent_mat = np.vstack(centroids)

    cult_dense = cult_log.toarray().T
    corr = pearson_corr(cult_dense, cent_mat)
    best = np.argmax(corr, axis=1)
    expected_assigned = [unique_types[i] for i in best]
    expected_assign = pd.DataFrame({
        'cell_id': cult_cells['cell_id'].values,
        'condition': cult_cells['condition'].values,
        'assigned_type': expected_assigned,
    })

    comp_rows = []
    for cond in ['Basal', 'Differentiation', 'EGF_IGF']:
        sub = expected_assign[expected_assign['condition'] == cond]
        total = len(sub)
        for ct in unique_types:
            n = int((sub['assigned_type'] == ct).sum())
            comp_rows.append({'condition': cond, 'cell_type': ct, 'n_cells': n, 'proportion': n / total})
    expected_comp = pd.DataFrame(comp_rows)

    props = expected_comp.pivot(index='condition', columns='cell_type', values='proportion').fillna(0.0)
    enrich_rows = []
    for cond in ['Basal', 'Differentiation', 'EGF_IGF']:
        others = [c for c in props.index if c != cond]
        mean_other = props.loc[others].mean(axis=0)
        log2fc = np.log2((props.loc[cond] + 1e-6) / (mean_other + 1e-6))
        enriched_type = log2fc.idxmax()
        enrich_rows.append({'condition': cond, 'enriched_type': enriched_type, 'log2FC': float(log2fc[enriched_type])})
    expected_enrich = pd.DataFrame(enrich_rows)

    sertoli = expected_assign[expected_assign['assigned_type'] == 'Sertoli']
    egf_ids = sertoli[sertoli['condition'] == 'EGF_IGF']['cell_id'].tolist()
    basal_ids = sertoli[sertoli['condition'] == 'Basal']['cell_id'].tolist()
    cid2idx = {cid: i for i, cid in enumerate(cult_cells['cell_id'])}
    egf_idx = [cid2idx[c] for c in egf_ids]
    basal_idx = [cid2idx[c] for c in basal_ids]

    if len(egf_idx) == 0 or len(basal_idx) == 0:
        expected_de = pd.DataFrame(columns=['gene', 'log2FC', 'p_value', 'adj_p_value'])
    else:
        egf_expr = cult_log[:, egf_idx].toarray()
        basal_expr = cult_log[:, basal_idx].toarray()
        mean_egf = egf_expr.mean(axis=1)
        mean_basal = basal_expr.mean(axis=1)
        log2fc = mean_egf - mean_basal
        pvals = []
        for i in range(egf_expr.shape[0]):
            if np.all(egf_expr[i] == egf_expr[i, 0]) and np.all(basal_expr[i] == basal_expr[i, 0]):
                pvals.append(1.0)
            else:
                _, p = ttest_ind(egf_expr[i], basal_expr[i], equal_var=False)
                pvals.append(1.0 if np.isnan(p) else float(p))
        pvals = np.array(pvals)
        adj = bh_adjust(pvals)
        de = pd.DataFrame({
            'gene': genes_filt,
            'log2FC': log2fc,
            'p_value': pvals,
            'adj_p_value': adj,
        })
        de_up = de[de['log2FC'] > 0].copy()
        de_up = de_up.sort_values(by=['adj_p_value', 'log2FC', 'gene'], ascending=[True, False, True])
        expected_de = de_up.head(10).reset_index(drop=True)

    scores = {}

    qc_path = os.path.join(OUT, 'qc_summary.json')
    qc_score = 0.0
    if os.path.exists(qc_path):
        try:
            with open(qc_path) as f:
                qc = json.load(f)
            checks = [
                qc.get('n_genes_filtered') == n_genes_filtered,
                qc.get('n_reference_cells') == len(ref_cells),
                qc.get('n_cultured_cells') == len(cult_cells),
                qc.get('n_sertoli_EGF_IGF') == len(egf_ids),
                qc.get('n_sertoli_Basal') == len(basal_ids),
            ]
            qc_score = sum(checks) / len(checks)
        except Exception:
            import traceback as _tb, sys as _sys
            _err = _tb.format_exc()
            try:
                import os as _os
                _os.makedirs('/logs/verifier', exist_ok=True)
                open('/logs/verifier/grader_error.txt','w').write(_err)
            except Exception:
                pass
            print('GRADER_EXCEPTION\n' + _err, file=_sys.stderr, flush=True)
            qc_score = 0.0
    scores['qc_summary'] = float(qc_score)

    assign_path = os.path.join(OUT, 'assignments.csv')
    assign = load_csv(assign_path, ['cell_id', 'condition', 'assigned_type'])
    assign_score = 0.0
    if assign is not None:
        assign = assign.drop_duplicates(subset=['cell_id'])
        expected_ids = set(cult_cells['cell_id'])
        got_ids = set(assign['cell_id'])
        valid_types = set(unique_types)
        completeness = len(got_ids & expected_ids) / len(expected_ids) if expected_ids else 0.0
        type_valid = assign['assigned_type'].isin(valid_types).mean() if len(assign) > 0 else 0.0
        if len(assign) == len(cult_cells) and got_ids == expected_ids:
            assign_score = 1.0 * type_valid
        else:
            assign_score = 0.5 * completeness + 0.5 * type_valid
    scores['assignments_valid'] = float(assign_score)

    map_score = 0.0
    if assign is not None:
        merged = expected_assign.merge(assign, on='cell_id', suffixes=('_exp', '_got'))
        if len(merged) > 0:
            map_score = float((merged['assigned_type_exp'] == merged['assigned_type_got']).mean())
        else:
            map_score = 0.0
    scores['assignment_accuracy'] = float(map_score)

    comp_path = os.path.join(OUT, 'composition.csv')
    comp = load_csv(comp_path, ['condition', 'cell_type', 'n_cells', 'proportion'])
    comp_score = 0.0
    if comp is not None:
        comp = comp.drop_duplicates(subset=['condition', 'cell_type'])
        merged = expected_comp.merge(comp, on=['condition', 'cell_type'], suffixes=('_exp', '_got'), how='left')
        n_ok = (merged['n_cells_exp'] == merged['n_cells_got']).sum()
        p_ok = (np.abs(merged['proportion_exp'] - merged['proportion_got']) <= 1e-6).sum()
        comp_score = float((n_ok + p_ok) / (2 * len(expected_comp)))
    scores['composition'] = float(comp_score)

    enrich_path = os.path.join(OUT, 'enrichment.csv')
    enrich = load_csv(enrich_path, ['condition', 'enriched_type', 'log2FC'])
    enrich_score = 0.0
    if enrich is not None:
        enrich = enrich.drop_duplicates(subset=['condition'])
        merged = expected_enrich.merge(enrich, on='condition', suffixes=('_exp', '_got'), how='left')
        type_ok = (merged['enriched_type_exp'] == merged['enriched_type_got']).sum()
        fc_ok = (np.abs(merged['log2FC_exp'] - merged['log2FC_got']) <= 1e-2).sum()
        enrich_score = float((type_ok + fc_ok) / (2 * len(expected_enrich)))
    scores['enrichment'] = float(enrich_score)

    de_path = os.path.join(OUT, 'sertoli_de.csv')
    de_out = load_csv(de_path, ['gene', 'log2FC', 'p_value', 'adj_p_value'])
    de_score = 0.0
    if de_out is not None and len(expected_de) > 0:
        de_out = de_out.drop_duplicates(subset=['gene'])
        expected_genes = set(expected_de['gene'])
        got_genes = set(de_out['gene'])
        overlap = len(expected_genes & got_genes)
        gene_score = overlap / len(expected_genes) if expected_genes else 0.0
        if overlap > 0:
            merged = expected_de.merge(de_out, on='gene', suffixes=('_exp', '_got'))
            fc_ok = (np.abs(merged['log2FC_exp'] - merged['log2FC_got']) <= 1e-2).mean()
            p_ok = (np.abs(merged['adj_p_value_exp'] - merged['adj_p_value_got']) <= 1e-3).mean()
            de_score = float(0.5 * gene_score + 0.25 * fc_ok + 0.25 * p_ok)
        else:
            de_score = 0.0
    scores['sertoli_de'] = float(de_score)

    total = float(np.mean(list(scores.values())))
    report = {
        'scores': scores,
        'total_score': total,
        'n_genes_filtered_expected': n_genes_filtered,
        'n_sertoli_egf_igf_expected': len(egf_ids),
        'n_sertoli_basal_expected': len(basal_ids),
    }

    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(REWARD_PATH, 'w') as f:
        json.dump({'score': total, 'reward': total}, f)
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
