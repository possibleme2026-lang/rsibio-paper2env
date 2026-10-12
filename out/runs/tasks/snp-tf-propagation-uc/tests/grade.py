import os
import json
import numpy as np
import pandas as pd
from scipy import stats

INPUT_DIR = '/app/data'
OUTPUT_DIR = '/app/output'
REWARD_FILE = '/logs/verifier/reward.json'

def load_inputs():
    geno = pd.read_csv(os.path.join(INPUT_DIR, 'genotypes.tsv'), sep='\t')
    snp_map = pd.read_csv(os.path.join(INPUT_DIR, 'snp_tf_map.tsv'), sep='\t')
    net = pd.read_csv(os.path.join(INPUT_DIR, 'tf_network.tsv'), sep='\t', index_col=0)
    clin = pd.read_csv(os.path.join(INPUT_DIR, 'clinical.tsv'), sep='\t')
    return geno, snp_map, net, clin

def compute_expected(geno, snp_map, net, clin):
    sample_ids = geno['sample_id'].values
    snp_cols = [c for c in geno.columns if c != 'sample_id']
    geno_mat = geno[snp_cols].values
    tf_ids = net.index.tolist()
    n_tfs = len(tf_ids)
    n_samples = geno_mat.shape[0]
    P = np.zeros((n_samples, n_tfs))
    tf_to_idx = {tf: i for i, tf in enumerate(tf_ids)}
    for _, row in snp_map.iterrows():
        snp = row['snp_id']
        tf = row['tf_id']
        w = row['effect_weight']
        if snp not in snp_cols or tf not in tf_to_idx:
            continue
        snp_idx = snp_cols.index(snp)
        tf_idx = tf_to_idx[tf]
        P[:, tf_idx] += geno_mat[:, snp_idx] * w
    W = net.values.astype(float)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_norm = W / row_sums
    alpha = 0.7
    I = np.eye(n_tfs)
    A = I - alpha * W_norm
    B = (1 - alpha) * P.T
    F_T = np.linalg.solve(A, B)
    F = F_T.T
    Z = (F - F.mean(axis=0)) / F.std(axis=0, ddof=0)
    severity = clin.set_index('sample_id').loc[sample_ids, 'severity'].values
    correlations = []
    for i, tf in enumerate(tf_ids):
        r, p = stats.pearsonr(Z[:, i], severity)
        correlations.append({'tf': tf, 'pearson_r': r, 'p_value': p})
    corr_df = pd.DataFrame(correlations)
    corr_df['abs_r'] = corr_df['pearson_r'].abs()
    top = corr_df.sort_values('abs_r', ascending=False).head(3).reset_index(drop=True)
    top['rank'] = top.index + 1
    top = top[['rank', 'tf', 'abs_r']]
    return sample_ids, tf_ids, Z, corr_df, top

def check_outputs(sample_ids, tf_ids, Z, corr_df, top):
    score = 0.0
    max_score = 0.0
    details = {}
    max_score += 1.0
    try:
        act = pd.read_csv(os.path.join(OUTPUT_DIR, 'tf_activity.tsv'), sep='\t')
        if act.shape[0] != len(sample_ids) or act.shape[1] != len(tf_ids) + 1:
            details['tf_activity'] = 'wrong shape'
        elif set(act['sample_id']) != set(sample_ids):
            details['tf_activity'] = 'wrong sample ids'
        else:
            act = act.set_index('sample_id').loc[sample_ids].reset_index()
            ok = True
            for i, tf in enumerate(tf_ids):
                col = act[tf].values
                if np.isnan(col).any():
                    ok = False
                    break
                mean = np.mean(col)
                std = np.std(col, ddof=0)
                if abs(mean) > 0.01 or abs(std - 1.0) > 0.01:
                    ok = False
                    break
            if ok:
                score += 1.0
                details['tf_activity'] = 'passed'
            else:
                details['tf_activity'] = 'z-score check failed'
    except Exception as e:
        details['tf_activity'] = f'error: {str(e)}'
    max_score += 1.0
    try:
        corr = pd.read_csv(os.path.join(OUTPUT_DIR, 'correlations.tsv'), sep='\t')
        if corr.shape[0] != len(tf_ids) or set(corr['tf']) != set(tf_ids):
            details['correlations'] = 'wrong shape or tf ids'
        else:
            corr = corr.set_index('tf').loc[tf_ids].reset_index()
            ok = True
            for i, row in corr_df.iterrows():
                tf = row['tf']
                expected_r = row['pearson_r']
                expected_p = row['p_value']
                got = corr[corr['tf'] == tf].iloc[0]
                if abs(got['pearson_r'] - expected_r) > 1e-4 or abs(got['p_value'] - expected_p) > 1e-4:
                    ok = False
                    break
            if ok:
                score += 1.0
                details['correlations'] = 'passed'
            else:
                details['correlations'] = 'value mismatch'
    except Exception as e:
        details['correlations'] = f'error: {str(e)}'
    max_score += 1.0
    try:
        top_got = pd.read_csv(os.path.join(OUTPUT_DIR, 'top_tfs.tsv'), sep='\t')
        if top_got.shape[0] != 3:
            details['top_tfs'] = 'wrong number of rows'
        else:
            ok = True
            for i in range(3):
                expected_tf = top.iloc[i]['tf']
                expected_abs_r = top.iloc[i]['abs_r']
                got_tf = top_got.iloc[i]['tf']
                got_abs_r = top_got.iloc[i]['abs_r']
                if got_tf != expected_tf or abs(got_abs_r - expected_abs_r) > 1e-4:
                    ok = False
                    break
            if ok:
                score += 1.0
                details['top_tfs'] = 'passed'
            else:
                details['top_tfs'] = 'mismatch'
    except Exception as e:
        details['top_tfs'] = f'error: {str(e)}'
    max_score += 1.0
    try:
        act = pd.read_csv(os.path.join(OUTPUT_DIR, 'tf_activity.tsv'), sep='\t')
        if act.isnull().values.any():
            details['no_nan'] = 'tf_activity has NaN'
        else:
            corr = pd.read_csv(os.path.join(OUTPUT_DIR, 'correlations.tsv'), sep='\t')
            if corr.isnull().values.any():
                details['no_nan'] = 'correlations has NaN'
            else:
                top_got = pd.read_csv(os.path.join(OUTPUT_DIR, 'top_tfs.tsv'), sep='\t')
                if top_got.isnull().values.any():
                    details['no_nan'] = 'top_tfs has NaN'
                else:
                    if list(act.columns) != ['sample_id'] + tf_ids:
                        details['no_nan'] = 'tf_activity wrong headers'
                    elif list(corr.columns) != ['tf', 'pearson_r', 'p_value']:
                        details['no_nan'] = 'correlations wrong headers'
                    elif list(top_got.columns) != ['rank', 'tf', 'abs_r']:
                        details['no_nan'] = 'top_tfs wrong headers'
                    else:
                        score += 1.0
                        details['no_nan'] = 'passed'
    except Exception as e:
        details['no_nan'] = f'error: {str(e)}'
    final_score = score / max_score if max_score > 0 else 0.0
    return final_score, details

def main():
    os.makedirs(os.path.dirname(REWARD_FILE), exist_ok=True)
    try:
        geno, snp_map, net, clin = load_inputs()
        sample_ids, tf_ids, Z, corr_df, top = compute_expected(geno, snp_map, net, clin)
        score, details = check_outputs(sample_ids, tf_ids, Z, corr_df, top)
    except Exception as e:
        score = 0.0
        details = {'error': str(e)}
    with open(REWARD_FILE, 'w') as f:
        json.dump({'score': score, 'reward': score}, f)
    print(json.dumps(details, indent=2))
    print(f"Final score: {score}")

if __name__ == '__main__':
    main()
