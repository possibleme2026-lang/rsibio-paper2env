#!/usr/bin/env python3
import os, sys, json, math
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gold_gen

RESULTS_DIR = '/app/results'
LOG_DIR = '/logs/verifier'
os.makedirs(LOG_DIR, exist_ok=True)

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def check_format_and_constraints(comp_df, scores_df, loadings):
    details = {}
    score_parts = []
    total_parts = 0
    required_comp_cols = {'component_id', 'sharing_type', 'modalities', 'variance_explained'}
    required_scores_cols = {'sample_id', 'component_id', 'score'}
    required_load_cols = {'feature_id', 'component_id', 'loading'}
    format_ok = True
    if comp_df is None or not required_comp_cols.issubset(comp_df.columns):
        format_ok = False
    if scores_df is None or not required_scores_cols.issubset(scores_df.columns):
        format_ok = False
    for mod, df in loadings.items():
        if df is None or not required_load_cols.issubset(df.columns):
            format_ok = False
    details['format'] = format_ok
    score_parts.append(1.0 if format_ok else 0.0)
    total_parts += 1
    if not format_ok:
        return sum(score_parts) / total_parts, total_parts, details
    n_comp = len(comp_df)
    comp_count_ok = 10 <= n_comp <= 16
    sharing_consistent = True
    for _, row in comp_df.iterrows():
        st = row['sharing_type']
        mods = str(row['modalities']).split(';')
        if st == 'joint' and len(mods) != 3:
            sharing_consistent = False
        elif st == 'partial' and len(mods) != 2:
            sharing_consistent = False
        elif st == 'individual' and len(mods) != 1:
            sharing_consistent = False
        elif st not in ['joint', 'partial', 'individual']:
            sharing_consistent = False
    details['n_components'] = n_comp
    details['comp_count_ok'] = comp_count_ok
    details['sharing_consistent'] = sharing_consistent
    score_parts.append(1.0 if comp_count_ok else 0.0)
    score_parts.append(1.0 if sharing_consistent else 0.0)
    total_parts += 2
    score_std_ok = True
    scores_pivot = scores_df.pivot_table(index='sample_id', columns='component_id', values='score')
    for comp_id in scores_pivot.columns:
        vals = scores_pivot[comp_id].dropna()
        if len(vals) > 1:
            mean = np.abs(vals.mean())
            sd = vals.std()
            if mean > 0.1 or sd < 0.9 or sd > 1.1:
                score_std_ok = False
                break
    details['score_std_ok'] = score_std_ok
    score_parts.append(1.0 if score_std_ok else 0.0)
    total_parts += 1
    orth_ok = True
    if scores_pivot.shape[1] > 1:
        corr_mat = scores_pivot.corr().abs()
        np.fill_diagonal(corr_mat.values, 0)
        max_corr = corr_mat.max().max()
        if max_corr > 0.3:
            orth_ok = False
    details['orthogonality'] = orth_ok
    score_parts.append(1.0 if orth_ok else 0.0)
    total_parts += 1
    total_var = 0.0
    for mod in ['transcriptomics', 'proteomics', 'metabolomics']:
        path = f'/app/data/{mod}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'sample_id' in df.columns:
                df = df.drop(columns=['sample_id'])
            total_var += df.var(ddof=0).sum()
        else:
            total_var = None
            break
    var_exp_ok = True
    if total_var is not None and total_var > 0:
        for _, row in comp_df.iterrows():
            comp_id = row['component_id']
            stated_ve = row['variance_explained']
            score_vals = scores_pivot[comp_id].dropna().values
            if len(score_vals) < 2:
                continue
            score_var = np.var(score_vals, ddof=0)
            comp_var = 0.0
            mods = str(row['modalities']).split(';')
            for mod in mods:
                mod_key = mod.strip()
                load_df = loadings.get(mod_key)
                if load_df is None:
                    continue
                sub = load_df[load_df['component_id'] == comp_id]
                if len(sub) == 0:
                    continue
                load_vals = sub['loading'].values
                comp_var += np.sum(load_vals**2) * score_var
            for mod in ['transcriptomics', 'proteomics', 'metabolomics']:
                if mod not in mods:
                    load_df = loadings.get(mod)
                    if load_df is None:
                        continue
                    sub = load_df[load_df['component_id'] == comp_id]
                    if len(sub) > 0:
                        load_vals = sub['loading'].values
                        comp_var += np.sum(load_vals**2) * score_var
            recomputed_ve = comp_var / total_var
            if stated_ve <= 0:
                var_exp_ok = False
                break
            rel_err = abs(recomputed_ve - stated_ve) / max(stated_ve, 1e-6)
            if rel_err > 0.3:
                var_exp_ok = False
                break
    else:
        var_exp_ok = False
    details['var_exp_ok'] = var_exp_ok
    score_parts.append(1.0 if var_exp_ok else 0.0)
    total_parts += 1
    sum_ve = comp_df['variance_explained'].sum()
    sum_ve_ok = 0.5 <= sum_ve <= 1.0
    details['sum_ve'] = sum_ve
    details['sum_ve_ok'] = sum_ve_ok
    score_parts.append(1.0 if sum_ve_ok else 0.0)
    total_parts += 1
    mod_consistency = True
    for _, row in comp_df.iterrows():
        comp_id = row['component_id']
        mods = set(str(row['modalities']).split(';'))
        for mod in ['transcriptomics', 'proteomics', 'metabolomics']:
            load_df = loadings.get(mod)
            if load_df is None:
                continue
            sub = load_df[load_df['component_id'] == comp_id]
            if mod in mods:
                if len(sub) == 0 or sub['loading'].abs().max() < 0.02:
                    mod_consistency = False
            else:
                if len(sub) > 0 and sub['loading'].abs().max() > 0.02:
                    mod_consistency = False
    details['mod_consistency'] = mod_consistency
    score_parts.append(1.0 if mod_consistency else 0.0)
    total_parts += 1
    return sum(score_parts) / total_parts, total_parts, details

def match_components(true_comps, agent_comp_df, agent_scores_pivot):
    true_ids = [c['id'] for c in true_comps]
    true_scores = np.column_stack([c['scores'] for c in true_comps])
    agent_ids = agent_scores_pivot.columns.tolist()
    agent_scores = agent_scores_pivot.values
    n_true = len(true_ids)
    n_agent = len(agent_ids)
    if n_agent == 0 or n_true == 0:
        return 0.0, 0.0, {}
    corr_mat = np.zeros((n_true, n_agent))
    for i in range(n_true):
        for j in range(n_agent):
            if np.std(true_scores[:, i]) > 0 and np.std(agent_scores[:, j]) > 0:
                corr_mat[i, j] = abs(np.corrcoef(true_scores[:, i], agent_scores[:, j])[0, 1])
            else:
                corr_mat[i, j] = 0.0
    row_ind, col_ind = linear_sum_assignment(-corr_mat)
    tp = 0
    correct_mod = 0
    matches = []
    for i, j in zip(row_ind, col_ind):
        if corr_mat[i, j] >= 0.7:
            tp += 1
            true_mods = set(true_comps[i]['modalities'])
            agent_mods = set(str(agent_comp_df.loc[agent_comp_df['component_id'] == agent_ids[j], 'modalities'].values[0]).split(';'))
            if true_mods == agent_mods:
                correct_mod += 1
            matches.append({
                'true_id': true_ids[i],
                'agent_id': agent_ids[j],
                'corr': float(corr_mat[i, j]),
                'true_mods': list(true_mods),
                'agent_mods': list(agent_mods),
                'mod_match': true_mods == agent_mods
            })
    precision = tp / n_agent if n_agent > 0 else 0.0
    recall = tp / n_true if n_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    sharing_acc = correct_mod / tp if tp > 0 else 0.0
    details = {
        'tp': tp,
        'n_true': n_true,
        'n_agent': n_agent,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sharing_acc': sharing_acc,
        'matches': matches
    }
    return f1, sharing_acc, details

def main():
    true_comps = gold_gen.get_true_components()
    comp_path = os.path.join(RESULTS_DIR, 'components.csv')
    scores_path = os.path.join(RESULTS_DIR, 'scores.csv')
    load_paths = {
        'transcriptomics': os.path.join(RESULTS_DIR, 'loadings_transcriptomics.csv'),
        'proteomics': os.path.join(RESULTS_DIR, 'loadings_proteomics.csv'),
        'metabolomics': os.path.join(RESULTS_DIR, 'loadings_metabolomics.csv')
    }
    comp_df = load_csv(comp_path)
    scores_df = load_csv(scores_path)
    loadings = {mod: load_csv(path) for mod, path in load_paths.items()}
    format_score, total_parts, format_details = check_format_and_constraints(comp_df, scores_df, loadings)
    f1 = 0.0
    sharing_acc = 0.0
    match_details = {}
    if comp_df is not None and scores_df is not None:
        try:
            scores_pivot = scores_df.pivot_table(index='sample_id', columns='component_id', values='score')
            f1, sharing_acc, match_details = match_components(true_comps, comp_df, scores_pivot)
        except Exception as e:
            match_details = {'error': str(e)}
    final_score = 0.3 * format_score + 0.5 * f1 + 0.2 * sharing_acc
    reward = {
        'score': round(final_score, 4),
        'reward': round(final_score, 4)
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, 'reward.json'), 'w') as f:
        json.dump(reward, f)
    report = {
        'format_score': format_score,
        'format_details': format_details,
        'f1': f1,
        'sharing_accuracy': sharing_acc,
        'match_details': match_details,
        'final_score': final_score
    }
    with open(os.path.join(LOG_DIR, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))

if __name__ == '__main__':
    main()
