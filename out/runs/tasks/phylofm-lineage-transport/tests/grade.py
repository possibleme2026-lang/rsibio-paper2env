import os, json, numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('/tests')
import gold_gen

def load_agent_outputs():
    out_dir = '/app/output'
    transport = np.load(os.path.join(out_dir, 'transport.npz'))
    T_list = [transport[f'T_{t}'] for t in range(4)]
    fate_df = pd.read_csv(os.path.join(out_dir, 'fate_probabilities.csv'))
    commit_df = pd.read_csv(os.path.join(out_dir, 'commitment_times.csv'))
    module_df = pd.read_csv(os.path.join(out_dir, 'module_dynamics.csv'))
    return T_list, fate_df, commit_df, module_df

def main():
    gold = gold_gen.compute_gold()
    T_gold = gold['T']
    fate_gold = gold['fate_probs']
    commit_gold = gold['commitment_times']
    module_gold = gold['module_dynamics']
    true_fate = gold['true_fate']

    T_agent, fate_df, commit_df, module_df = load_agent_outputs()

    passes = []
    details = {}

    valid = True
    for t, T in enumerate(T_agent):
        if np.any(T < -1e-9): valid = False
        if not np.allclose(T.sum(axis=1), 1, atol=1e-6): valid = False
        if not np.allclose(T.sum(axis=0), 1, atol=1e-6): valid = False
    passes.append(valid)
    details['transport_valid'] = valid

    meta = pd.read_csv('/app/data/cell_metadata.csv')
    lineage_ok = True
    for t in range(4):
        meta_t = meta[meta['timepoint'] == t].reset_index(drop=True)
        meta_t1 = meta[meta['timepoint'] == t+1].reset_index(drop=True)
        T = T_agent[t]
        global_avg = T.mean()
        clone_masses = []
        for clone in meta_t['clone_id'].unique():
            idx_t = meta_t.index[meta_t['clone_id'] == clone].tolist()
            idx_t1 = meta_t1.index[meta_t1['clone_id'] == clone].tolist()
            if len(idx_t) > 0 and len(idx_t1) > 0:
                mass = T[np.ix_(idx_t, idx_t1)].mean()
                clone_masses.append(mass)
        if len(clone_masses) > 0:
            ratios = np.array(clone_masses) / (global_avg + 1e-12)
            frac_good = np.mean(ratios >= 3.0)
            if frac_good < 0.9:
                lineage_ok = False
    passes.append(lineage_ok)
    details['lineage_consistency'] = lineage_ok

    merged = fate_df.merge(true_fate, on='cell_id')
    aurocs = []
    for k in range(3):
        y_true = (merged['true_fate'] == k).astype(int)
        y_score = merged[f'fate_{k}']
        if y_true.nunique() > 1:
            aurocs.append(roc_auc_score(y_true, y_score))
    macro_auroc = np.mean(aurocs) if aurocs else 0.0
    fate_ok = (macro_auroc >= 0.85)
    prob_cols = ['fate_0', 'fate_1', 'fate_2']
    if np.any(merged[prob_cols] < -1e-9): fate_ok = False
    if not np.allclose(merged[prob_cols].sum(axis=1), 1, atol=1e-6): fate_ok = False
    passes.append(fate_ok)
    details['fate_auroc'] = macro_auroc

    commit_ok = True
    for branch in range(3):
        agent_time = commit_df[commit_df['branch'] == branch]['commitment_time'].values
        gold_time = commit_gold[branch]
        if len(agent_time) == 0 or abs(agent_time[0] - gold_time) > 1:
            commit_ok = False
    gold_order = sorted(commit_gold.keys(), key=lambda k: (commit_gold[k], k))
    agent_times = {row['branch']: row['commitment_time'] for _, row in commit_df.iterrows()}
    agent_order = sorted(agent_times.keys(), key=lambda k: (agent_times[k], k))
    if agent_order != gold_order:
        commit_ok = False
    passes.append(commit_ok)
    details['commitment_ok'] = commit_ok

    merged_mod = module_df.merge(module_gold, on=['branch', 'timepoint', 'module'], suffixes=('_agent', '_gold'))
    module_corrs = []
    for mod in module_gold['module'].unique():
        mod_corrs = []
        for branch in range(3):
            sub = merged_mod[(merged_mod['module'] == mod) & (merged_mod['branch'] == branch)]
            if len(sub) >= 3:
                r, _ = pearsonr(sub['mean_expression_agent'], sub['mean_expression_gold'])
                if not np.isnan(r):
                    mod_corrs.append(r)
        if mod_corrs:
            module_corrs.append(np.mean(mod_corrs))
    avg_r = np.mean(module_corrs) if module_corrs else 0.0
    frac_good = np.mean([r >= 0.6 for r in module_corrs]) if module_corrs else 0.0
    module_ok = (avg_r >= 0.7) and (frac_good >= 0.75)
    passes.append(module_ok)
    details['module_corr'] = avg_r

    transport_errors = []
    for t in range(4):
        T_a = T_agent[t]
        T_g = T_gold[t]
        if T_a.shape != T_g.shape:
            transport_errors.append(1.0)
        else:
            err = np.linalg.norm(T_a - T_g) / np.sqrt(T_a.size)
            transport_errors.append(err)
    avg_err = np.mean(transport_errors)
    transport_ok = (avg_err <= 1e-4)
    passes.append(transport_ok)
    details['transport_error'] = avg_err

    score = sum(passes) / len(passes)
    reward = score
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': score, 'reward': reward, 'details': details}, f)
    print(json.dumps({'score': score, 'reward': reward, 'details': details}, indent=2))

if __name__ == '__main__':
    main()
