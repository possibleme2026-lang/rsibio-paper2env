#!/usr/bin/env python3
import os, sys, json, subprocess, shutil
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, false_discovery_control
from skbio.stats.distance import permanova
from skbio import DistanceMatrix

sys.path.append('/tests')
import gold_gen

REWARD_PATH = '/logs/verifier/reward.json'
REPORT_PATH = '/logs/verifier/report.json'

def load_json(path):
    with open(path) as f:
        return json.load(f)

def compute_permanova_r2(dist_matrix, labels):
    labels = np.asarray(labels)
    n = len(labels)
    D2 = dist_matrix.to_data_frame() ** 2
    SS_T = D2.values.sum() / (2 * n)
    unique_groups = np.unique(labels)
    SS_W = 0.0
    for g in unique_groups:
        idx = np.where(labels == g)[0]
        n_g = len(idx)
        sub = D2.iloc[idx, idx]
        SS_W += sub.values.sum() / (2 * n_g)
    SS_A = SS_T - SS_W
    return SS_A / SS_T if SS_T > 0 else 0.0

def diff_abundance(df, group):
    results = []
    for col in df.columns:
        a = df.loc[group==0, col].values
        b = df.loc[group==1, col].values
        mean_a = np.mean(a) + 1e-9
        mean_b = np.mean(b) + 1e-9
        log2fc = np.log2(mean_b / mean_a)
        stat, p = mannwhitneyu(a, b, alternative='two-sided')
        results.append((col, log2fc, p))
    res = pd.DataFrame(results, columns=['feature', 'log2FC', 'pvalue'])
    res['qvalue'] = false_discovery_control(res['pvalue'])
    return res

def main():
    hidden_dir = '/tmp/hidden_data'
    if os.path.exists(hidden_dir):
        shutil.rmtree(hidden_dir)
    gold_gen.generate(hidden_dir, seed=2027, save_truth=True)
    truth = load_json(os.path.join(hidden_dir, 'truth.json'))
    out_dir = '/tmp/agent_out'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    agent_script = '/app/analysis.py'
    if not os.path.exists(agent_script):
        result = {'score': 0.0, 'reward': 0.0}
        os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
        with open(REWARD_PATH, 'w') as f:
            json.dump(result, f)
        with open(REPORT_PATH, 'w') as f:
            json.dump({'error': 'agent script not found'}, f)
        print(json.dumps({'error': 'agent script not found', 'score': 0.0}))
        return
    try:
        subprocess.run(['python', agent_script, '--data-dir', hidden_dir, '--out-dir', out_dir],
                       check=True, timeout=600, capture_output=True, text=True)
    except Exception as e:
        result = {'score': 0.0, 'reward': 0.0}
        os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
        with open(REWARD_PATH, 'w') as f:
            json.dump(result, f)
        with open(REPORT_PATH, 'w') as f:
            json.dump({'error': str(e), 'score': 0.0}, f)
        print(json.dumps({'error': str(e), 'score': 0.0}))
        return
    results_path = os.path.join(out_dir, 'results.json')
    if not os.path.exists(results_path):
        result = {'score': 0.0, 'reward': 0.0}
        os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
        with open(REWARD_PATH, 'w') as f:
            json.dump(result, f)
        with open(REPORT_PATH, 'w') as f:
            json.dump({'error': 'results.json not found', 'score': 0.0}, f)
        print(json.dumps({'error': 'results.json not found', 'score': 0.0}))
        return
    results = load_json(results_path)
    checks = []
    def add_check(name, passed, detail=''):
        checks.append({'name': name, 'passed': bool(passed), 'detail': str(detail)})
    required_keys = [
        'facility_cv_accuracy', 'treatment_cv_accuracy',
        'facility_variance_explained', 'treatment_variance_explained',
        'n_signature_species',
        'n_differential_species_facility', 'n_differential_species_treatment',
        'n_differential_genes_facility', 'n_differential_genes_treatment',
        'n_differential_metabolites_facility', 'n_differential_metabolites_treatment'
    ]
    missing = [k for k in required_keys if k not in results]
    add_check('results_keys', len(missing)==0, f'missing={missing}')
    acc_fac = results.get('facility_cv_accuracy', -1)
    add_check('facility_cv_accuracy', isinstance(acc_fac, (int, float)) and 0.85 <= acc_fac <= 1.0, f'value={acc_fac}')
    acc_trt = results.get('treatment_cv_accuracy', -1)
    add_check('treatment_cv_accuracy', isinstance(acc_trt, (int, float)) and acc_trt <= 0.70, f'value={acc_trt}')
    r2_fac = results.get('facility_variance_explained', -1)
    add_check('facility_variance', isinstance(r2_fac, (int, float)) and r2_fac >= 0.15, f'value={r2_fac}')
    r2_trt = results.get('treatment_variance_explained', -1)
    add_check('treatment_variance', isinstance(r2_trt, (int, float)) and r2_trt < 0.05, f'value={r2_trt}')
    sig_path = os.path.join(out_dir, 'signature_species.txt')
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig_list = [line.strip() for line in f if line.strip()]
        n_sig = len(sig_list)
        add_check('signature_count', 15 <= n_sig <= 30, f'n={n_sig}')
        sp_cols = pd.read_csv(os.path.join(hidden_dir, 'microbiome_species.csv'), nrows=1).columns[1:].tolist()
        valid = all(s in sp_cols for s in sig_list)
        add_check('signature_valid_ids', valid, f'valid={valid}')
        true_sig = set(truth['signature_species'])
        overlap = len(set(sig_list) & true_sig)
        add_check('signature_overlap', overlap >= 0.5 * len(true_sig), f'overlap={overlap}/{len(true_sig)}')
    else:
        add_check('signature_count', False, 'file missing')
        add_check('signature_valid_ids', False, 'file missing')
        add_check('signature_overlap', False, 'file missing')
    diff_sp_path = os.path.join(out_dir, 'differential_species.csv')
    if os.path.exists(diff_sp_path):
        df_sp = pd.read_csv(diff_sp_path)
        required_cols = ['species_id', 'log2FC_facility', 'pvalue_facility', 'qvalue_facility',
                         'log2FC_treatment', 'pvalue_treatment', 'qvalue_treatment']
        has_cols = all(c in df_sp.columns for c in required_cols)
        add_check('diff_species_cols', has_cols, f'cols={list(df_sp.columns)}')
        if has_cols:
            n_fac = int((df_sp['qvalue_facility'] < 0.05).sum())
            n_trt = int((df_sp['qvalue_treatment'] < 0.05).sum())
            reported_fac = results.get('n_differential_species_facility', -1)
            reported_trt = results.get('n_differential_species_treatment', -1)
            add_check('diff_species_facility_count', n_fac >= 20 and n_fac == reported_fac, f'computed={n_fac}, reported={reported_fac}')
            add_check('diff_species_treatment_count', n_trt <= 20 and n_trt == reported_trt, f'computed={n_trt}, reported={reported_trt}')
            q_fac = df_sp['qvalue_facility'].values
            q_trt = df_sp['qvalue_treatment'].values
            q_ok = np.all((q_fac >= 0) & (q_fac <= 1)) and np.all((q_trt >= 0) & (q_trt <= 1))
            add_check('diff_species_qvalues', q_ok, 'qvalues in [0,1]')
        else:
            add_check('diff_species_facility_count', False, 'missing cols')
            add_check('diff_species_treatment_count', False, 'missing cols')
            add_check('diff_species_qvalues', False, 'missing cols')
    else:
        add_check('diff_species_cols', False, 'file missing')
        add_check('diff_species_facility_count', False, 'file missing')
        add_check('diff_species_treatment_count', False, 'file missing')
        add_check('diff_species_qvalues', False, 'file missing')
    n_gene_fac = results.get('n_differential_genes_facility', -1)
    n_gene_trt = results.get('n_differential_genes_treatment', -1)
    add_check('diff_genes_facility', isinstance(n_gene_fac, (int, float)) and n_gene_fac >= 800, f'value={n_gene_fac}')
    add_check('diff_genes_treatment', isinstance(n_gene_trt, (int, float)) and n_gene_trt <= 150, f'value={n_gene_trt}')
    n_met_fac = results.get('n_differential_metabolites_facility', -1)
    n_met_trt = results.get('n_differential_metabolites_treatment', -1)
    add_check('diff_metabolites_facility', isinstance(n_met_fac, (int, float)) and n_met_fac >= 50, f'value={n_met_fac}')
    add_check('diff_metabolites_treatment', isinstance(n_met_trt, (int, float)) and n_met_trt <= 30, f'value={n_met_trt}')
    vp_path = os.path.join(out_dir, 'variance_partition.csv')
    if os.path.exists(vp_path):
        vp = pd.read_csv(vp_path)
        terms_lower = [str(t).lower() for t in vp['term'].tolist()]
        has_terms = 'facility' in terms_lower and 'treatment' in terms_lower
        add_check('variance_partition_terms', has_terms, f'terms={vp["term"].tolist()}')
        r2_ok = vp['R2'].between(0, 1).all()
        add_check('variance_partition_r2', r2_ok, 'R2 in [0,1]')
    else:
        add_check('variance_partition_terms', False, 'file missing')
        add_check('variance_partition_r2', False, 'file missing')
    total = len(checks)
    passed = sum(1 for c in checks if c['passed'])
    score = passed / total if total > 0 else 0.0
    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(REWARD_PATH, 'w') as f:
        json.dump({'score': score, 'reward': score}, f)
    with open(REPORT_PATH, 'w') as f:
        json.dump({'checks': checks, 'score': score, 'reward': score}, f, indent=2)
    print(json.dumps({'checks': checks, 'score': score, 'reward': score}, indent=2))

if __name__ == '__main__':
    main()
