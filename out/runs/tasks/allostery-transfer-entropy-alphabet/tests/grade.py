import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('/tests')
from gold_gen import *

def load_agent_tracks(path):
    df = pd.read_csv(path, sep='\t')
    return df

def check_constraint_1(agent_tracks, agent_metrics, agent_vocab):
    ok = True
    if not os.path.exists('/app/results/allosteric_tracks.tsv'):
        ok = False
    if not os.path.exists('/app/results/validation_metrics.json'):
        ok = False
    if not os.path.exists('/app/results/switch_vocabulary.tsv'):
        ok = False
    if not ok:
        return False
    expected_cols = ['protein', 'residue_index', 'residue_name', 'net_te', 'source', 'sink', 'switch']
    if list(agent_tracks.columns) != expected_cols:
        return False
    expected_vocab_cols = ['word', 'std_net_te', 'count', 'hydrophobic_fraction']
    if list(agent_vocab.columns) != expected_vocab_cols:
        return False
    return True

def check_constraint_2(gold_tracks, agent_tracks):
    gold_counts = gold_tracks.groupby('protein').size().to_dict()
    agent_counts = agent_tracks.groupby('protein').size().to_dict()
    if gold_counts != agent_counts:
        return False
    for protein in gold_counts:
        gold_res = set(gold_tracks[gold_tracks['protein']==protein]['residue_index'])
        agent_res = set(agent_tracks[agent_tracks['protein']==protein]['residue_index'])
        if gold_res != agent_res:
            return False
    return True

def check_constraint_3(gold_tracks, agent_tracks):
    merged = pd.merge(gold_tracks, agent_tracks, on=['protein','residue_index'], suffixes=('_gold','_agent'))
    if len(merged) != len(gold_tracks):
        return False
    for col in ['net_te', 'source', 'sink']:
        diff = np.abs(merged[f'{col}_gold'] - merged[f'{col}_agent'])
        if np.any(diff > 1e-6):
            return False
    return True

def check_constraint_4(gold_tracks, agent_tracks):
    merged = pd.merge(gold_tracks, agent_tracks, on=['protein','residue_index'], suffixes=('_gold','_agent'))
    diff = np.abs(merged['switch_gold'] - merged['switch_agent'])
    if np.any(diff > 1e-6):
        return False
    return True

def check_constraint_5(gold_metrics, agent_metrics):
    for key in ['source_rmsd_mean_rho', 'source_rmsd_positive_count', 'sink_rmsd_mean_rho', 'sink_rmsd_negative_count']:
        if key not in agent_metrics:
            return False
    if abs(gold_metrics['source_rmsd_mean_rho'] - agent_metrics['source_rmsd_mean_rho']) > 0.01:
        return False
    if gold_metrics['source_rmsd_positive_count'] != agent_metrics['source_rmsd_positive_count']:
        return False
    if abs(gold_metrics['sink_rmsd_mean_rho'] - agent_metrics['sink_rmsd_mean_rho']) > 0.01:
        return False
    if gold_metrics['sink_rmsd_negative_count'] != agent_metrics['sink_rmsd_negative_count']:
        return False
    return True

def check_constraint_6(agent_tracks, target_dir):
    labels = compute_hidden_labels(target_dir)
    y_true = []
    y_score = []
    for protein, lab_dict in labels.items():
        df = agent_tracks[agent_tracks['protein'] == protein].sort_values('residue_index')
        if len(df) == 0:
            return False
        for _, row in df.iterrows():
            y_true.append(lab_dict[row['residue_index']])
            y_score.append(row['sink'])
    auc = roc_auc_score(y_true, y_score)
    return auc >= 0.65

def check_constraint_7(gold_vocab, agent_vocab):
    if len(agent_vocab) != 20:
        return False
    merged = pd.merge(gold_vocab, agent_vocab, on='word', suffixes=('_gold','_agent'))
    if len(merged) != 20:
        return False
    for col in ['std_net_te', 'count', 'hydrophobic_fraction']:
        if col == 'count':
            if not np.array_equal(merged[f'{col}_gold'], merged[f'{col}_agent']):
                return False
        else:
            diff = np.abs(merged[f'{col}_gold'] - merged[f'{col}_agent'])
            if np.any(diff > 1e-6):
                return False
    mean_hyd = agent_vocab['hydrophobic_fraction'].mean()
    return mean_hyd >= 0.4

def main():
    agent_tracks = load_agent_tracks('/app/results/allosteric_tracks.tsv')
    with open('/app/results/validation_metrics.json') as f:
        agent_metrics = json.load(f)
    agent_vocab = pd.read_csv('/app/results/switch_vocabulary.tsv', sep='\t')

    background_dir = '/app/data/background'
    target_dir = '/app/data/targets'
    alphabet = build_alphabet(background_dir)
    gold_tracks = project_targets(target_dir, alphabet)
    gold_metrics = compute_metrics(gold_tracks, target_dir)
    gold_vocab = compute_switch_vocab(alphabet)

    results = {}
    results['c1_outputs_exist_and_columns'] = check_constraint_1(agent_tracks, agent_metrics, agent_vocab)
    results['c2_row_counts_and_indices'] = check_constraint_2(gold_tracks, agent_tracks)
    results['c3_net_source_sink_match'] = check_constraint_3(gold_tracks, agent_tracks)
    results['c4_switch_match'] = check_constraint_4(gold_tracks, agent_tracks)
    results['c5_validation_metrics_match'] = check_constraint_5(gold_metrics, agent_metrics)
    results['c6_sink_auc'] = check_constraint_6(agent_tracks, target_dir)
    results['c7_switch_vocabulary'] = check_constraint_7(gold_vocab, agent_vocab)

    passed = sum(results.values())
    total = len(results)
    score = passed / total
    reward = score
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': score, 'reward': reward}, f)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()