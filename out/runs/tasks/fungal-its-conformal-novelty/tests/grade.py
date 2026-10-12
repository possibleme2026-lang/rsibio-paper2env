#!/usr/bin/env python3
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/tests')
import gold_gen

OUT_DIR = '/app/output'
REPORT_PATH = '/logs/verifier/report.json'
REWARD_PATH = '/logs/verifier/reward.json'

def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None

def main():
    ref_df, cal_df, val_df, query_df, query_labels, soil_df = gold_gen.generate_all(seed=20260729)
    alpha = 0.05
    
    cal_scores = {}
    thresholds = {}
    for (region, method), group in cal_df.groupby(['region', 'method']):
        scores = np.sort(group['score'].values)
        n = len(scores)
        r = int(np.floor(alpha * (n + 1)))
        t = scores[r - 1] if r > 0 else scores[0] - 1e-6
        thresholds[f'{region}|{method}'] = t
        cal_scores[(region, method)] = scores
    
    query_df = query_df.copy()
    query_df['p_value_gold'] = 0.0
    query_df['novel_flag_gold'] = 0
    for (region, method), group in query_df.groupby(['region', 'method']):
        scores = cal_scores[(region, method)]
        n = len(scores)
        for idx, row in group.iterrows():
            s = row['score']
            C = np.sum(scores <= s)
            p = (1 + C) / (n + 1)
            query_df.at[idx, 'p_value_gold'] = p
            query_df.at[idx, 'novel_flag_gold'] = 1 if p <= alpha else 0
    query_df['true_genus'] = query_df['query_id'].map(query_labels)
    query_df['is_novel'] = query_df['true_genus'].str.startswith('N')
    
    val_df = val_df.copy()
    val_df['novel_flag_gold'] = 0
    for (region, method), group in val_df.groupby(['region', 'method']):
        scores = cal_scores[(region, method)]
        for idx, row in group.iterrows():
            s = row['score']
            C = np.sum(scores <= s)
            p = (1 + C) / (len(scores) + 1)
            val_df.at[idx, 'novel_flag_gold'] = 1 if p <= alpha else 0
    
    soil_df = soil_df.copy()
    soil_df['novel_flag_gold'] = 0
    scores = cal_scores[('ITS1', 'alignment')]
    for idx, row in soil_df.iterrows():
        s = row['score']
        C = np.sum(scores <= s)
        p = (1 + C) / (len(scores) + 1)
        soil_df.at[idx, 'novel_flag_gold'] = 1 if p <= alpha else 0
    gold_soil_frac_all = soil_df['novel_flag_gold'].mean()
    abundant = soil_df['abundance'] >= 10
    gold_soil_frac_abund = soil_df.loc[abundant, 'novel_flag_gold'].mean()
    
    agent_flags = safe_read_csv(f'{OUT_DIR}/novelty_flags.tsv', sep='\t')
    agent_thresholds = None
    try:
        with open(f'{OUT_DIR}/thresholds.json') as f:
            agent_thresholds = json.load(f)
    except Exception:
        pass
    agent_coverage = None
    try:
        with open(f'{OUT_DIR}/coverage_report.json') as f:
            agent_coverage = json.load(f)
    except Exception:
        pass
    agent_soil_summary = None
    try:
        with open(f'{OUT_DIR}/soil_summary.json') as f:
            agent_soil_summary = json.load(f)
    except Exception:
        pass
    
    c1_checks = []
    if agent_flags is not None:
        c1_checks.append(agent_flags.shape[0] == len(query_df))
        required_cols = ['query_id', 'region', 'method', 'p_value', 'novel_flag']
        c1_checks.append(set(required_cols).issubset(agent_flags.columns))
        if set(required_cols).issubset(agent_flags.columns):
            c1_checks.append(agent_flags['p_value'].between(0, 1).all())
            c1_checks.append(agent_flags['novel_flag'].isin([0, 1]).all())
            sorted_df = agent_flags.sort_values(['query_id', 'region', 'method'])
            c1_checks.append(agent_flags[['query_id', 'region', 'method']].equals(sorted_df[['query_id', 'region', 'method']]))
            c1_checks.append(not agent_flags.duplicated(subset=['query_id', 'region', 'method']).any())
    else:
        c1_checks.append(False)
    c1_score = sum(c1_checks) / len(c1_checks) if c1_checks else 0.0
    
    c2_score = 0.0
    if agent_flags is not None and 'p_value' in agent_flags.columns:
        merged = query_df[['query_id', 'region', 'method', 'p_value_gold']].merge(
            agent_flags[['query_id', 'region', 'method', 'p_value']],
            on=['query_id', 'region', 'method'], how='left')
        if len(merged) == len(query_df):
            diff = np.abs(merged['p_value'] - merged['p_value_gold'])
            c2_score = (diff <= 1e-6).mean()
    
    c3_score = 0.0
    if agent_thresholds is not None:
        correct = 0
        total = len(thresholds)
        for key, gold_t in thresholds.items():
            agent_t = agent_thresholds.get(key)
            if agent_t is not None and abs(agent_t - gold_t) <= 1e-6:
                correct += 1
        c3_score = correct / total
    
    c4_score = 0.0
    if agent_coverage is not None:
        correct = 0
        total = 8
        for (region, method), group in val_df.groupby(['region', 'method']):
            key = f'{region}|{method}'
            gold_rate = group['novel_flag_gold'].mean()
            agent_rate = agent_coverage.get(key)
            if agent_rate is not None and gold_rate <= 0.07 and abs(agent_rate - gold_rate) <= 0.01:
                correct += 1
        c4_score = correct / total
    
    c5_score = 0.0
    if agent_flags is not None:
        merged = query_df[['query_id', 'region', 'method', 'is_novel']].merge(
            agent_flags[['query_id', 'region', 'method', 'novel_flag']],
            on=['query_id', 'region', 'method'], how='left')
        merged['novel_flag'] = merged['novel_flag'].fillna(0)
        known = merged[merged['is_novel'] == False]
        correct = 0
        total = 8
        for (region, method), group in known.groupby(['region', 'method']):
            fdr = group['novel_flag'].mean()
            if fdr <= 0.08:
                correct += 1
        c5_score = correct / total
    
    c6_score = 0.0
    if agent_flags is not None:
        merged = query_df[['query_id', 'region', 'method', 'is_novel']].merge(
            agent_flags[['query_id', 'region', 'method', 'novel_flag']],
            on=['query_id', 'region', 'method'], how='left')
        merged['novel_flag'] = merged['novel_flag'].fillna(0)
        novel = merged[merged['is_novel'] == True]
        checks = []
        for (region, method), group in novel.groupby(['region', 'method']):
            det = group['novel_flag'].mean()
            checks.append(det >= 0.10)
        overall_det = novel['novel_flag'].mean()
        checks.append(overall_det >= 0.15)
        c6_score = sum(checks) / len(checks) if checks else 0.0
    
    c7_score = 0.0
    if agent_flags is not None:
        merged = query_df[['query_id', 'region', 'method', 'is_novel']].merge(
            agent_flags[['query_id', 'region', 'method', 'novel_flag']],
            on=['query_id', 'region', 'method'], how='left')
        merged['novel_flag'] = merged['novel_flag'].fillna(0)
        flagged = merged[merged['novel_flag'] == 1]
        if len(flagged) > 0:
            precision = flagged['is_novel'].mean()
            c7_score = 1.0 if precision >= 0.50 else 0.0
        else:
            c7_score = 0.0
    
    c8_score = 0.0
    if agent_soil_summary is not None:
        c8_score = 1.0
        if abs(agent_soil_summary.get('fraction_novel_all', -1) - gold_soil_frac_all) > 0.005:
            c8_score = 0.0
        if abs(agent_soil_summary.get('fraction_novel_abundant', -1) - gold_soil_frac_abund) > 0.005:
            c8_score = 0.0
    
    scores = {
        'C1_format': c1_score,
        'C2_pvalue': c2_score,
        'C3_threshold': c3_score,
        'C4_validation_coverage': c4_score,
        'C5_known_fdr': c5_score,
        'C6_detection': c6_score,
        'C7_precision': c7_score,
        'C8_soil': c8_score,
    }
    total_score = float(np.mean(list(scores.values())))
    
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        json.dump(scores, f, indent=2)
    with open(REWARD_PATH, 'w') as f:
        json.dump({'score': total_score, 'reward': total_score}, f)
    
    print(json.dumps(scores, indent=2))
    print(f'Total score: {total_score}')

if __name__ == '__main__':
    main()
