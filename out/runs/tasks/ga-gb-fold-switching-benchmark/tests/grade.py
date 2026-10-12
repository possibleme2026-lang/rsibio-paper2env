#!/usr/bin/env python3
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, '/tests')
import gold_gen

DATA_PATH = '/app/data/benchmark_data.csv'
PRED_PATH = '/app/output/benchmark_report.json'
REWARD_PATH = '/logs/verifier/reward.json'

METHODS = ['alphafold2', 'esmfold', 'rosetta', 'foldx', 'evfold', 'sparks']
STATES = ['p_3', 'p_4b_plus', 'p_mixed', 'p_unfolded']
POSITIONS = ['26', '33', '45']

def compute_metrics(df, method, position=None):
    if position is not None:
        df = df[df['position'] == int(position)]
    pred_cols = [f'{method}_{s}' for s in STATES]
    sub = df.dropna(subset=pred_cols)
    if len(sub) == 0:
        return None
    rmse = {}
    spearman = {}
    for s in STATES:
        meas = sub[f'measured_{s}'].values
        pred = sub[f'{method}_{s}'].values
        rmse[s] = float(np.sqrt(np.mean((pred - meas)**2)))
        if len(np.unique(meas)) < 2 or len(np.unique(pred)) < 2:
            spearman[s] = float('nan')
        else:
            rho, _ = spearmanr(meas, pred)
            spearman[s] = float(rho)
    return {'rmse': rmse, 'spearman': spearman}

def compute_expected(df):
    overall = {}
    for method in METHODS:
        overall[method] = compute_metrics(df, method)
    per_pos = {}
    for pos in POSITIONS:
        per_pos[pos] = {}
        for method in METHODS:
            per_pos[pos][method] = compute_metrics(df, method, pos)
    best = {}
    for pos in POSITIONS:
        avg_rmse = {}
        avg_spearman = {}
        for method in METHODS:
            m = per_pos[pos][method]
            if m is None:
                continue
            avg_rmse[method] = np.mean(list(m['rmse'].values()))
            avg_spearman[method] = np.nanmean(list(m['spearman'].values()))
        best_rmse = sorted(avg_rmse.items(), key=lambda x: (x[1], x[0]))[0][0]
        best_spearman = sorted(avg_spearman.items(), key=lambda x: (-x[1], x[0]))[0][0]
        best[pos] = {'best_rmse_method': best_rmse, 'best_spearman_method': best_spearman}
    return overall, per_pos, best

def check_structure(data):
    if not isinstance(data, dict):
        return False, 'not a dict'
    required = ['overall_metrics', 'per_position_metrics', 'best_methods']
    for k in required:
        if k not in data:
            return False, f'missing top-level key {k}'
    om = data['overall_metrics']
    if not isinstance(om, dict) or set(om.keys()) != set(METHODS):
        return False, 'overall_metrics keys mismatch'
    for m in METHODS:
        if not isinstance(om[m], dict):
            return False, f'overall_metrics[{m}] not dict'
        if set(om[m].keys()) != {'rmse', 'spearman'}:
            return False, f'overall_metrics[{m}] keys mismatch'
        for metric in ['rmse', 'spearman']:
            if set(om[m][metric].keys()) != set(STATES):
                return False, f'overall_metrics[{m}][{metric}] keys mismatch'
            for s in STATES:
                if not isinstance(om[m][metric][s], (int, float)):
                    return False, f'overall_metrics[{m}][{metric}][{s}] not a number'
    ppm = data['per_position_metrics']
    if not isinstance(ppm, dict) or set(ppm.keys()) != set(POSITIONS):
        return False, 'per_position_metrics keys mismatch'
    for pos in POSITIONS:
        if not isinstance(ppm[pos], dict) or set(ppm[pos].keys()) != set(METHODS):
            return False, f'per_position_metrics[{pos}] keys mismatch'
        for m in METHODS:
            if not isinstance(ppm[pos][m], dict):
                return False, f'per_position_metrics[{pos}][{m}] not dict'
            if set(ppm[pos][m].keys()) != {'rmse', 'spearman'}:
                return False, f'per_position_metrics[{pos}][{m}] keys mismatch'
            for metric in ['rmse', 'spearman']:
                if set(ppm[pos][m][metric].keys()) != set(STATES):
                    return False, f'per_position_metrics[{pos}][{m}][{metric}] keys mismatch'
                for s in STATES:
                    if not isinstance(ppm[pos][m][metric][s], (int, float)):
                        return False, f'per_position_metrics[{pos}][{m}][{metric}][{s}] not a number'
    bm = data['best_methods']
    if not isinstance(bm, dict) or set(bm.keys()) != set(POSITIONS):
        return False, 'best_methods keys mismatch'
    for pos in POSITIONS:
        if not isinstance(bm[pos], dict):
            return False, f'best_methods[{pos}] not dict'
        if set(bm[pos].keys()) != {'best_rmse_method', 'best_spearman_method'}:
            return False, f'best_methods[{pos}] keys mismatch'
        for k in ['best_rmse_method', 'best_spearman_method']:
            if bm[pos][k] not in METHODS:
                return False, f'best_methods[{pos}][{k}] invalid method'
    return True, ''

def check_rmse_ranges(data):
    om = data['overall_metrics']
    for m in METHODS:
        for s in STATES:
            v = om[m]['rmse'][s]
            if not (0 <= v <= 1):
                return False
    ppm = data['per_position_metrics']
    for pos in POSITIONS:
        for m in METHODS:
            for s in STATES:
                v = ppm[pos][m]['rmse'][s]
                if not (0 <= v <= 1):
                    return False
    return True

def check_spearman_ranges(data):
    om = data['overall_metrics']
    for m in METHODS:
        for s in STATES:
            v = om[m]['spearman'][s]
            if not (-1 <= v <= 1):
                return False
    ppm = data['per_position_metrics']
    for pos in POSITIONS:
        for m in METHODS:
            for s in STATES:
                v = ppm[pos][m]['spearman'][s]
                if not (-1 <= v <= 1):
                    return False
    return True

def main():
    score = 0.0
    total = 16.0
    results = {}
    try:
        df = gold_gen.generate_data()
    except Exception as e:
        results['gold_gen_error'] = str(e)
        reward = {'score': 0.0, 'reward': 0.0}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open(REWARD_PATH, 'w') as f:
            json.dump(reward, f)
        return
    expected_overall, expected_per_pos, expected_best = compute_expected(df)
    if not os.path.exists(PRED_PATH):
        agent_data = None
    else:
        try:
            with open(PRED_PATH) as f:
                agent_data = json.load(f)
        except Exception:
            agent_data = None
    if agent_data is not None:
        results['C1_valid_json'] = True
        score += 1
    else:
        results['C1_valid_json'] = False
        reward = {'score': score/total, 'reward': score/total}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open(REWARD_PATH, 'w') as f:
            json.dump(reward, f)
        return
    if isinstance(agent_data, dict) and set(agent_data.keys()) == {'overall_metrics', 'per_position_metrics', 'best_methods'}:
        results['C2_toplevel_keys'] = True
        score += 1
    else:
        results['C2_toplevel_keys'] = False
    struct_ok, struct_msg = check_structure(agent_data)
    if struct_ok:
        results['C3_overall_structure'] = True
        score += 1
    else:
        results['C3_overall_structure'] = False
        results['C3_msg'] = struct_msg
    if struct_ok:
        results['C4_perpos_structure'] = True
        score += 1
    else:
        results['C4_perpos_structure'] = False
    if struct_ok:
        results['C5_best_methods_structure'] = True
        score += 1
    else:
        results['C5_best_methods_structure'] = False
    if struct_ok:
        if check_rmse_ranges(agent_data):
            results['C6_rmse_ranges'] = True
            score += 1
        else:
            results['C6_rmse_ranges'] = False
    else:
        results['C6_rmse_ranges'] = False
    if struct_ok:
        if check_spearman_ranges(agent_data):
            results['C7_spearman_ranges'] = True
            score += 1
        else:
            results['C7_spearman_ranges'] = False
    else:
        results['C7_spearman_ranges'] = False
    if struct_ok:
        exp = expected_overall['alphafold2']['rmse']['p_3']
        got = agent_data['overall_metrics']['alphafold2']['rmse']['p_3']
        if abs(got - exp) <= 0.01:
            results['C8_af2_p3_rmse'] = True
            score += 1
        else:
            results['C8_af2_p3_rmse'] = False
            results['C8_msg'] = f'expected {exp:.4f}, got {got:.4f}'
    else:
        results['C8_af2_p3_rmse'] = False
        results['C8_msg'] = 'structure invalid'
    if struct_ok:
        exp = expected_overall['esmfold']['spearman']['p_4b_plus']
        got = agent_data['overall_metrics']['esmfold']['spearman']['p_4b_plus']
        if abs(got - exp) <= 0.02:
            results['C9_esmfold_p4b_spearman'] = True
            score += 1
        else:
            results['C9_esmfold_p4b_spearman'] = False
            results['C9_msg'] = f'expected {exp:.4f}, got {got:.4f}'
    else:
        results['C9_esmfold_p4b_spearman'] = False
        results['C9_msg'] = 'structure invalid'
    if struct_ok:
        exp = expected_best['33']['best_rmse_method']
        got = agent_data['best_methods']['33']['best_rmse_method']
        if got == exp:
            results['C10_best_rmse_pos33'] = True
            score += 1
        else:
            results['C10_best_rmse_pos33'] = False
            results['C10_msg'] = f'expected {exp}, got {got}'
    else:
        results['C10_best_rmse_pos33'] = False
        results['C10_msg'] = 'structure invalid'
    if struct_ok:
        exp = expected_best['45']['best_spearman_method']
        got = agent_data['best_methods']['45']['best_spearman_method']
        if got == exp:
            results['C11_best_spearman_pos45'] = True
            score += 1
        else:
            results['C11_best_spearman_pos45'] = False
            results['C11_msg'] = f'expected {exp}, got {got}'
    else:
        results['C11_best_spearman_pos45'] = False
        results['C11_msg'] = 'structure invalid'
    if struct_ok:
        mae = 0.0
        count = 0
        for m in METHODS:
            for s in STATES:
                exp = expected_overall[m]['rmse'][s]
                got = agent_data['overall_metrics'][m]['rmse'][s]
                mae += abs(exp - got)
                count += 1
        mae /= count
        if mae < 0.015:
            results['C12_overall_rmse_mae'] = True
            score += 1
        else:
            results['C12_overall_rmse_mae'] = False
            results['C12_msg'] = f'MAE {mae:.4f}'
    else:
        results['C12_overall_rmse_mae'] = False
        results['C12_msg'] = 'structure invalid'
    if struct_ok:
        mae = 0.0
        count = 0
        for m in METHODS:
            for s in STATES:
                exp = expected_overall[m]['spearman'][s]
                got = agent_data['overall_metrics'][m]['spearman'][s]
                if not (np.isnan(exp) or np.isnan(got)):
                    mae += abs(exp - got)
                    count += 1
        if count > 0:
            mae /= count
        if mae < 0.02:
            results['C13_overall_spearman_mae'] = True
            score += 1
        else:
            results['C13_overall_spearman_mae'] = False
            results['C13_msg'] = f'MAE {mae:.4f}'
    else:
        results['C13_overall_spearman_mae'] = False
        results['C13_msg'] = 'structure invalid'
    if struct_ok:
        mae = 0.0
        count = 0
        for pos in POSITIONS:
            for m in METHODS:
                for s in STATES:
                    exp = expected_per_pos[pos][m]['rmse'][s]
                    got = agent_data['per_position_metrics'][pos][m]['rmse'][s]
                    mae += abs(exp - got)
                    count += 1
        mae /= count
        if mae < 0.02:
            results['C14_perpos_rmse_mae'] = True
            score += 1
        else:
            results['C14_perpos_rmse_mae'] = False
            results['C14_msg'] = f'MAE {mae:.4f}'
    else:
        results['C14_perpos_rmse_mae'] = False
        results['C14_msg'] = 'structure invalid'
    if struct_ok:
        mae = 0.0
        count = 0
        for pos in POSITIONS:
            for m in METHODS:
                for s in STATES:
                    exp = expected_per_pos[pos][m]['spearman'][s]
                    got = agent_data['per_position_metrics'][pos][m]['spearman'][s]
                    if not (np.isnan(exp) or np.isnan(got)):
                        mae += abs(exp - got)
                        count += 1
        if count > 0:
            mae /= count
        if mae < 0.025:
            results['C15_perpos_spearman_mae'] = True
            score += 1
        else:
            results['C15_perpos_spearman_mae'] = False
            results['C15_msg'] = f'MAE {mae:.4f}'
    else:
        results['C15_perpos_spearman_mae'] = False
        results['C15_msg'] = 'structure invalid'
    if struct_ok:
        all_ok = True
        for pos in POSITIONS:
            if agent_data['best_methods'][pos]['best_rmse_method'] != expected_best[pos]['best_rmse_method']:
                all_ok = False
            if agent_data['best_methods'][pos]['best_spearman_method'] != expected_best[pos]['best_spearman_method']:
                all_ok = False
        if all_ok:
            results['C16_all_best_methods'] = True
            score += 1
        else:
            results['C16_all_best_methods'] = False
            results['C16_msg'] = 'some best methods incorrect'
    else:
        results['C16_all_best_methods'] = False
        results['C16_msg'] = 'structure invalid'
    final_score = score / total
    reward = {'score': final_score, 'reward': final_score}
    os.makedirs('/logs/verifier', exist_ok=True)
    with open(REWARD_PATH, 'w') as f:
        json.dump(reward, f)
    with open('/logs/verifier/details.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
