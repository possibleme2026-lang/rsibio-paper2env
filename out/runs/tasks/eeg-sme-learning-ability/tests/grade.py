#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

DATA_PATH = Path('/app/data/encoding_events.csv')
OUTPUT_PATH = Path('/app/output/results.json')
REWARD_PATH = Path('/logs/verifier/reward.json')
REPORT_PATH = Path('/logs/verifier/report.json')
REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)


def compute_reference():
    df = pd.read_csv(DATA_PATH)
    df['serial_position_sq'] = df['serial_position'] ** 2
    formula = 'recalled ~ theta_power + alpha_power + gamma_power + serial_position + serial_position_sq'
    model = smf.logit(formula, data=df)
    res = model.fit(disp=False, cov_type='cluster', cov_kwds={'groups': df['subject_id']})
    group_level = {}
    for var in ['theta_power', 'alpha_power', 'gamma_power', 'serial_position', 'serial_position_sq']:
        group_level[var] = {
            'coef': float(res.params[var]),
            'se': float(res.bse[var]),
            'z': float(res.tvalues[var]),
            'p': float(res.pvalues[var])
        }
    per_subject = []
    for subj in sorted(df['subject_id'].unique()):
        sub_df = df[df['subject_id'] == subj].copy()
        if sub_df['recalled'].nunique() < 2:
            coefs = {'theta': 0.0, 'alpha': 0.0, 'gamma': 0.0}
        else:
            try:
                m = smf.logit(formula, data=sub_df)
                r = m.fit(disp=False)
                coefs = {
                    'theta': float(r.params['theta_power']),
                    'alpha': float(r.params['alpha_power']),
                    'gamma': float(r.params['gamma_power'])
                }
            except Exception:
                coefs = {'theta': 0.0, 'alpha': 0.0, 'gamma': 0.0}
        per_subject.append({
            'subject_id': int(subj),
            'theta': coefs['theta'],
            'alpha': coefs['alpha'],
            'gamma': coefs['gamma']
        })
    acc = df.groupby('subject_id')['recalled'].mean().to_dict()
    theta_sme = np.array([abs(p['theta']) for p in per_subject])
    alpha_sme = np.array([abs(p['alpha']) for p in per_subject])
    gamma_sme = np.array([abs(p['gamma']) for p in per_subject])
    accs = np.array([acc[p['subject_id']] for p in per_subject])
    r_theta, p_theta = pearsonr(accs, theta_sme)
    r_alpha, p_alpha = pearsonr(accs, alpha_sme)
    r_gamma, p_gamma = pearsonr(accs, gamma_sme)
    correlations = {
        'theta': {'r': float(r_theta), 'p': float(p_theta)},
        'alpha': {'r': float(r_alpha), 'p': float(p_alpha)},
        'gamma': {'r': float(r_gamma), 'p': float(p_gamma)}
    }
    return {
        'group_level': group_level,
        'per_subject': per_subject,
        'correlations': correlations
    }


def load_agent_output():
    if not OUTPUT_PATH.exists():
        return None
    try:
        with open(OUTPUT_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def check(cond, msg):
    if cond:
        print(f'PASS: {msg}')
        return 1
    else:
        print(f'FAIL: {msg}')
        return 0


def main():
    ref = compute_reference()
    agent = load_agent_output()
    checks_passed = 0
    total_checks = 10

    checks_passed += check(agent is not None, 'Output file exists and is valid JSON')
    if agent is None:
        score = 0.0
        with open(REWARD_PATH, 'w') as f:
            json.dump({'score': score, 'reward': score}, f)
        with open(REPORT_PATH, 'w') as f:
            json.dump({'checks_passed': checks_passed, 'total_checks': total_checks}, f)
        return

    req_vars = ['theta_power', 'alpha_power', 'gamma_power', 'serial_position', 'serial_position_sq']
    has_all = all(v in agent.get('group_level', {}) for v in req_vars)
    for v in req_vars:
        if v in agent.get('group_level', {}):
            sub = agent['group_level'][v]
            has_all = has_all and all(k in sub for k in ['coef', 'se', 'z', 'p'])
    checks_passed += check(has_all, 'Group-level section has all required predictors with coef, se, z, p')

    theta_ok = False
    if has_all:
        theta = agent['group_level']['theta_power']
        theta_ok = theta['coef'] > 0 and theta['p'] < 0.05
    checks_passed += check(theta_ok, 'Theta coefficient > 0 and p < 0.05')

    alpha_ok = False
    if has_all:
        alpha = agent['group_level']['alpha_power']
        alpha_ok = alpha['coef'] < 0 and alpha['p'] < 0.05
    checks_passed += check(alpha_ok, 'Alpha coefficient < 0 and p < 0.05')

    gamma_ok = False
    if has_all:
        gamma = agent['group_level']['gamma_power']
        gamma_ok = gamma['p'] > 0.05
    checks_passed += check(gamma_ok, 'Gamma p > 0.05')

    match_group = False
    if has_all:
        match_group = True
        for v in req_vars:
            ref_coef = ref['group_level'][v]['coef']
            agent_coef = agent['group_level'][v]['coef']
            if abs(ref_coef - agent_coef) > 0.05:
                match_group = False
                break
    checks_passed += check(match_group, 'Group-level coefficients match reference within ±0.05')

    per_subj_ok = False
    if 'per_subject' in agent and isinstance(agent['per_subject'], list):
        per_subj_ok = len(agent['per_subject']) == 80
        if per_subj_ok:
            for entry in agent['per_subject']:
                if not all(k in entry for k in ['subject_id', 'theta', 'alpha', 'gamma']):
                    per_subj_ok = False
                    break
    checks_passed += check(per_subj_ok, 'Per-subject section contains exactly 80 subjects with theta, alpha, gamma coefficients')

    match_subj = False
    if per_subj_ok:
        agent_dict = {e['subject_id']: e for e in agent['per_subject']}
        ref_dict = {e['subject_id']: e for e in ref['per_subject']}
        np.random.seed(123)
        sample_ids = np.random.choice(list(agent_dict.keys()), size=5, replace=False)
        match_subj = True
        for sid in sample_ids:
            for band in ['theta', 'alpha', 'gamma']:
                ref_val = ref_dict[sid][band]
                agent_val = agent_dict[sid][band]
                if abs(ref_val - agent_val) > 0.1:
                    match_subj = False
                    break
            if not match_subj:
                break
    checks_passed += check(match_subj, 'Per-subject coefficients for 5 random subjects match reference within ±0.1')

    corr_ok = False
    if 'correlations' in agent:
        c = agent['correlations']
        if 'theta' in c and 'alpha' in c:
            corr_ok = c['theta']['r'] < 0 and c['alpha']['r'] < 0
    checks_passed += check(corr_ok, 'Correlation theta r < 0 and alpha r < 0')

    match_corr = False
    if 'correlations' in agent:
        c = agent['correlations']
        match_corr = True
        for band in ['theta', 'alpha', 'gamma']:
            if band not in c:
                match_corr = False
                break
            ref_r = ref['correlations'][band]['r']
            agent_r = c[band]['r']
            if abs(ref_r - agent_r) > 0.1:
                match_corr = False
                break
    checks_passed += check(match_corr, 'Correlation values match reference within ±0.1')

    score = checks_passed / total_checks
    with open(REWARD_PATH, 'w') as f:
        json.dump({'score': score, 'reward': score}, f)
    with open(REPORT_PATH, 'w') as f:
        json.dump({'checks_passed': checks_passed, 'total_checks': total_checks, 'score': score}, f)
    print(f'Score: {checks_passed}/{total_checks} = {score:.4f}')


if __name__ == '__main__':
    main()
