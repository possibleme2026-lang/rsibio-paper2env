#!/usr/bin/env python3
import os, sys, json, math
import numpy as np
from scipy.stats import pearsonr

sys.path.append('/tests')
from gold_gen import run_simulation

def main():
    _, gold = run_simulation(seed=42)
    true_params = gold['true_params']
    true_probs = np.array(gold['true_2020_probs'])
    true_mean = gold['true_mean_size_2020']
    bin_mids = np.array(gold['bin_mids'])
    
    sub_dir = '/app/submission'
    pred_path = os.path.join(sub_dir, 'predictions.csv')
    params_path = os.path.join(sub_dir, 'params.json')
    
    constraints = {}
    details = {}
    
    c1_pass = False
    agent_probs = None
    try:
        import csv
        with open(pred_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if len(rows) != 20:
            details['c1'] = f'Expected 20 rows, got {len(rows)}'
        else:
            required_cols = {'bin_lower', 'bin_upper', 'predicted_count', 'predicted_prob'}
            if not required_cols.issubset(rows[0].keys()):
                details['c1'] = f'Missing columns. Got {list(rows[0].keys())}'
            else:
                bin_lowers = [float(r['bin_lower']) for r in rows]
                if bin_lowers != sorted(bin_lowers):
                    details['c1'] = 'Rows not sorted by bin_lower'
                else:
                    counts = [float(r['predicted_count']) for r in rows]
                    probs = [float(r['predicted_prob']) for r in rows]
                    if any(c < 0 for c in counts):
                        details['c1'] = 'Negative predicted_count'
                    elif any(p < 0 for p in probs):
                        details['c1'] = 'Negative predicted_prob'
                    elif abs(sum(probs) - 1.0) > 0.01:
                        details['c1'] = f'predicted_prob sum to {sum(probs)}'
                    else:
                        c1_pass = True
                        agent_probs = np.array(probs)
    except Exception as e:
        details['c1'] = f'Error reading predictions: {e}'
    constraints['c1_format'] = c1_pass
    
    c2_pass = False
    agent_params = None
    try:
        with open(params_path, 'r') as f:
            agent_params = json.load(f)
        required_keys = {'growth_vpd', 'growth_intercept', 'survival_vpd', 'survival_intercept', 'recruitment'}
        missing = required_keys - set(agent_params.keys())
        if missing:
            details['c2'] = f'Missing keys: {missing}'
        else:
            for k in required_keys:
                v = agent_params[k]
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    details['c2'] = f'{k} is not a finite number: {v}'
                    break
            else:
                c2_pass = True
    except Exception as e:
        details['c2'] = f'Error reading params: {e}'
    constraints['c2_params_format'] = c2_pass
    
    c3_pass = False
    if c2_pass:
        gv = agent_params['growth_vpd']
        true_gv = true_params['g_vpd']
        if gv < 0 and abs(gv - true_gv) <= 0.08:
            c3_pass = True
            details['c3'] = f'growth_vpd={gv:.4f} (true={true_gv})'
        else:
            details['c3'] = f'growth_vpd={gv:.4f} not negative or not within 0.08 of {true_gv}'
    else:
        details['c3'] = 'params.json invalid'
    constraints['c3_growth_vpd'] = c3_pass
    
    c4_pass = False
    rmse = None
    if c1_pass:
        rmse = math.sqrt(np.mean((agent_probs - true_probs)**2))
        if rmse <= 0.02:
            c4_pass = True
            details['c4'] = f'RMSE={rmse:.6f}'
        else:
            details['c4'] = f'RMSE={rmse:.6f} > 0.02'
    else:
        details['c4'] = 'predictions invalid'
    constraints['c4_rmse'] = c4_pass
    
    c5_pass = False
    mean_size = None
    if c1_pass:
        mean_size = np.sum(agent_probs * bin_mids)
        if abs(mean_size - true_mean) <= 3.0:
            c5_pass = True
            details['c5'] = f'mean_size={mean_size:.2f} (true={true_mean:.2f})'
        else:
            details['c5'] = f'mean_size={mean_size:.2f} not within 3.0 of {true_mean:.2f}'
    else:
        details['c5'] = 'predictions invalid'
    constraints['c5_mean_size'] = c5_pass
    
    c6_pass = False
    corr = None
    if c1_pass:
        corr, _ = pearsonr(agent_probs, true_probs)
        if corr >= 0.85:
            c6_pass = True
            details['c6'] = f'correlation={corr:.4f}'
        else:
            details['c6'] = f'correlation={corr:.4f} < 0.85'
    else:
        details['c6'] = 'predictions invalid'
    constraints['c6_correlation'] = c6_pass
    
    c7_pass = False
    if c2_pass:
        rec = agent_params['recruitment']
        true_rec = true_params['R']
        if true_rec > 0 and abs(rec - true_rec) / true_rec <= 0.20:
            c7_pass = True
            details['c7'] = f'recruitment={rec:.2f} (true={true_rec})'
        else:
            details['c7'] = f'recruitment={rec:.2f} not within 20% of {true_rec}'
    else:
        details['c7'] = 'params.json invalid'
    constraints['c7_recruitment'] = c7_pass
    
    c8_pass = False
    if c2_pass:
        sv = agent_params['survival_vpd']
        true_sv = true_params['s_vpd']
        if sv < 0 and abs(sv - true_sv) <= 0.3:
            c8_pass = True
            details['c8'] = f'survival_vpd={sv:.4f} (true={true_sv})'
        else:
            details['c8'] = f'survival_vpd={sv:.4f} not negative or not within 0.3 of {true_sv}'
    else:
        details['c8'] = 'params.json invalid'
    constraints['c8_survival_vpd'] = c8_pass
    
    weights = {
        'c1_format': 0.10,
        'c2_params_format': 0.10,
        'c3_growth_vpd': 0.20,
        'c4_rmse': 0.20,
        'c5_mean_size': 0.10,
        'c6_correlation': 0.10,
        'c7_recruitment': 0.10,
        'c8_survival_vpd': 0.10
    }
    score = sum(weights[k] for k, v in constraints.items() if v)
    
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': score, 'reward': score}, f)
    
    report = {
        'constraints': constraints,
        'details': details,
        'score': score,
        'true_params': true_params,
        'true_2020_probs': true_probs.tolist(),
        'true_mean_size_2020': true_mean,
        'agent_growth_vpd': agent_params.get('growth_vpd') if agent_params else None,
        'agent_recruitment': agent_params.get('recruitment') if agent_params else None,
        'rmse': rmse,
        'mean_size': mean_size if mean_size is not None else None,
        'correlation': corr if corr is not None else None
    }
    with open('/logs/verifier/report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
