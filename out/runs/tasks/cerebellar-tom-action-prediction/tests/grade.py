import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import os
import sys

def load_data():
    gmv = pd.read_csv('/app/data/cerebellar_gmv.csv')
    beh = pd.read_csv('/app/data/behavior.csv')
    net = pd.read_csv('/app/data/cortical_network_scores.csv')
    df = gmv.merge(beh, on='subject_id').merge(net, on='subject_id')
    regions = [c for c in gmv.columns if c != 'subject_id']
    return df, regions

def fit_region_models(df, regions, behavior):
    results = []
    for reg in regions:
        X = df[[reg, 'age_months', 'sex', 'site', 'tiv_mm3']].copy()
        X = pd.get_dummies(X, columns=['sex', 'site'], drop_first=True)
        X = sm.add_constant(X)
        y = df[behavior]
        model = sm.OLS(y, X).fit()
        beta = model.params[reg]
        t = model.tvalues[reg]
        p = model.pvalues[reg]
        results.append({'region': reg, 'beta': beta, 't': t, 'p': p})
    res_df = pd.DataFrame(results)
    _, q = fdrcorrection(res_df['p'].values, alpha=0.05)
    res_df['q'] = q
    res_df = res_df.sort_values('p')
    return res_df

def cv_r2(df, regions, behavior):
    if len(regions) == 0:
        return 0.0
    X = df[regions].copy()
    X = (X - X.mean()) / X.std()
    y = df[behavior].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s = []
    for train_idx, test_idx in kf.split(X):
        n_comp = min(2, len(regions))
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X.iloc[train_idx], y[train_idx])
        y_pred = pls.predict(X.iloc[test_idx]).flatten()
        r2s.append(r2_score(y[test_idx], y_pred))
    return float(np.mean(r2s))

def network_corr(df, regions, network_col):
    if len(regions) == 0:
        return 0.0, 1.0
    X = df[regions].copy()
    X = (X - X.mean()) / X.std()
    mean_gmv = X.mean(axis=1)
    r, p = pearsonr(mean_gmv, df[network_col])
    return float(r), float(p)

def main():
    try:
        with open('/app/results.json') as f:
            agent = json.load(f)
    except Exception as e:
        print(f"Failed to load results.json: {e}")
        reward = {"score": 0.0, "reward": 0.0}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump(reward, f)
        sys.exit(0)

    df, regions = load_data()
    tom_res = fit_region_models(df, regions, 'explicit_tom_score')
    action_res = fit_region_models(df, regions, 'action_prediction_accuracy')
    tom_sig_ref = tom_res[tom_res['q'] < 0.05]['region'].tolist()
    action_sig_ref = action_res[action_res['q'] < 0.05]['region'].tolist()
    intersection_ref = len(set(tom_sig_ref) & set(action_sig_ref))
    tom_cv_r2_ref = cv_r2(df, tom_sig_ref, 'explicit_tom_score')
    action_cv_r2_ref = cv_r2(df, action_sig_ref, 'action_prediction_accuracy')
    tom_r_ref, tom_p_ref = network_corr(df, tom_sig_ref, 'DefaultMode')
    action_r_ref, action_p_ref = network_corr(df, action_sig_ref, 'VentralAttention')

    checks = []
    def add_check(name, condition, details=""):
        checks.append({'name': name, 'passed': bool(condition), 'details': details})

    required_keys = ['tom_regions', 'action_regions', 'dissociation_intersection',
                     'tom_cv_r2', 'action_cv_r2', 'tom_network_correlation',
                     'action_network_correlation', 'tom_fdr_table', 'action_fdr_table']
    has_keys = all(k in agent for k in required_keys)
    add_check('json_structure', has_keys, f"Missing keys: {[k for k in required_keys if k not in agent]}")

    agent_tom = set(agent.get('tom_regions', []))
    ref_tom = set(tom_sig_ref)
    add_check('tom_regions_match', agent_tom == ref_tom, f"Agent: {agent_tom}, Ref: {ref_tom}")

    agent_action = set(agent.get('action_regions', []))
    ref_action = set(action_sig_ref)
    add_check('action_regions_match', agent_action == ref_action, f"Agent: {agent_action}, Ref: {ref_action}")

    agent_intersect = agent.get('dissociation_intersection', None)
    add_check('dissociation_intersection', agent_intersect == intersection_ref, f"Agent: {agent_intersect}, Ref: {intersection_ref}")

    agent_tom_r2 = agent.get('tom_cv_r2', 0)
    cond = agent_tom_r2 > 0.15 and abs(agent_tom_r2 - tom_cv_r2_ref) < 0.05
    add_check('tom_cv_r2', cond, f"Agent: {agent_tom_r2}, Ref: {tom_cv_r2_ref}")

    agent_action_r2 = agent.get('action_cv_r2', 0)
    cond = agent_action_r2 > 0.15 and abs(agent_action_r2 - action_cv_r2_ref) < 0.05
    add_check('action_cv_r2', cond, f"Agent: {agent_action_r2}, Ref: {action_cv_r2_ref}")

    agent_tom_net = agent.get('tom_network_correlation', {})
    cond = (agent_tom_net.get('network') == 'DefaultMode' and
            agent_tom_net.get('r', 0) > 0.3 and
            agent_tom_net.get('p', 1) < 0.05 and
            abs(agent_tom_net.get('r', 0) - tom_r_ref) < 0.05)
    add_check('tom_network_correlation', cond, f"Agent: {agent_tom_net}, Ref r={tom_r_ref}, p={tom_p_ref}")

    agent_action_net = agent.get('action_network_correlation', {})
    cond = (agent_action_net.get('network') == 'VentralAttention' and
            agent_action_net.get('r', 0) > 0.3 and
            agent_action_net.get('p', 1) < 0.05 and
            abs(agent_action_net.get('r', 0) - action_r_ref) < 0.05)
    add_check('action_network_correlation', cond, f"Agent: {agent_action_net}, Ref r={action_r_ref}, p={action_p_ref}")

    agent_tom_table = agent.get('tom_fdr_table', [])
    tom_table_ok = True
    if not isinstance(agent_tom_table, list) or len(agent_tom_table) != len(regions):
        tom_table_ok = False
    else:
        ref_dict = tom_res.set_index('region').to_dict(orient='index')
        agent_dict = {row['region']: row for row in agent_tom_table}
        for reg in regions:
            if reg not in agent_dict:
                tom_table_ok = False
                break
            a = agent_dict[reg]
            r = ref_dict[reg]
            if abs(a.get('p', -1) - r['p']) > 1e-3 or abs(a.get('q', -1) - r['q']) > 1e-3:
                tom_table_ok = False
                break
    add_check('tom_fdr_table', tom_table_ok, f"Agent table length: {len(agent_tom_table) if isinstance(agent_tom_table, list) else 'N/A'}")

    agent_action_table = agent.get('action_fdr_table', [])
    action_table_ok = True
    if not isinstance(agent_action_table, list) or len(agent_action_table) != len(regions):
        action_table_ok = False
    else:
        ref_dict = action_res.set_index('region').to_dict(orient='index')
        agent_dict = {row['region']: row for row in agent_action_table}
        for reg in regions:
            if reg not in agent_dict:
                action_table_ok = False
                break
            a = agent_dict[reg]
            r = ref_dict[reg]
            if abs(a.get('p', -1) - r['p']) > 1e-3 or abs(a.get('q', -1) - r['q']) > 1e-3:
                action_table_ok = False
                break
    add_check('action_fdr_table', action_table_ok, f"Agent table length: {len(agent_action_table) if isinstance(agent_action_table, list) else 'N/A'}")

    tom_names = ' '.join(agent_tom)
    cond = ('CrusII' in tom_names.replace('_', '')) and ('VIIB' not in tom_names) and ('Vermis' not in tom_names)
    add_check('tom_biological_validity', cond, f"ToM regions: {agent_tom}")

    action_names = ' '.join(agent_action)
    has_viib = 'VIIB' in action_names
    has_vermis = any(v in action_names for v in ['Vermis_VI', 'Vermis_VII', 'Vermis_VIII'])
    no_crus = 'CrusII' not in action_names.replace('_', '')
    cond = has_viib and has_vermis and no_crus
    add_check('action_biological_validity', cond, f"Action regions: {agent_action}")

    total = len(checks)
    passed = sum(1 for c in checks if c['passed'])
    score = passed / total if total > 0 else 0.0

    reward = {"score": score, "reward": score}
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump(reward, f)

    report = {'checks': checks, 'passed': passed, 'total': total, 'score': score}
    with open('/logs/verifier/report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
