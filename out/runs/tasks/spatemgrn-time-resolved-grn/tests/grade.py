import os
import json
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append('/tests')
from gold_gen import reference_edges

OUTPUT_DIR = '/logs/verifier'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def write_reward(score, reward):
    with open(os.path.join(OUTPUT_DIR, 'reward.json'), 'w') as f:
        json.dump({'score': score, 'reward': reward}, f)

def load_agent_output(path='/app/output/edges.csv'):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

def check_constraints(agent):
    results = {}
    expected_cols = ['timepoint','spatial_unit','source','target','score']
    results['c1_format'] = list(agent.columns) == expected_cols
    t_vals = [0,1,2]
    u_vals = ['Q1','Q2','Q3','Q4']
    required = {(t,u) for t in t_vals for u in u_vals}
    try:
        present = set(zip(agent['timepoint'], agent['spatial_unit']))
        results['c2_coverage'] = required.issubset(present)
    except Exception:
        results['c2_coverage'] = False
    c3 = True
    for t in t_vals:
        for u in u_vals:
            sub = agent[(agent['timepoint']==t) & (agent['spatial_unit']==u)]
            cnt = len(sub)
            if cnt < 10 or cnt > 100:
                c3 = False
                break
        if not c3:
            break
    results['c3_edge_count'] = c3
    c4 = True
    if not agent['score'].between(0,1).all():
        c4 = False
    if (agent['source'] == agent['target']).any():
        c4 = False
    if c4:
        for t in t_vals:
            for u in u_vals:
                sub = agent[(agent['timepoint']==t) & (agent['spatial_unit']==u)]
                edges = set()
                for _, row in sub.iterrows():
                    key = frozenset([row['source'], row['target']])
                    if key in edges:
                        c4 = False
                        break
                    edges.add(key)
                if not c4:
                    break
            if not c4:
                break
    results['c4_valid_edges'] = c4
    c5 = True
    try:
        sorted_agent = agent.sort_values(['timepoint','spatial_unit','score'], ascending=[True,True,False])
        if not agent.reset_index(drop=True).equals(sorted_agent.reset_index(drop=True)):
            c5 = False
        if c5:
            for t in t_vals:
                for u in u_vals:
                    sub = agent[(agent['timepoint']==t) & (agent['spatial_unit']==u)]
                    if not sub['score'].is_monotonic_decreasing:
                        c5 = False
                        break
                if not c5:
                    break
    except Exception:
        c5 = False
    results['c5_sorting'] = c5
    return results

def compute_auroc(agent, ref):
    t_vals = [0,1,2]
    u_vals = ['Q1','Q2','Q3','Q4']
    aurocs = []
    for t in t_vals:
        for u in u_vals:
            ref_sub = ref[(ref['timepoint']==t) & (ref['spatial_unit']==u)]
            ag_sub = agent[(agent['timepoint']==t) & (agent['spatial_unit']==u)]
            if len(ref_sub) == 0:
                aurocs.append(0.5)
                continue
            ref_sorted = ref_sub.sort_values('score', ascending=False)
            top_n = min(10, len(ref_sorted))
            positives = set()
            for _, row in ref_sorted.head(top_n).iterrows():
                key = (min(row['source'], row['target']), max(row['source'], row['target']))
                positives.add(key)
            union = set()
            for _, row in ref_sub.iterrows():
                key = (min(row['source'], row['target']), max(row['source'], row['target']))
                union.add(key)
            ag_scores = {}
            for _, row in ag_sub.iterrows():
                key = (min(row['source'], row['target']), max(row['source'], row['target']))
                union.add(key)
                ag_scores[key] = row['score']
            y_true = []
            y_score = []
            for key in union:
                y_true.append(1 if key in positives else 0)
                y_score.append(ag_scores.get(key, 0.0))
            if len(set(y_true)) < 2:
                aurocs.append(0.5)
            else:
                try:
                    aurocs.append(roc_auc_score(y_true, y_score))
                except Exception:
                    aurocs.append(0.5)
    return np.mean(aurocs) if aurocs else 0.5

def main():
    agent = load_agent_output()
    if agent is None:
        write_reward(0.0, 0.0)
        print('Agent output missing or unreadable.')
        return
    constraints = check_constraints(agent)
    num_passed = sum(1 for v in constraints.values() if v)
    try:
        ref = reference_edges()
    except Exception as e:
        print(f'Reference generation failed: {e}')
        ref = pd.DataFrame(columns=['timepoint','spatial_unit','source','target','score'])
    mean_auroc = compute_auroc(agent, ref)
    score = 0.4 * (num_passed / 5.0) + 0.6 * mean_auroc
    score = min(max(score, 0.0), 1.0)
    write_reward(score, score)
    print(f'Constraints: {constraints}')
    print(f'Constraints passed: {num_passed}/5')
    print(f'Mean AUROC: {mean_auroc:.4f}')
    print(f'Final score: {score:.4f}')

if __name__ == '__main__':
    main()
