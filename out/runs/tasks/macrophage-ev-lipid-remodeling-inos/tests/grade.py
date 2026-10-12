#!/usr/bin/env python3
import os, json, math
import pandas as pd
import numpy as np
from gold_gen import reference_analysis

def load_agent():
    lipid_path = '/app/results/ev_lipid_signature.csv'
    summary_path = '/app/results/ev_summary.json'
    if not os.path.exists(lipid_path) or not os.path.exists(summary_path):
        return None, None
    try:
        agent_lipid = pd.read_csv(lipid_path)
    except Exception:
        return None, None
    try:
        with open(summary_path) as f:
            agent_summary = json.load(f)
    except Exception:
        return None, None
    return agent_lipid, agent_summary

def check_schema(agent_lipid, agent_summary):
    if agent_lipid is None or agent_summary is None:
        return 0.0
    required_cols = ['feature_id', 'lipid_class', 'log2FC_BL_vs_Control', 'fdr_BL', 'log2FC_LPS_vs_Control', 'fdr_LPS', 'signature_group']
    for c in required_cols:
        if c not in agent_lipid.columns:
            return 0.0
    required_keys = ['iNOS_positive_conditions', 'endothelial_induction', 'signature_counts']
    for k in required_keys:
        if k not in agent_summary:
            return 0.0
    return 1.0

def score_annotation(agent_lipid, ref_lipid):
    if agent_lipid is None:
        return 0.0
    ref = ref_lipid.set_index('feature_id')
    agent = agent_lipid.set_index('feature_id')
    common = agent.index.intersection(ref.index)
    if len(common) == 0:
        return 0.0
    correct = 0
    for fid in common:
        if agent.loc[fid, 'lipid_class'] == ref.loc[fid, 'lipid_class']:
            correct += 1
    frac = correct / len(common)
    return min(1.0, frac / 0.95)

def score_filtering(agent_lipid, ref_lipid):
    if agent_lipid is None:
        return 0.0
    ref_ids = set(ref_lipid['feature_id'])
    agent_ids = set(agent_lipid['feature_id'])
    if len(ref_ids) == 0:
        return 0.0
    tp = len(ref_ids & agent_ids)
    fp = len(agent_ids - ref_ids)
    fn = len(ref_ids - agent_ids)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp+fp) > 0 else 0
    rec = tp / (tp + fn) if (tp+fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return f1

def score_differential(agent_lipid, ref_lipid):
    if agent_lipid is None:
        return 0.0
    ref = ref_lipid.set_index('feature_id')
    agent = agent_lipid.set_index('feature_id')
    common = agent.index.intersection(ref.index)
    if len(common) == 0:
        return 0.0
    common = list(common)
    if len(common) > 50:
        common = np.random.RandomState(42).choice(common, 50, replace=False)
    checks = 0
    correct = 0
    for fid in common:
        for col, tol in [('log2FC_BL_vs_Control', 0.1), ('fdr_BL', 0.01), ('log2FC_LPS_vs_Control', 0.1), ('fdr_LPS', 0.01)]:
            checks += 1
            try:
                av = float(agent.loc[fid, col])
                rv = float(ref.loc[fid, col])
                if abs(av - rv) <= tol:
                    correct += 1
            except Exception:
                pass
    return correct / checks if checks > 0 else 0.0

def score_signature(agent_lipid, ref_lipid):
    if agent_lipid is None:
        return 0.0
    ref = ref_lipid.set_index('feature_id')
    agent = agent_lipid.set_index('feature_id')
    common = agent.index.intersection(ref.index)
    if len(common) == 0:
        return 0.0
    classes = ['BL_enriched', 'LPS_enriched', 'shared', 'none']
    f1s = []
    for cls in classes:
        tp = sum(1 for fid in common if agent.loc[fid, 'signature_group'] == cls and ref.loc[fid, 'signature_group'] == cls)
        fp = sum(1 for fid in common if agent.loc[fid, 'signature_group'] == cls and ref.loc[fid, 'signature_group'] != cls)
        fn = sum(1 for fid in common if agent.loc[fid, 'signature_group'] != cls and ref.loc[fid, 'signature_group'] == cls)
        prec = tp / (tp + fp) if (tp+fp) > 0 else 0
        rec = tp / (tp + fn) if (tp+fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
    return sum(f1s) / len(f1s)

def score_inos(agent_summary, ref_summary):
    if agent_summary is None:
        return 0.0
    ref = set(ref_summary['iNOS_positive_conditions'])
    agent = set(agent_summary.get('iNOS_positive_conditions', []))
    conditions = ['Control', 'BL', 'LPS']
    correct = 0
    for cond in conditions:
        if (cond in ref) == (cond in agent):
            correct += 1
    return correct / len(conditions)

def score_endothelial(agent_summary, ref_summary):
    if agent_summary is None:
        return 0.0
    ref = ref_summary['endothelial_induction']
    agent = agent_summary.get('endothelial_induction', {})
    pairs = []
    for cyto in ['IL6', 'CCL5', 'ICAM1']:
        for cond in ['BL', 'LPS']:
            pairs.append((cyto, cond))
    correct = 0
    for cyto, cond in pairs:
        ref_list = ref.get(cyto, [])
        agent_list = agent.get(cyto, [])
        if (cond in ref_list) == (cond in agent_list):
            correct += 1
    return correct / len(pairs)

def score_biological(agent_lipid, ref_lipid):
    if agent_lipid is None:
        return 0.0
    bl = agent_lipid[agent_lipid['signature_group'] == 'BL_enriched']
    if len(bl) == 0:
        return 0.0
    prop = bl['lipid_class'].isin(['SaturatedFattyAcid', 'Ceramide']).mean()
    return min(1.0, prop / 0.8)

def main():
    ref = reference_analysis('/app/data')
    agent_lipid, agent_summary = load_agent()
    scores = {
        'schema': check_schema(agent_lipid, agent_summary),
        'annotation': 0.0,
        'filtering': 0.0,
        'differential': 0.0,
        'signature': 0.0,
        'inos': 0.0,
        'endothelial': 0.0,
        'biological': 0.0
    }
    if agent_lipid is not None and agent_summary is not None:
        scores['annotation'] = score_annotation(agent_lipid, ref['lipid_signature'])
        scores['filtering'] = score_filtering(agent_lipid, ref['lipid_signature'])
        scores['differential'] = score_differential(agent_lipid, ref['lipid_signature'])
        scores['signature'] = score_signature(agent_lipid, ref['lipid_signature'])
        scores['inos'] = score_inos(agent_summary, ref['summary'])
        scores['endothelial'] = score_endothelial(agent_summary, ref['summary'])
        scores['biological'] = score_biological(agent_lipid, ref['lipid_signature'])
    total = sum(scores.values()) / len(scores)
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': total, 'reward': total}, f)

if __name__ == '__main__':
    main()
