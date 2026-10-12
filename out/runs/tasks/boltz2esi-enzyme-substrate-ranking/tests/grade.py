import os, sys, subprocess, json, math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def _jsonable(o):
    try:
        import numpy as _np
        if isinstance(o, (_np.bool_, _np.integer, _np.floating)):
            return o.item()
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def load_agent():
    features = pd.read_csv('/app/output/features.tsv', sep='\t')
    preds = pd.read_csv('/app/output/predictions.tsv', sep='\t')
    return features, preds

def load_gold():
    features = pd.read_csv('/tmp/gold_features.tsv', sep='\t')
    preds = pd.read_csv('/tmp/gold_predictions.tsv', sep='\t')
    return features, preds

def check_format(features, preds):
    expected_feat_cols = ['enzyme_id','substrate_id','lig_mw','lig_logp','lig_tpsa','lig_hbd','lig_hba','lig_rings','lig_fsp3','lig_charge','pocket_hydrophobicity','pocket_polarity','pocket_charge','pocket_aromatic','kmer_jsd','tanimoto_max']
    expected_pred_cols = ['enzyme_id','substrate_id','predicted_score','rank']
    if list(features.columns) != expected_feat_cols:
        return False
    if list(preds.columns) != expected_pred_cols:
        return False
    if len(features) != 4500 or len(preds) != 4500:
        return False
    if features.isnull().any().any() or preds.isnull().any().any():
        return False
    return True

def check_pair_integrity(features, preds):
    f_pairs = set(zip(features['enzyme_id'], features['substrate_id']))
    p_pairs = set(zip(preds['enzyme_id'], preds['substrate_id']))
    if len(f_pairs) != 4500 or len(p_pairs) != 4500:
        return False
    if f_pairs != p_pairs:
        return False
    return True

def check_score_range(preds):
    return preds['predicted_score'].between(0,1).all()

def check_ranks(preds):
    for enz, grp in preds.groupby('enzyme_id'):
        if len(grp) != 150:
            return False
        if set(grp['rank']) != set(range(1,151)):
            return False
        grp_sorted = grp.sort_values('rank')
        if not grp_sorted['predicted_score'].is_monotonic_decreasing:
            return False
    return True

def check_feature_accuracy(agent_feat, gold_feat):
    import random
    random.seed(42)
    pairs = list(zip(gold_feat['enzyme_id'], gold_feat['substrate_id']))
    sample = random.sample(pairs, 100)
    agent_idx = agent_feat.set_index(['enzyme_id','substrate_id'])
    gold_idx = gold_feat.set_index(['enzyme_id','substrate_id'])
    feat_cols = ['lig_mw','lig_logp','lig_tpsa','lig_hbd','lig_hba','lig_rings','lig_fsp3','lig_charge','pocket_hydrophobicity','pocket_polarity','pocket_charge','pocket_aromatic','kmer_jsd','tanimoto_max']
    tol = {'lig_mw':0.05, 'lig_logp':0.05, 'lig_tpsa':0.05, 'lig_hbd':0.01, 'lig_hba':0.01, 'lig_rings':0.01, 'lig_fsp3':0.01, 'lig_charge':0.01, 'pocket_hydrophobicity':0.02, 'pocket_polarity':0.02, 'pocket_charge':0.02, 'pocket_aromatic':0.02, 'kmer_jsd':0.02, 'tanimoto_max':0.05}
    pass_count = 0
    for pair in sample:
        try:
            a = agent_idx.loc[pair]
            g = gold_idx.loc[pair]
        except KeyError:
            return False
        ok = True
        for col in feat_cols:
            if abs(float(a[col]) - float(g[col])) > tol[col]:
                ok = False
                break
        if ok:
            pass_count += 1
    return pass_count >= 90

def check_rmse(agent_pred, gold_pred):
    merged = agent_pred.merge(gold_pred, on=['enzyme_id','substrate_id'], suffixes=('_agent','_gold'))
    rmse = np.sqrt(np.mean((merged['predicted_score_agent'] - merged['predicted_score_gold'])**2))
    return rmse <= 0.15, rmse

def check_spearman(agent_pred, gold_pred):
    merged = agent_pred.merge(gold_pred, on=['enzyme_id','substrate_id'], suffixes=('_agent','_gold'))
    correlations = []
    for enz, grp in merged.groupby('enzyme_id'):
        if len(grp) < 3: continue
        rho, _ = spearmanr(grp['predicted_score_agent'], grp['predicted_score_gold'])
        if not np.isnan(rho):
            correlations.append(rho)
    avg_rho = np.mean(correlations) if correlations else 0.0
    return avg_rho >= 0.70, avg_rho

def check_top5_overlap(agent_pred, gold_pred):
    merged = agent_pred.merge(gold_pred, on=['enzyme_id','substrate_id'], suffixes=('_agent','_gold'))
    overlaps = []
    for enz, grp in merged.groupby('enzyme_id'):
        top5_agent = set(grp.nlargest(5, 'predicted_score_agent')['substrate_id'])
        top5_gold = set(grp.nlargest(5, 'predicted_score_gold')['substrate_id'])
        overlaps.append(len(top5_agent & top5_gold) / 5.0)
    avg_overlap = np.mean(overlaps) if overlaps else 0.0
    return avg_overlap >= 0.60, avg_overlap

def main():
    subprocess.run([sys.executable, '/tests/gold_gen.py'], check=True)
    agent_feat, agent_pred = load_agent()
    gold_feat, gold_pred = load_gold()
    constraints = []
    c1 = check_format(agent_feat, agent_pred)
    constraints.append(('format', c1))
    c2 = check_pair_integrity(agent_feat, agent_pred)
    constraints.append(('pair_integrity', c2))
    c3 = check_score_range(agent_pred)
    constraints.append(('score_range', c3))
    c4 = check_ranks(agent_pred)
    constraints.append(('ranks', c4))
    c5 = check_feature_accuracy(agent_feat, gold_feat)
    constraints.append(('feature_accuracy', c5))
    c6, rmse = check_rmse(agent_pred, gold_pred)
    constraints.append(('rmse', c6))
    c7, rho = check_spearman(agent_pred, gold_pred)
    constraints.append(('spearman', c7))
    c8, overlap = check_top5_overlap(agent_pred, gold_pred)
    constraints.append(('top5_overlap', c8))
    passed = sum(1 for _, ok in constraints if ok)
    total = len(constraints)
    score = passed / total
    reward = score
    result = {'score': score, 'reward': reward, 'details': {'constraints': {name: ok for name, ok in constraints}, 'rmse': float(rmse) if 'rmse' in locals() else None, 'spearman': float(rho) if 'rho' in locals() else None, 'top5_overlap': float(overlap) if 'overlap' in locals() else None}}
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump(result, f, indent=2, default=_jsonable)
    print(json.dumps(result, indent=2, default=_jsonable))

if __name__ == '__main__':
    main()
