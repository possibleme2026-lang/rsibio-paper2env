#!/usr/bin/env python3
import os, sys, json
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/tests')
import gold_gen

OUT = '/app/output'
REWARD_DIR = '/logs/verifier'
os.makedirs(REWARD_DIR, exist_ok=True)

def bh_fdr(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n+1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out

def compute_reference(train_df, pathways_df):
    meta = ['sample_id','participant_id','cohort','condition','timepoint']
    proteins = [c for c in train_df.columns if c not in meta]
    groups = train_df.groupby(['cohort','condition'])
    keep = []
    for prot in proteins:
        ok = True
        for _, g in groups:
            if g[prot].isna().mean() > 0.2:
                ok = False
                break
        if ok:
            keep.append(prot)
    proteins = keep
    records = []
    for cohort in ['A','B']:
        for cond in ['stimulated','control']:
            sub = train_df[(train_df.cohort==cohort) & (train_df.condition==cond)]
            pre = sub[sub.timepoint=='pre'].set_index('participant_id')
            post = sub[sub.timepoint=='post'].set_index('participant_id')
            common = pre.index.intersection(post.index)
            for prot in proteins:
                a = pre.loc[common, prot].astype(float)
                b = post.loc[common, prot].astype(float)
                valid = ~(a.isna() | b.isna())
                if valid.sum() < 10:
                    continue
                diff = (b[valid] - a[valid]).values
                mean_fc = float(np.mean(diff))
                tstat, pval = stats.ttest_rel(b[valid], a[valid])
                records.append({'cohort':cohort,'condition':cond,'protein':prot,'log2FC':mean_fc,'pval':float(pval),'n':int(valid.sum())})
    res = pd.DataFrame(records)
    for (cohort, cond), idx in res.groupby(['cohort','condition']).groups.items():
        pvals = res.loc[idx, 'pval'].values
        res.loc[idx, 'FDR'] = bh_fdr(pvals)
    sig = []
    sig_stats = {}
    for prot in proteins:
        rows = res[res.protein==prot]
        if len(rows) < 4:
            continue
        def get(c, cond):
            r = rows[(rows.cohort==c) & (rows.condition==cond)]
            return r.iloc[0] if len(r)>0 else None
        sa = get('A','stimulated'); sb = get('B','stimulated')
        ca = get('A','control'); cb = get('B','control')
        if any(_x is None for _x in (sa,sb,ca,cb)):
            continue
        if abs(sa.log2FC)>=1.0 and abs(sb.log2FC)>=1.0 and sa.FDR<0.05 and sb.FDR<0.05 and np.sign(sa.log2FC)==np.sign(sb.log2FC) and ca.FDR>=0.05 and cb.FDR>=0.05:
            sig.append(prot)
            sig_stats[prot] = {
                'log2FC_A_stim': float(sa.log2FC), 'FDR_A_stim': float(sa.FDR),
                'log2FC_B_stim': float(sb.log2FC), 'FDR_B_stim': float(sb.FDR),
                'log2FC_A_ctrl': float(ca.log2FC), 'FDR_A_ctrl': float(ca.FDR),
                'log2FC_B_ctrl': float(cb.log2FC), 'FDR_B_ctrl': float(cb.FDR),
                'direction': 'up' if sa.log2FC>0 else 'down'
            }
    universe = set(proteins)
    enr = []
    for path, group in pathways_df.groupby('pathway'):
        pp = set(group.protein) & universe
        if not pp:
            continue
        ov = set(sig) & pp
        pv = stats.hypergeom.sf(len(ov)-1, len(universe), len(pp), len(sig))
        enr.append({'pathway': path, 'p_value': float(pv), 'overlap_count': len(ov)})
    enr = pd.DataFrame(enr).sort_values(['p_value','pathway']).reset_index(drop=True)
    return sig, sig_stats, enr

def main():
    train_df, blind_df, blind_labels, pathways_df = gold_gen.generate()
    ref_sig, ref_stats, ref_enr = compute_reference(train_df, pathways_df)
    constraints = {}
    agent_sig = None
    try:
        agent_sig = pd.read_csv(os.path.join(OUT,'signature.tsv'), sep='\t')
        c1 = (len(agent_sig)==14 and agent_sig['protein'].nunique()==14)
    except Exception:
        c1 = False
    constraints['C1_signature_14_proteins'] = float(c1)
    c2 = False
    if c1:
        try:
            c2 = set(agent_sig['protein']) == set(ref_sig)
        except Exception:
            c2 = False
    constraints['C2_signature_set'] = float(c2)
    c3 = False
    if c1 and c2:
        c3 = True
        required = ['log2FC_A_stim','FDR_A_stim','log2FC_B_stim','FDR_B_stim','direction','log2FC_A_ctrl','FDR_A_ctrl','log2FC_B_ctrl','FDR_B_ctrl']
        try:
            for _, row in agent_sig.iterrows():
                p = row['protein']
                ref = ref_stats.get(p)
                if ref is None:
                    c3 = False; break
                if not all(col in row for col in required):
                    c3 = False; break
                if abs(float(row['log2FC_A_stim']) - ref['log2FC_A_stim']) > 0.25:
                    c3 = False; break
                if abs(float(row['log2FC_B_stim']) - ref['log2FC_B_stim']) > 0.25:
                    c3 = False; break
                if not (float(row['FDR_A_stim']) < 0.05 and float(row['FDR_B_stim']) < 0.05):
                    c3 = False; break
                if not (float(row['FDR_A_ctrl']) >= 0.05 and float(row['FDR_B_ctrl']) >= 0.05):
                    c3 = False; break
                if str(row['direction']).lower() != ref['direction']:
                    c3 = False; break
        except Exception:
            c3 = False
    constraints['C3_signature_statistics'] = float(c3)
    c4 = False
    try:
        agent_enr = pd.read_csv(os.path.join(OUT,'enrichment.tsv'), sep='\t')
        if len(agent_enr) >= 3:
            ref_top = ref_enr.head(3)
            if list(agent_enr.head(3)['pathway']) == list(ref_top['pathway']):
                c4 = True
                for _, row in agent_enr.head(3).iterrows():
                    ref_row = ref_top[ref_top['pathway']==row['pathway']].iloc[0]
                    if abs(float(row['p_value']) - ref_row['p_value']) > 1e-6:
                        c4 = False; break
    except Exception:
        c4 = False
    constraints['C4_enrichment_top3'] = float(c4)
    pred = None
    c5 = False
    try:
        pred = pd.read_csv(os.path.join(OUT,'predictions.csv'))
        if len(pred)==112 and 'sample_id' in pred.columns and 'post_probability' in pred.columns:
            if pred['post_probability'].between(0,1).all() and set(pred['sample_id']) == set(blind_labels['sample_id']):
                c5 = True
    except Exception:
        c5 = False
    constraints['C5_predictions_format'] = float(c5)
    c6 = False; c7 = False
    if c5:
        try:
            merged = pred.merge(blind_labels, on='sample_id', how='left')
            stim = merged[merged.condition=='stimulated']
            ctrl = merged[merged.condition=='control']
            y_stim = (stim.timepoint=='post').astype(int)
            y_ctrl = (ctrl.timepoint=='post').astype(int)
            auroc_stim = roc_auc_score(y_stim, stim.post_probability)
            auroc_ctrl = roc_auc_score(y_ctrl, ctrl.post_probability)
            c6 = auroc_stim >= 0.85
            c7 = auroc_ctrl <= 0.65
        except Exception:
            pass
    constraints['C6_AUROC_stimulated'] = float(c6)
    constraints['C7_AUROC_control'] = float(c7)
    score = sum(constraints.values()) / len(constraints)
    reward = {'score': score, 'reward': score, 'constraints': constraints}
    with open(os.path.join(REWARD_DIR,'reward.json'),'w') as f:
        json.dump(reward, f)
    print(json.dumps(reward, indent=2))

if __name__ == '__main__':
    main()
