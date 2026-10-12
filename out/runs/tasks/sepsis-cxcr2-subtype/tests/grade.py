import json
import os
import sys
import numpy as np
import pandas as pd

DATA_DIR = '/app/data'
OUTPUT_PATH = '/app/output/result.json'
REWARD_DIR = '/logs/verifier'

def load_data():
    meta = pd.read_csv(os.path.join(DATA_DIR, 'metadata.tsv'), sep='\t')
    blood = pd.read_csv(os.path.join(DATA_DIR, 'blood_counts.tsv'), sep='\t', index_col=0)
    liver = pd.read_csv(os.path.join(DATA_DIR, 'liver_counts.tsv'), sep='\t', index_col=0)
    with open(os.path.join(DATA_DIR, 'signatures.json')) as f:
        signatures = json.load(f)
    return meta, blood, liver, signatures

def compute_reference(meta, blood, liver, signatures):
    def score_tissue(counts):
        lib = counts.sum(axis=0)
        cpm = counts.div(lib, axis=1) * 1e6
        log2cpm = np.log2(cpm + 1)
        mean = log2cpm.mean(axis=1)
        std = log2cpm.std(axis=1, ddof=0)
        std[std == 0] = 1.0
        z = log2cpm.sub(mean, axis=0).div(std, axis=0)
        scores = {}
        for mts, genes in signatures.items():
            genes_in = [g for g in genes if g in z.index]
            scores[mts] = z.loc[genes_in].mean(axis=0)
        return pd.DataFrame(scores)
    blood_scores = score_tissue(blood)
    liver_scores = score_tissue(liver)
    combined = (blood_scores + liver_scores) / 2.0
    labels = combined.idxmax(axis=1)
    meta_idx = meta.set_index('mouse_id')
    meta_idx['MTS'] = labels
    summary = {}
    for mts in ['MTS1', 'MTS2', 'MTS3']:
        sub = meta_idx[meta_idx['MTS'] == mts]
        summary[mts] = {
            'n': int(len(sub)),
            'mean_pathogen_log10': float(sub['pathogen_log10'].mean()),
            'mean_IL6': float(sub['IL6_pg_ml'].mean())
        }
    effects = {}
    for mts in ['MTS1', 'MTS2', 'MTS3']:
        sub = meta_idx[meta_idx['MTS'] == mts]
        veh = sub[sub['treatment'] == 'Vehicle']
        dan = sub[sub['treatment'] == 'Danirixin']
        effects[mts] = {
            'survival_diff': float(dan['survival_7d'].mean() - veh['survival_7d'].mean()),
            'pathogen_diff': float(dan['pathogen_log10'].mean() - veh['pathogen_log10'].mean()),
            'netosis_diff': float(dan['NETosis_units'].mean() - veh['NETosis_units'].mean()),
            'hepatic_injury_diff': float(dan['hepatic_injury_ALT'].mean() - veh['hepatic_injury_ALT'].mean())
        }
    return {
        'combined_scores': combined,
        'labels': labels,
        'summary': summary,
        'effects': effects
    }

def load_agent_output():
    if not os.path.exists(OUTPUT_PATH):
        return None
    try:
        with open(OUTPUT_PATH) as f:
            return json.load(f)
    except Exception:
        return None

def check_output(agent, ref, meta):
    checks = []
    def add(name, passed, details=''):
        checks.append({'name': name, 'passed': bool(passed), 'details': details})
    if agent is None:
        add('output_exists', False, 'result.json missing or invalid JSON')
        return checks
    required_keys = ['sample_subtypes', 'signature_scores', 'subtype_summary', 'treatment_effects', 'recommendation']
    has_keys = all(k in agent for k in required_keys)
    add('required_keys', has_keys, 'missing keys: ' + str([k for k in required_keys if k not in agent]))
    ss = agent.get('sample_subtypes', {})
    mouse_ids = set(meta['mouse_id'])
    has_all_mice = isinstance(ss, dict) and set(ss.keys()) == mouse_ids
    valid_labels = isinstance(ss, dict) and all(v in ['MTS1', 'MTS2', 'MTS3'] for v in ss.values())
    add('sample_subtypes', has_all_mice and valid_labels, 'has_all_mice=' + str(has_all_mice) + ', valid_labels=' + str(valid_labels))
    sig_scores = agent.get('signature_scores', {})
    ref_scores = ref['combined_scores']
    score_ok = True
    max_diff = 0.0
    if not isinstance(sig_scores, dict):
        score_ok = False
    else:
        for mid in mouse_ids:
            if mid not in sig_scores:
                score_ok = False
                break
            for mts in ['MTS1', 'MTS2', 'MTS3']:
                if mts not in sig_scores[mid]:
                    score_ok = False
                    break
                agent_val = sig_scores[mid][mts]
                ref_val = float(ref_scores.loc[mid, mts])
                diff = abs(agent_val - ref_val)
                max_diff = max(max_diff, diff)
                if diff > 0.01:
                    score_ok = False
    add('signature_scores', score_ok, 'max_diff=' + str(round(max_diff, 4)))
    correct = 0
    total = len(mouse_ids)
    if isinstance(ss, dict):
        for mid in mouse_ids:
            if ss.get(mid) == ref['labels'].loc[mid]:
                correct += 1
    acc = correct / total if total > 0 else 0
    add('subtype_accuracy', acc >= 34/36, 'accuracy=' + str(round(acc, 4)) + ' (' + str(correct) + '/' + str(total) + ')')
    summary = agent.get('subtype_summary', {})
    summary_ok = True
    summary_details = []
    for mts in ['MTS1', 'MTS2', 'MTS3']:
        if mts not in summary:
            summary_ok = False
            summary_details.append(mts + ' missing')
            continue
        ref_sum = ref['summary'][mts]
        for k in ['n', 'mean_pathogen_log10', 'mean_IL6']:
            if k not in summary[mts]:
                summary_ok = False
                summary_details.append(mts + '.' + k + ' missing')
        if 'mean_pathogen_log10' in summary[mts]:
            diff = abs(summary[mts]['mean_pathogen_log10'] - ref_sum['mean_pathogen_log10'])
            if diff > 0.1:
                summary_ok = False
                summary_details.append(mts + ' pathogen diff ' + str(round(diff, 3)))
        if 'mean_IL6' in summary[mts]:
            ref_val = ref_sum['mean_IL6']
            diff = abs(summary[mts]['mean_IL6'] - ref_val)
            rel_diff = diff / ref_val if ref_val != 0 else diff
            if rel_diff > 0.05:
                summary_ok = False
                summary_details.append(mts + ' IL6 rel diff ' + str(round(rel_diff, 3)))
        if 'n' in summary[mts]:
            if summary[mts]['n'] != ref_sum['n']:
                summary_ok = False
                summary_details.append(mts + ' n mismatch')
    add('subtype_summary', summary_ok, '; '.join(summary_details))
    bio_ok = True
    bio_details = []
    if summary_ok:
        mts1 = summary.get('MTS1', {})
        mts3 = summary.get('MTS3', {})
        if mts3.get('mean_pathogen_log10', -np.inf) <= mts1.get('mean_pathogen_log10', np.inf):
            bio_ok = False
            bio_details.append('MTS3 pathogen not > MTS1')
        if mts3.get('mean_IL6', -np.inf) <= mts1.get('mean_IL6', np.inf):
            bio_ok = False
            bio_details.append('MTS3 IL6 not > MTS1')
    else:
        bio_ok = False
        bio_details.append('summary invalid')
    add('subtype_biology', bio_ok, '; '.join(bio_details))
    effects = agent.get('treatment_effects', {})
    effects_ok = True
    effects_details = []
    for mts in ['MTS1', 'MTS2', 'MTS3']:
        if mts not in effects:
            effects_ok = False
            effects_details.append(mts + ' missing')
            continue
        ref_eff = ref['effects'][mts]
        for k in ['survival_diff', 'pathogen_diff', 'netosis_diff', 'hepatic_injury_diff']:
            if k not in effects[mts]:
                effects_ok = False
                effects_details.append(mts + '.' + k + ' missing')
                continue
            agent_val = effects[mts][k]
            ref_val = ref_eff[k]
            if k == 'survival_diff':
                tol = 0.1
            elif k == 'pathogen_diff':
                tol = 0.2
            elif k == 'netosis_diff':
                tol = 1.0
            else:
                tol = 10.0
            if abs(agent_val - ref_val) > tol:
                effects_ok = False
                effects_details.append(mts + '.' + k + ' diff ' + str(round(abs(agent_val - ref_val), 3)) + ' > ' + str(tol))
    add('treatment_effects_values', effects_ok, '; '.join(effects_details))
    sign_ok = True
    sign_details = []
    if 'MTS3' in effects and 'survival_diff' in effects.get('MTS3', {}):
        if effects['MTS3']['survival_diff'] <= 0:
            sign_ok = False
            sign_details.append('MTS3 survival_diff <= 0')
    else:
        sign_ok = False
        sign_details.append('MTS3 survival_diff missing')
    if 'MTS1' in effects:
        if effects['MTS1'].get('pathogen_diff', -1) <= 0:
            sign_ok = False
            sign_details.append('MTS1 pathogen_diff <= 0')
        if effects['MTS1'].get('netosis_diff', 1) >= 0:
            sign_ok = False
            sign_details.append('MTS1 netosis_diff >= 0')
        if effects['MTS1'].get('hepatic_injury_diff', 1) >= 0:
            sign_ok = False
            sign_details.append('MTS1 hepatic_injury_diff >= 0')
    else:
        sign_ok = False
        sign_details.append('MTS1 effects missing')
    add('treatment_effect_signs', sign_ok, '; '.join(sign_details))
    rec = agent.get('recommendation', {})
    rec_ok = True
    rec_details = []
    if rec.get('preferred_subtype') != 'MTS3':
        rec_ok = False
        rec_details.append('preferred_subtype != MTS3')
    if rec.get('avoid_subtype') != 'MTS1':
        rec_ok = False
        rec_details.append('avoid_subtype != MTS1')
    if not isinstance(rec.get('rationale', ''), str) or len(rec.get('rationale', '').strip()) == 0:
        rec_ok = False
        rec_details.append('rationale missing or empty')
    add('recommendation', rec_ok, '; '.join(rec_details))
    return checks

def main():
    os.makedirs(REWARD_DIR, exist_ok=True)
    meta, blood, liver, signatures = load_data()
    ref = compute_reference(meta, blood, liver, signatures)
    agent = load_agent_output()
    checks = check_output(agent, ref, meta)
    passed = sum(1 for c in checks if c['passed'])
    total = len(checks)
    score = passed / total if total > 0 else 0.0
    reward = {'score': float(score), 'reward': float(score)}
    with open(os.path.join(REWARD_DIR, 'reward.json'), 'w') as f:
        json.dump(reward, f)
    report = {
        'score': float(score),
        'passed': passed,
        'total': total,
        'checks': checks
    }
    with open(os.path.join(REWARD_DIR, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    sys.exit(0)

if __name__ == '__main__':
    main()
