#!/usr/bin/env python3
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

ALLOWED_ION_TYPES = set()
for ad in ['M+H', 'M+Na', 'M+NH4', 'M+K']:
    for iso in ['iso0', 'iso1', 'iso2']:
        ALLOWED_ION_TYPES.add(f'{ad}_{iso}')
ALLOWED_ION_TYPES.add('noise')

ADDUCT_MASS = {'M+H': 1.007276, 'M+Na': 22.989218, 'M+NH4': 18.033823, 'M+K': 38.963158}
ISOTOPE_SHIFT = {'iso0': 0.0, 'iso1': 1.003355, 'iso2': 2.006710}

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def write_reward(score, reward, constraints, metrics, error=None):
    os.makedirs('/logs/verifier', exist_ok=True)
    result = {'score': score, 'reward': reward, 'constraints': constraints, 'metrics': metrics}
    if error:
        result['error'] = error
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump(result, f, indent=2)

def main():
    workdir = tempfile.mkdtemp()
    hidden_features = os.path.join(workdir, 'hidden_features.csv')
    hidden_labels = os.path.join(workdir, 'hidden_labels.csv')
    pred_path = os.path.join(workdir, 'pred.csv')

    try:
        subprocess.run([
            sys.executable, '/tests/gold_gen.py',
            '--seed', '12345',
            '--n_compounds', '50',
            '--n_noise', '200',
            '--out_features', hidden_features,
            '--out_labels', hidden_labels
        ], check=True, timeout=120)
    except Exception as e:
        write_reward(0.0, 0.0, {}, {}, error=f'gold_gen failed: {e}')
        return

    try:
        subprocess.run([
            sys.executable, '/app/predict.py',
            hidden_features, pred_path
        ], check=True, timeout=300)
    except subprocess.TimeoutExpired:
        write_reward(0.0, 0.0, {'runtime': False}, {}, error='timeout')
        return
    except Exception as e:
        write_reward(0.0, 0.0, {}, {}, error=f'agent script failed: {e}')
        return

    if not os.path.exists(pred_path):
        write_reward(0.0, 0.0, {}, {}, error='output file not found')
        return

    features = read_csv(hidden_features)
    gold = read_csv(hidden_labels)
    pred = read_csv(pred_path)

    constraints = {}
    metrics = {}

    expected_header = ['feature_id', 'is_valid', 'compound_id', 'ion_type']
    actual_header = list(pred[0].keys()) if pred else []
    constraints['format'] = (actual_header == expected_header)

    input_ids = set(int(f['feature_id']) for f in features)
    try:
        pred_ids = [int(r['feature_id']) for r in pred]
        constraints['ids'] = (len(pred_ids) == len(input_ids) and set(pred_ids) == input_ids and len(pred_ids) == len(set(pred_ids)))
    except Exception:
        constraints['ids'] = False

    gold_dict = {int(r['feature_id']): r for r in gold}
    pred_dict = {}
    for r in pred:
        try:
            fid = int(r['feature_id'])
            pred_dict[fid] = {
                'is_valid': int(r['is_valid']),
                'compound_id': int(r['compound_id']),
                'ion_type': r['ion_type']
            }
        except Exception:
            pass

    values_ok = True
    for fid, r in pred_dict.items():
        if r['is_valid'] not in (0, 1):
            values_ok = False
            break
        if r['is_valid'] == 0:
            if r['compound_id'] != -1 or r['ion_type'] != 'noise':
                values_ok = False
                break
        else:
            if r['compound_id'] < 1 or r['ion_type'] not in ALLOWED_ION_TYPES:
                values_ok = False
                break
    constraints['values'] = values_ok

    tp = fp = fn = tn = 0
    for fid, g in gold_dict.items():
        g_val = int(g['is_valid'])
        p = pred_dict.get(fid)
        p_val = p['is_valid'] if p else 0
        if g_val == 1 and p_val == 1:
            tp += 1
        elif g_val == 0 and p_val == 1:
            fp += 1
        elif g_val == 1 and p_val == 0:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    metrics['detection_f1'] = f1
    metrics['detection_precision'] = precision
    metrics['detection_recall'] = recall
    constraints['detection_f1'] = (f1 >= 0.90)

    correct_ann = 0
    total_gold_valid = 0
    for fid, g in gold_dict.items():
        if int(g['is_valid']) == 1:
            total_gold_valid += 1
            p = pred_dict.get(fid)
            if p and p['is_valid'] == 1 and p['ion_type'] == g['ion_type']:
                correct_ann += 1
    ann_acc = correct_ann / total_gold_valid if total_gold_valid > 0 else 0.0
    metrics['annotation_accuracy'] = ann_acc
    constraints['annotation_accuracy'] = (ann_acc >= 0.85)

    gold_valid_fids = [fid for fid, g in gold_dict.items() if int(g['is_valid']) == 1]
    pred_comp = {}
    for fid in gold_valid_fids:
        p = pred_dict.get(fid)
        if p and p['is_valid'] == 1:
            pred_comp[fid] = p['compound_id']
        else:
            pred_comp[fid] = -fid
    tp_pair = fp_pair = fn_pair = 0
    n = len(gold_valid_fids)
    for i in range(n):
        fid1 = gold_valid_fids[i]
        g1 = int(gold_dict[fid1]['compound_id'])
        for j in range(i + 1, n):
            fid2 = gold_valid_fids[j]
            g2 = int(gold_dict[fid2]['compound_id'])
            same_gold = (g1 == g2)
            same_pred = (pred_comp[fid1] == pred_comp[fid2])
            if same_gold and same_pred:
                tp_pair += 1
            elif not same_gold and same_pred:
                fp_pair += 1
            elif same_gold and not same_pred:
                fn_pair += 1
    pair_prec = tp_pair / (tp_pair + fp_pair) if (tp_pair + fp_pair) > 0 else 0.0
    pair_rec = tp_pair / (tp_pair + fn_pair) if (tp_pair + fn_pair) > 0 else 0.0
    pair_f1 = 2 * pair_prec * pair_rec / (pair_prec + pair_rec) if (pair_prec + pair_rec) > 0 else 0.0
    metrics['grouping_f1'] = pair_f1
    constraints['grouping_f1'] = (pair_f1 >= 0.85)

    groups = defaultdict(list)
    for fid, p in pred_dict.items():
        if p['is_valid'] == 1:
            groups[p['compound_id']].append(fid)
    consistent_groups = 0
    total_groups = 0
    feature_mz = {int(f['feature_id']): float(f['mz']) for f in features}
    for cid, fids in groups.items():
        if len(fids) < 2:
            continue
        total_groups += 1
        masses = []
        ok = True
        for fid in fids:
            p = pred_dict[fid]
            ion_type = p['ion_type']
            if ion_type == 'noise' or ion_type not in ALLOWED_ION_TYPES:
                ok = False
                break
            ad, iso = ion_type.rsplit('_', 1)
            shift = ADDUCT_MASS[ad] + ISOTOPE_SHIFT[iso]
            mz = feature_mz.get(fid)
            if mz is None:
                ok = False
                break
            masses.append(mz - shift)
        if ok and len(masses) >= 2:
            mean_m = sum(masses) / len(masses)
            max_ppm = max(abs(m - mean_m) / mean_m * 1e6 for m in masses)
            if max_ppm <= 10.0:
                consistent_groups += 1
    mass_consistency = consistent_groups / total_groups if total_groups > 0 else 1.0
    metrics['mass_consistency'] = mass_consistency
    metrics['total_groups'] = total_groups
    metrics['consistent_groups'] = consistent_groups
    constraints['mass_consistency'] = (mass_consistency >= 0.90)

    constraints['runtime'] = True

    total_constraints = len(constraints)
    passed = sum(1 for v in constraints.values() if v)
    score = passed / total_constraints if total_constraints > 0 else 0.0
    reward = score

    write_reward(score, reward, constraints, metrics)

if __name__ == '__main__':
    main()