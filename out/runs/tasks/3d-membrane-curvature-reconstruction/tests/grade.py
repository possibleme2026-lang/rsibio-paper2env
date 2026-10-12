#!/usr/bin/env python3
import os
import sys
import json
import csv
import math

sys.path.insert(0, '/tests')
from gold_gen import compute_gold_metrics

RESULTS_DIR = '/app/data/results'
REWARD_DIR = '/logs/verifier'
REWARD_PATH = os.path.join(REWARD_DIR, 'reward.json')
REPORT_PATH = os.path.join(REWARD_DIR, 'report.json')

def load_pred_csv():
    path = os.path.join(RESULTS_DIR, 'curvature_summary.csv')
    if not os.path.isfile(path):
        return None, 'curvature_summary.csv not found'
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        expected = {'volume_id', 'surface_area_nm2', 'mean_abs_H_nm_inv', 'mean_abs_K_nm_inv2'}
        if set(reader.fieldnames) != expected:
            return None, f'CSV columns mismatch: {reader.fieldnames}'
        for row in reader:
            rows.append(row)
    return rows, None

def safe_float(s):
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (ValueError, TypeError):
        return None

def check_format(rows, gold):
    report = {'format_ok': False, 'errors': []}
    if rows is None:
        report['errors'].append('CSV missing or invalid')
        return report
    if len(rows) != 5:
        report['errors'].append(f'Expected 5 rows, got {len(rows)}')
    expected_ids = {'vol_00', 'vol_01', 'vol_02', 'vol_03', 'vol_04'}
    seen_ids = set()
    for row in rows:
        vid = row.get('volume_id', '').strip()
        if vid not in expected_ids:
            report['errors'].append(f'Unknown volume_id: {vid}')
        seen_ids.add(vid)
        for col in ['surface_area_nm2', 'mean_abs_H_nm_inv', 'mean_abs_K_nm_inv2']:
            val = safe_float(row.get(col, ''))
            if val is None:
                report['errors'].append(f'Invalid {col} for {vid}: {row.get(col)}')
    if seen_ids != expected_ids:
        report['errors'].append(f'Missing volume IDs: {expected_ids - seen_ids}')
    report['format_ok'] = len(report['errors']) == 0
    return report

def check_metric(rows, gold, metric, tol):
    if rows is None:
        return 0.0, {'error': 'no rows'}
    details = {}
    correct = 0
    for row in rows:
        vid = row['volume_id']
        if vid not in gold:
            details[vid] = 'unknown id'
            continue
        pred = safe_float(row.get(metric))
        true = gold[vid][metric]
        if pred is None:
            details[vid] = 'invalid value'
            continue
        rel_err = abs(pred - true) / max(abs(true), 1e-12)
        details[vid] = {'pred': pred, 'true': true, 'rel_err': rel_err, 'pass': rel_err <= tol}
        if rel_err <= tol:
            correct += 1
    return correct / 5.0, details

def check_neck(gold_neck):
    path = os.path.join(RESULTS_DIR, 'neck_metrics.json')
    details = {}
    if not os.path.isfile(path):
        return 0.0, {'error': 'neck_metrics.json not found'}
    try:
        with open(path) as f:
            pred = json.load(f)
    except Exception as e:
        return 0.0, {'error': f'JSON parse error: {e}'}
    center = pred.get('neck_center_nm')
    if not isinstance(center, list) or len(center) != 3:
        details['center'] = 'invalid center'
    else:
        true_center = gold_neck['neck_center_nm']
        dist = math.sqrt(sum((float(c) - float(t))**2 for c, t in zip(center, true_center)))
        details['center'] = {'pred': center, 'true': true_center, 'dist': dist, 'pass': dist <= 4.0}
    k1 = pred.get('kappa1_nm_inv')
    k2 = pred.get('kappa2_nm_inv')
    if k1 is None or k2 is None:
        details['signs'] = 'missing kappa'
    else:
        try:
            k1f = float(k1); k2f = float(k2)
            details['signs'] = {'k1': k1f, 'k2': k2f, 'pass': (k1f * k2f) < 0}
        except:
            details['signs'] = 'invalid kappa'
    if k1 is not None and k2 is not None:
        try:
            k1f = float(k1); k2f = float(k2)
            true_k1 = gold_neck['kappa1_nm_inv']
            true_k2 = gold_neck['kappa2_nm_inv']
            mag_ok = (abs(abs(k1f) - abs(true_k1)) / abs(true_k1) <= 0.25 and
                      abs(abs(k2f) - abs(true_k2)) / abs(true_k2) <= 0.25)
            details['magnitudes'] = {'k1': k1f, 'k2': k2f, 'pass': mag_ok}
        except:
            details['magnitudes'] = 'invalid'
    aniso = pred.get('anisotropy_index')
    if aniso is None:
        details['anisotropy'] = 'missing'
    else:
        try:
            aniso_f = float(aniso)
            details['anisotropy'] = {'pred': aniso_f, 'pass': aniso_f >= 0.7}
        except:
            details['anisotropy'] = 'invalid'
    passes = 0
    total = 0
    for key in ['center', 'signs', 'magnitudes', 'anisotropy']:
        total += 1
        val = details.get(key)
        if isinstance(val, dict) and val.get('pass'):
            passes += 1
    return passes / total, details

def main():
    os.makedirs(REWARD_DIR, exist_ok=True)
    gold = compute_gold_metrics()
    rows, err = load_pred_csv()
    report = {'errors': [], 'details': {}}
    if err:
        report['errors'].append(err)
    fmt = check_format(rows, gold)
    report['details']['format'] = fmt
    format_score = 0.2 if fmt['format_ok'] else 0.0
    sa_score, sa_details = check_metric(rows, gold, 'surface_area_nm2', 0.12)
    report['details']['surface_area'] = sa_details
    h_score, h_details = check_metric(rows, gold, 'mean_abs_H_nm_inv', 0.18)
    report['details']['mean_abs_H'] = h_details
    k_score, k_details = check_metric(rows, gold, 'mean_abs_K_nm_inv2', 0.25)
    report['details']['mean_abs_K'] = k_details
    neck_score, neck_details = check_neck(gold['neck'])
    report['details']['neck'] = neck_details
    total_score = format_score + 0.2 * sa_score + 0.2 * h_score + 0.2 * k_score + 0.2 * neck_score
    total_score = round(total_score, 4)
    report['score'] = total_score
    with open(REWARD_PATH, 'w') as f:
        json.dump({'score': total_score, 'reward': total_score}, f)
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(f'Final score: {total_score}')

if __name__ == '__main__':
    main()
