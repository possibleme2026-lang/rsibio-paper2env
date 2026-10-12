import sys
import os
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

sys.path.insert(0, '/tests')
import gold_gen

def load_predictions(path):
    preds = {}
    queries = set()
    targets = set()
    with open(path, 'r') as f:
        header = f.readline().strip().split('\t')
        if header != ['query_id', 'target_id', 'predicted_tm']:
            raise ValueError('Bad header: ' + str(header))
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                raise ValueError('Bad line: ' + line)
            q, t, val = parts
            try:
                v = float(val)
            except:
                raise ValueError('Bad float: ' + val)
            if not (0 <= v <= 1):
                raise ValueError('Out of range: ' + val)
            if (q, t) in preds:
                raise ValueError('Duplicate: ' + q + ', ' + t)
            preds[(q, t)] = v
            queries.add(q)
            targets.add(t)
    return preds, queries, targets

def validate_format(preds, queries, targets):
    expected_queries = set(f'Q{i+1:02d}' for i in range(30))
    expected_targets = set(f'DB{i+1:03d}' for i in range(200))
    if queries != expected_queries:
        return False, 'Query IDs mismatch. Expected 30 queries Q01-Q30.'
    if targets != expected_targets:
        return False, 'Target IDs mismatch. Expected 200 targets DB001-DB200.'
    if len(preds) != 6000:
        return False, f'Expected 6000 rows, got {len(preds)}.'
    for q in expected_queries:
        for t in expected_targets:
            if (q, t) not in preds:
                return False, f'Missing pair: {q}, {t}'
    return True, 'Format OK'

def compute_metrics(preds, gold):
    expected_queries = set(f'Q{i+1:02d}' for i in range(30))
    expected_targets = set(f'DB{i+1:03d}' for i in range(200))
    keys = list(preds.keys())
    y_pred = np.array([preds[k] for k in keys])
    y_true = np.array([gold[k] for k in keys])
    rho, _ = spearmanr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    top1_correct = 0
    for q in expected_queries:
        q_preds = {t: preds[(q, t)] for t in expected_targets}
        best_t = max(q_preds, key=q_preds.get)
        if gold[(q, best_t)] >= 0.5:
            top1_correct += 1
    top1_prec = top1_correct / len(expected_queries)
    f1s = []
    for q in expected_queries:
        tp = 0
        fp = 0
        fn = 0
        for t in expected_targets:
            true_pos = gold[(q, t)] >= 0.5
            pred_pos = preds[(q, t)] >= 0.5
            if true_pos and pred_pos:
                tp += 1
            elif not true_pos and pred_pos:
                fp += 1
            elif true_pos and not pred_pos:
                fn += 1
        if tp + fp + fn == 0:
            f1 = 1.0
        else:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    avg_f1 = np.mean(f1s)
    pos_preds = []
    neg_preds = []
    for k in keys:
        if gold[k] >= 0.5:
            pos_preds.append(preds[k])
        else:
            neg_preds.append(preds[k])
    calib = None
    if len(pos_preds) > 0 and len(neg_preds) > 0:
        calib = np.mean(pos_preds) - np.mean(neg_preds)
    return rho, rmse, top1_prec, avg_f1, calib

def main():
    gold = gold_gen.get_gold()
    pred_path = '/app/output/predictions.tsv'
    if not os.path.exists(pred_path):
        print('predictions.tsv not found')
        result = {'score': 0.0, 'reward': 0.0}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump(result, f)
        return
    try:
        preds, queries, targets = load_predictions(pred_path)
    except Exception as e:
        print('Format error: ' + str(e))
        result = {'score': 0.0, 'reward': 0.0}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open('/logs/verifier/reward.json', 'w') as f:
            json.dump(result, f)
        return
    constraints = {}
    format_ok, msg = validate_format(preds, queries, targets)
    constraints['format'] = format_ok
    if not format_ok:
        print('Format constraint failed: ' + msg)
        for c in ['spearman', 'rmse', 'top1', 'f1', 'calibration']:
            constraints[c] = False
    else:
        rho, rmse, top1_prec, avg_f1, calib = compute_metrics(preds, gold)
        constraints['spearman'] = bool(rho >= 0.5)
        constraints['rmse'] = bool(rmse <= 0.20)
        constraints['top1'] = bool(top1_prec >= 0.6)
        constraints['f1'] = bool(avg_f1 >= 0.6)
        constraints['calibration'] = bool(calib is not None and calib >= 0.25)
    passed = sum(1 for v in constraints.values() if v)
    total = len(constraints)
    score = passed / total if total > 0 else 0.0
    report = {'score': score, 'constraints': constraints, 'detail': {}}
    if format_ok:
        report['detail'] = {
            'spearman': float(rho),
            'rmse': float(rmse),
            'top1_precision': float(top1_prec),
            'avg_f1': float(avg_f1),
            'calibration': float(calib) if calib is not None else None,
            'n_pairs': len(preds)
        }
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': float(score), 'reward': float(score)}, f)
    with open('/logs/verifier/report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
