#!/usr/bin/env python3
"""Grader for CHO cell stability prediction task."""
import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, precision_score

sys.path.append('/tests')
from gold_gen import generate_data

def main():
    _, _, _, causal_features, y_test, test_sample_ids = generate_data()
    gold_df = pd.DataFrame({'sample_id': test_sample_ids, 'true_label': y_test})
    pred_path = '/app/output/predictions.csv'
    imp_path = '/app/output/feature_importance.csv'
    constraints = {}
    details = {}
    # C1: predictions format
    if not os.path.exists(pred_path):
        constraints['C1_predictions_format'] = False
        details['C1'] = 'predictions.csv missing'
    else:
        try:
            pred_df = pd.read_csv(pred_path)
            req = ['sample_id', 'predicted_label', 'probability_stable']
            if not all(c in pred_df.columns for c in req):
                constraints['C1_predictions_format'] = False
                details['C1'] = f'missing columns: {set(req) - set(pred_df.columns)}'
            elif len(pred_df) != 400:
                constraints['C1_predictions_format'] = False
                details['C1'] = f'expected 400 rows, got {len(pred_df)}'
            elif set(pred_df['sample_id']) != set(test_sample_ids):
                constraints['C1_predictions_format'] = False
                details['C1'] = 'sample_id set does not match test set'
            else:
                constraints['C1_predictions_format'] = True
                details['C1'] = 'ok'
        except Exception as e:
            constraints['C1_predictions_format'] = False
            details['C1'] = f'parse error: {str(e)}'
    # C2: valid values
    if constraints.get('C1_predictions_format', False):
        try:
            valid_labels = pred_df['predicted_label'].isin([0, 1]).all()
            valid_probs = pred_df['probability_stable'].between(0, 1).all()
            no_nan = not pred_df[['predicted_label', 'probability_stable']].isna().any().any()
            constraints['C2_values_valid'] = bool(valid_labels and valid_probs and no_nan)
            details['C2'] = f'valid_labels={valid_labels}, valid_probs={valid_probs}, no_nan={no_nan}'
        except Exception as e:
            constraints['C2_values_valid'] = False
            details['C2'] = str(e)
    else:
        constraints['C2_values_valid'] = False
        details['C2'] = 'skipped due to C1 failure'
    # C3-C7: performance metrics
    if constraints.get('C1_predictions_format', False):
        merged = pred_df.merge(gold_df, on='sample_id', how='inner')
        if len(merged) != 400:
            for c in ['C3_f1', 'C4_mcc', 'C5_recall', 'C6_precision', 'C7_positive_rate']:
                constraints[c] = False
                details[c] = 'merged length mismatch'
        else:
            y_true = merged['true_label'].values
            y_pred = merged['predicted_label'].values
            f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            constraints['C3_f1'] = bool(f1 >= 0.50)
            details['C3'] = {'f1': float(f1), 'threshold': 0.50}
            mcc = matthews_corrcoef(y_true, y_pred)
            constraints['C4_mcc'] = bool(mcc >= 0.15)
            details['C4'] = {'mcc': float(mcc), 'threshold': 0.15}
            rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            constraints['C5_recall'] = bool(rec >= 0.55)
            details['C5'] = {'recall': float(rec), 'threshold': 0.55}
            prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            constraints['C6_precision'] = bool(prec >= 0.35)
            details['C6'] = {'precision': float(prec), 'threshold': 0.35}
            pos_rate = float(y_pred.mean())
            constraints['C7_positive_rate'] = bool(0.15 <= pos_rate <= 0.45)
            details['C7'] = {'positive_rate': pos_rate, 'bounds': [0.15, 0.45]}
    else:
        for c in ['C3_f1', 'C4_mcc', 'C5_recall', 'C6_precision', 'C7_positive_rate']:
            constraints[c] = False
            details[c] = 'skipped due to C1 failure'
    # C8: feature importance overlap
    if not os.path.exists(imp_path):
        constraints['C8_feature_importance_overlap'] = False
        details['C8'] = 'feature_importance.csv missing'
    else:
        try:
            imp_df = pd.read_csv(imp_path)
            if 'feature' not in imp_df.columns or 'importance' not in imp_df.columns:
                constraints['C8_feature_importance_overlap'] = False
                details['C8'] = 'missing feature or importance column'
            elif len(imp_df) < 20:
                constraints['C8_feature_importance_overlap'] = False
                details['C8'] = f'only {len(imp_df)} rows, need at least 20'
            else:
                imp_df = imp_df.sort_values('importance', ascending=False)
                top20 = imp_df.head(20)['feature'].tolist()
                overlap = set(top20).intersection(set(causal_features))
                constraints['C8_feature_importance_overlap'] = bool(len(overlap) >= 4)
                details['C8'] = {'overlap_count': len(overlap), 'threshold': 4, 'top20': top20, 'causal_features': causal_features}
        except Exception as e:
            constraints['C8_feature_importance_overlap'] = False
            details['C8'] = f'parse error: {str(e)}'
    # C9: no metadata in top 20
    if os.path.exists(imp_path):
        try:
            imp_df = pd.read_csv(imp_path)
            if 'feature' in imp_df.columns:
                imp_df = imp_df.sort_values('importance', ascending=False)
                top20 = imp_df.head(20)['feature'].tolist()
                has_meta = 'batch_id' in top20 or 'passage_number' in top20
                constraints['C9_no_metadata'] = bool(not has_meta)
                details['C9'] = {'metadata_in_top20': has_meta, 'top20': top20}
            else:
                constraints['C9_no_metadata'] = False
                details['C9'] = 'missing feature column'
        except Exception as e:
            constraints['C9_no_metadata'] = False
            details['C9'] = str(e)
    else:
        constraints['C9_no_metadata'] = False
        details['C9'] = 'feature_importance.csv missing'
    total = len(constraints)
    passed = sum(1 for v in constraints.values() if v)
    score = passed / total if total > 0 else 0.0
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': score, 'reward': score}, f)
    report = {'constraints': constraints, 'details': details, 'score': score, 'passed': passed, 'total': total}
    with open('/logs/verifier/report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
