#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.append('/tests')
import gold_gen

def load_predictions():
    path = '/app/predictions.csv'
    if not os.path.exists(path):
        return None, "predictions.csv not found"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"Failed to read predictions.csv: {e}"
    return df, None

def load_ddr_features():
    path = '/app/test_ddr_features.csv'
    if not os.path.exists(path):
        return None, "test_ddr_features.csv not found"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"Failed to read test_ddr_features.csv: {e}"
    return df, None

def main():
    constraints = []
    # Get hidden test labels and reference AUCs
    test_df = gold_gen.get_hidden_test_labels()
    ref_aucs = gold_gen.get_reference_aucs()
    
    # Constraint 1: predictions.csv exists and has correct columns
    pred_df, err = load_predictions()
    if err:
        constraints.append((False, err))
    else:
        expected_cols = ['seq_id', 'Cytoplasm', 'Nucleus', 'Mitochondria', 'Endo_lysosome', 'Others']
        if list(pred_df.columns) != expected_cols:
            constraints.append((False, f"predictions.csv columns mismatch. Expected {expected_cols}, got {list(pred_df.columns)}"))
        else:
            constraints.append((True, "predictions.csv format OK"))
    
    # Constraint 2: all test IDs present exactly once
    if pred_df is not None:
        expected_ids = set(test_df['seq_id'])
        pred_ids = set(pred_df['seq_id'])
        if pred_ids != expected_ids:
            constraints.append((False, f"seq_id mismatch. Missing: {expected_ids - pred_ids}, Extra: {pred_ids - expected_ids}"))
        elif pred_df['seq_id'].duplicated().any():
            constraints.append((False, "Duplicate seq_id in predictions.csv"))
        else:
            constraints.append((True, "All test seq_ids present exactly once"))
    else:
        constraints.append((False, "Cannot check seq_ids: predictions.csv missing"))
    
    # Constraint 3: probabilities valid
    if pred_df is not None:
        prob_cols = ['Cytoplasm', 'Nucleus', 'Mitochondria', 'Endo_lysosome', 'Others']
        try:
            probs = pred_df[prob_cols].values.astype(float)
            if not np.all((probs >= 0) & (probs <= 1)):
                constraints.append((False, "Probabilities out of [0,1] range"))
            elif not np.allclose(probs.sum(axis=1), 1.0, atol=0.01):
                constraints.append((False, "Probabilities do not sum to 1 (tolerance 0.01)"))
            else:
                constraints.append((True, "Probabilities valid and sum to 1"))
        except Exception as e:
            constraints.append((False, f"Error parsing probabilities: {e}"))
    else:
        constraints.append((False, "Cannot check probabilities: predictions.csv missing"))
    
    # Constraint 4: DDR features file format
    ddr_df, err = load_ddr_features()
    if err:
        constraints.append((False, err))
    else:
        expected_ddr_cols = ['seq_id'] + [f"DDR_{aa}_{b}" for aa in gold_gen.ALPHABET for b in range(10)]
        if list(ddr_df.columns) != expected_ddr_cols:
            constraints.append((False, f"DDR columns mismatch. Expected {len(expected_ddr_cols)} columns, got {len(ddr_df.columns)}"))
        else:
            try:
                ddr_vals = ddr_df.iloc[:, 1:].values.astype(float)
                if not np.all(np.isfinite(ddr_vals)):
                    constraints.append((False, "DDR features contain non-finite values"))
                else:
                    constraints.append((True, "DDR feature file format OK"))
            except Exception as e:
                constraints.append((False, f"Error parsing DDR features: {e}"))
    
    # Constraint 5: DDR feature correctness for 10 random test sequences
    if ddr_df is not None and pred_df is not None:
        try:
            rng = np.random.RandomState(42)
            sample_ids = rng.choice(test_df['seq_id'].values, size=10, replace=False)
            sample_seqs = test_df.set_index('seq_id').loc[sample_ids, 'sequence'].values
            ref_ddr = gold_gen.compute_ddr(list(sample_seqs))
            agent_ddr = ddr_df.set_index('seq_id').loc[sample_ids].iloc[:, 1:].values.astype(float)
            max_diff = np.max(np.abs(ref_ddr - agent_ddr))
            if max_diff > 1e-6:
                constraints.append((False, f"DDR feature mismatch. Max diff = {max_diff}"))
            else:
                constraints.append((True, f"DDR features match reference (max diff {max_diff:.2e})"))
        except Exception as e:
            constraints.append((False, f"Error checking DDR features: {e}"))
    else:
        constraints.append((False, "Cannot check DDR features: missing files"))
    
    # Constraints 6-10: per-class AUC
    if pred_df is not None and test_df is not None:
        try:
            merged = pred_df.merge(test_df[['seq_id', 'localization']], on='seq_id', how='inner')
            if len(merged) != len(test_df):
                constraints.append((False, "Prediction merge failed to match all test sequences"))
            else:
                prob_cols = ['Cytoplasm', 'Nucleus', 'Mitochondria', 'Endo_lysosome', 'Others']
                for cls in prob_cols:
                    y_true = (merged['localization'] == cls).astype(int).values
                    y_score = merged[cls].values.astype(float)
                    if len(np.unique(y_true)) < 2:
                        constraints.append((False, f"Cannot compute AUC for {cls}: only one class present in test"))
                    else:
                        auc = roc_auc_score(y_true, y_score)
                        ref = ref_aucs.get(cls, 0.0)
                        if auc >= ref - 0.05:
                            constraints.append((True, f"{cls} AUC = {auc:.4f} (ref {ref:.4f})"))
                        else:
                            constraints.append((False, f"{cls} AUC = {auc:.4f} (ref {ref:.4f}, required >= {ref-0.05:.4f})"))
        except Exception as e:
            constraints.append((False, f"Error computing AUC: {e}"))
    else:
        for cls in ['Cytoplasm', 'Nucleus', 'Mitochondria', 'Endo_lysosome', 'Others']:
            constraints.append((False, f"Cannot compute AUC for {cls}: missing files"))
    
    # Compute score
    total = len(constraints)
    passed = sum(1 for ok, _ in constraints if ok)
    score = passed / total if total > 0 else 0.0
    reward = score
    
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({"score": score, "reward": reward}, f)
    
    for ok, msg in constraints:
        print(f"{'PASS' if ok else 'FAIL'}: {msg}")
    print(f"Score: {score:.4f}")

if __name__ == "__main__":
    main()
