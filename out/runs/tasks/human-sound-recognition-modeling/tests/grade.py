import os
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add tests directory to path to import gold_gen
sys.path.append('/tests')
import gold_gen

def check_predictions_format(pred_path):
    """Check that predictions.csv exists and has the correct columns."""
    if not os.path.exists(pred_path):
        return False, "predictions.csv not found", None
    try:
        df = pd.read_csv(pred_path)
    except Exception as e:
        return False, f"Error reading predictions.csv: {e}", None
    expected_cols = {'clip_id', 'predicted_accuracy'}
    if set(df.columns) != expected_cols:
        return False, f"Columns are {list(df.columns)}, expected {expected_cols}", df
    return True, "OK", df

def check_row_count_and_ids(pred_df, test_df):
    """Check row count and clip_id order."""
    if pred_df is None or 'clip_id' not in pred_df.columns:
        return False, "Missing predictions or clip_id column"
    if len(pred_df) != len(test_df):
        return False, f"Expected {len(test_df)} rows, got {len(pred_df)}"
    if not (pred_df['clip_id'].values == test_df['clip_id'].values).all():
        return False, "clip_ids do not match exactly in order"
    return True, "OK"

def check_range_no_nan(pred_df):
    """Check that predicted_accuracy is in [0,1] and no NaN."""
    if pred_df is None or 'predicted_accuracy' not in pred_df.columns:
        return False, "Missing predicted_accuracy column"
    preds = pred_df['predicted_accuracy'].values
    if np.any(np.isnan(preds)):
        return False, "Contains NaN"
    if np.any(preds < 0) or np.any(preds > 1):
        return False, "Values outside [0,1]"
    return True, "OK"

def check_model_summary(summary_path):
    """Check model_summary.json exists and has required keys."""
    if not os.path.exists(summary_path):
        return False, "model_summary.json not found"
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except Exception as e:
        return False, f"Error reading JSON: {e}"
    if 'validation_rmse' not in summary:
        return False, "Missing key 'validation_rmse'"
    if not isinstance(summary['validation_rmse'], (int, float)):
        return False, "'validation_rmse' is not a number"
    if 'model_type' not in summary:
        return False, "Missing key 'model_type'"
    if not isinstance(summary['model_type'], str):
        return False, "'model_type' is not a string"
    return True, "OK"

def main():
    # Paths
    test_path = '/app/data/test.csv'
    pred_path = '/app/output/predictions.csv'
    summary_path = '/app/output/model_summary.json'
    reward_dir = '/logs/verifier'
    os.makedirs(reward_dir, exist_ok=True)
    
    # Load test data
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
    test_df = pd.read_csv(test_path)
    n_test = len(test_df)
    
    # Load gold
    gold_df = gold_gen.get_test_gold()
    gold = gold_df.set_index('clip_id')['human_accuracy']
    gold = gold.loc[test_df['clip_id']].values
    
    # Initialize constraints dictionary
    constraints = {}
    details = {}
    
    # 1. predictions.csv format
    ok, msg, pred_df = check_predictions_format(pred_path)
    constraints['pred_file_format'] = ok
    details['pred_file_format'] = msg
    
    # 2. row count and clip_id order
    ok, msg = check_row_count_and_ids(pred_df, test_df)
    constraints['row_count_and_ids'] = ok
    details['row_count_and_ids'] = msg
    
    # 3. range and no NaN
    ok, msg = check_range_no_nan(pred_df)
    constraints['range_no_nan'] = ok
    details['range_no_nan'] = msg
    
    # 4. model_summary.json
    ok, msg = check_model_summary(summary_path)
    constraints['model_summary'] = ok
    details['model_summary'] = msg
    
    # If pred_df is valid, compute metrics
    preds = None
    if pred_df is not None and 'predicted_accuracy' in pred_df.columns and len(pred_df) == n_test:
        preds = pred_df['predicted_accuracy'].values
    
    if preds is None:
        for c in ['pearson_0.60', 'pearson_0.75', 'rmse_0.15', 'rmse_0.12', 'mae_0.10', 'std_0.02']:
            constraints[c] = False
            details[c] = 'Cannot compute: invalid predictions'
    else:
        corr, _ = pearsonr(gold, preds)
        rmse = np.sqrt(mean_squared_error(gold, preds))
        mae = mean_absolute_error(gold, preds)
        std_pred = np.std(preds)
        
        constraints['pearson_0.60'] = bool(corr >= 0.60)
        details['pearson_0.60'] = f'Pearson r = {corr:.4f} (threshold 0.60)'
        constraints['pearson_0.75'] = bool(corr >= 0.75)
        details['pearson_0.75'] = f'Pearson r = {corr:.4f} (threshold 0.75)'
        constraints['rmse_0.15'] = bool(rmse <= 0.15)
        details['rmse_0.15'] = f'RMSE = {rmse:.4f} (threshold 0.15)'
        constraints['rmse_0.12'] = bool(rmse <= 0.12)
        details['rmse_0.12'] = f'RMSE = {rmse:.4f} (threshold 0.12)'
        constraints['mae_0.10'] = bool(mae <= 0.10)
        details['mae_0.10'] = f'MAE = {mae:.4f} (threshold 0.10)'
        constraints['std_0.02'] = bool(std_pred > 0.02)
        details['std_0.02'] = f'Std dev of predictions = {std_pred:.4f} (threshold > 0.02)'
    
    # Compute score
    total = len(constraints)
    passed = sum(1 for v in constraints.values() if v)
    score = passed / total if total > 0 else 0.0
    
    # Write reward
    reward = {'score': score, 'reward': score, 'details': details}
    with open(os.path.join(reward_dir, 'reward.json'), 'w') as f:
        json.dump(reward, f, indent=2)
    
    print(f"Score: {score:.4f} ({passed}/{total} constraints passed)")
    for k, v in details.items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    main()
