#!/bin/bash
set -euo pipefail
cat > /app/solve.py <<'EOF'
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import os
import json
from sklearn.metrics import roc_auc_score

ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

def get_bin(d):
    if d <= 1: return 0
    elif d == 2: return 1
    elif d <= 4: return 2
    elif d <= 6: return 3
    elif d <= 9: return 4
    elif d <= 13: return 5
    elif d <= 18: return 6
    elif d <= 24: return 7
    elif d <= 30: return 8
    else: return 9

def compute_ddr(sequences):
    n = len(sequences)
    features = np.zeros((n, 200))
    for idx, seq in enumerate(sequences):
        for aa_idx, aa in enumerate(ALPHABET):
            positions = [i for i, c in enumerate(seq) if c == aa]
            hist = np.zeros(10)
            total_pairs = 0
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    d = positions[j] - positions[i]
                    hist[get_bin(d)] += 1
                    total_pairs += 1
            if total_pairs > 0:
                hist /= total_pairs
            features[idx, aa_idx*10:(aa_idx+1)*10] = hist
    return features

train = pd.read_csv('/app/data/train.csv')
val = pd.read_csv('/app/data/val.csv')
test = pd.read_csv('/app/data/test.csv')

all_train = pd.concat([train, val], ignore_index=True)
X_train = compute_ddr(all_train['sequence'].tolist())
y_train = all_train['localization'].tolist()
X_test = compute_ddr(test['sequence'].tolist())

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    loss_function='MultiClass',
    verbose=0
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)
classes = model.classes_
class_order = ['Cytoplasm', 'Nucleus', 'Mitochondria', 'Endo_lysosome', 'Others']
pred_df = pd.DataFrame({'seq_id': test['seq_id']})
for cls in class_order:
    idx = list(classes).index(cls)
    pred_df[cls] = probs[:, idx]
pred_df.to_csv('/app/predictions.csv', index=False)

ddr_cols = [f"DDR_{aa}_{b}" for aa in ALPHABET for b in range(10)]
ddr_df = pd.DataFrame(X_test, columns=ddr_cols)
ddr_df.insert(0, 'seq_id', test['seq_id'])
ddr_df.to_csv('/app/test_ddr_features.csv', index=False)

X_val = compute_ddr(val['sequence'].tolist())
val_probs = model.predict_proba(X_val)
val_metrics = {}
for cls in class_order:
    idx = list(classes).index(cls)
    y_true = (val['localization'] == cls).astype(int)
    val_metrics[cls] = roc_auc_score(y_true, val_probs[:, idx])
with open('/app/val_metrics.json', 'w') as f:
    json.dump(val_metrics, f)

print("Solution complete.")
EOF
python /app/solve.py
