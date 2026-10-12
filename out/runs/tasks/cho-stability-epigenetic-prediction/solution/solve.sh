#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python << 'PYEOF'
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

train = pd.read_csv('/app/data/train.csv')
test = pd.read_csv('/app/data/test_features.csv')

metadata_cols = ['batch_id', 'passage_number']
feature_cols = [c for c in train.columns if c not in ['sample_id', 'stable'] + metadata_cols]

X_train = train[feature_cols].values
y_train = train['stable'].values
X_test = test[feature_cols].values

imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf_full = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1)
clf_full.fit(X_train, y_train)

y_pred = clf_full.predict(X_test)
y_prob = clf_full.predict_proba(X_test)[:, 1]

pred_df = pd.DataFrame({'sample_id': test['sample_id'], 'predicted_label': y_pred, 'probability_stable': y_prob})
pred_df.to_csv('/app/output/predictions.csv', index=False)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1)
clf.fit(X_tr, y_tr)
result = permutation_importance(clf, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)
importance_df = pd.DataFrame({'feature': feature_cols, 'importance': result.importances_mean}).sort_values('importance', ascending=False)
importance_df.to_csv('/app/output/feature_importance.csv', index=False)
print('Solution completed.')
PYEOF
