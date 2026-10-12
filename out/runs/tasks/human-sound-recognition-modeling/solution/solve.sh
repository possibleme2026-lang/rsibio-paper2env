#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json

train = pd.read_csv('/app/data/train.csv')
test = pd.read_csv('/app/data/test.csv')

y = train['human_accuracy']
X = train.drop(columns=['human_accuracy', 'clip_id'])
X_test = test.drop(columns=['clip_id'])

cat_cols = ['category', 'distortion']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
val_rmse = np.sqrt(-cv_scores.mean())

model.fit(X, y)
preds = model.predict(X_test)
preds = np.clip(preds, 0.0, 1.0)

out = pd.DataFrame({'clip_id': test['clip_id'], 'predicted_accuracy': preds})
out.to_csv('/app/output/predictions.csv', index=False)

summary = {'validation_rmse': float(val_rmse), 'model_type': 'RandomForestRegressor'}
with open('/app/output/model_summary.json', 'w') as f:
    json.dump(summary, f)
PY
