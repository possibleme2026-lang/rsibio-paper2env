#!/bin/bash
python - <<'PY'
import pandas as pd
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('/app/data/train.csv')
test = pd.read_csv('/app/data/test.csv')
features = [c for c in train.columns if c not in ('sample_id', 'label')]
X = train[features]
y = train['label']
clf = LogisticRegression(max_iter=1000).fit(X, y)
p = clf.predict_proba(test[features])[:, 1]
out = pd.DataFrame({'sample_id': test['sample_id'], 'prob_neutron': p})
out.to_csv('/app/predictions.csv', index=False)
PY
