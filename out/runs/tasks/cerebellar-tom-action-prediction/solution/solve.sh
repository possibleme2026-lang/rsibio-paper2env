#!/bin/bash
set -euo pipefail
python - <<'PY'
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

gmv = pd.read_csv('/app/data/cerebellar_gmv.csv')
beh = pd.read_csv('/app/data/behavior.csv')
net = pd.read_csv('/app/data/cortical_network_scores.csv')
df = gmv.merge(beh, on='subject_id').merge(net, on='subject_id')
regions = [c for c in gmv.columns if c != 'subject_id']

def fit_region_models(df, regions, behavior):
    results = []
    for reg in regions:
        X = df[[reg, 'age_months', 'sex', 'site', 'tiv_mm3']].copy()
        X = pd.get_dummies(X, columns=['sex', 'site'], drop_first=True)
        X = sm.add_constant(X)
        y = df[behavior]
        model = sm.OLS(y, X).fit()
        results.append({'region': reg, 'beta': model.params[reg], 't': model.tvalues[reg], 'p': model.pvalues[reg]})
    res_df = pd.DataFrame(results)
    _, q = fdrcorrection(res_df['p'].values, alpha=0.05)
    res_df['q'] = q
    res_df = res_df.sort_values('p')
    return res_df

tom_res = fit_region_models(df, regions, 'explicit_tom_score')
action_res = fit_region_models(df, regions, 'action_prediction_accuracy')
tom_sig = tom_res[tom_res['q'] < 0.05]['region'].tolist()
action_sig = action_res[action_res['q'] < 0.05]['region'].tolist()
intersection = len(set(tom_sig) & set(action_sig))

def cv_r2(df, regions, behavior):
    X = df[regions].copy()
    X = (X - X.mean()) / X.std()
    y = df[behavior].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s = []
    for train_idx, test_idx in kf.split(X):
        pls = PLSRegression(n_components=min(2, len(regions)))
        pls.fit(X.iloc[train_idx], y[train_idx])
        y_pred = pls.predict(X.iloc[test_idx]).flatten()
        r2s.append(r2_score(y[test_idx], y_pred))
    return float(np.mean(r2s))

tom_cv_r2 = cv_r2(df, tom_sig, 'explicit_tom_score')
action_cv_r2 = cv_r2(df, action_sig, 'action_prediction_accuracy')

def network_corr(df, regions, network_col):
    X = df[regions].copy()
    X = (X - X.mean()) / X.std()
    mean_gmv = X.mean(axis=1)
    r, p = pearsonr(mean_gmv, df[network_col])
    return float(r), float(p)

tom_r, tom_p = network_corr(df, tom_sig, 'DefaultMode')
action_r, action_p = network_corr(df, action_sig, 'VentralAttention')

out = {
    'tom_regions': tom_sig,
    'action_regions': action_sig,
    'dissociation_intersection': intersection,
    'tom_cv_r2': tom_cv_r2,
    'action_cv_r2': action_cv_r2,
    'tom_network_correlation': {'network': 'DefaultMode', 'r': tom_r, 'p': tom_p},
    'action_network_correlation': {'network': 'VentralAttention', 'r': action_r, 'p': action_p},
    'tom_fdr_table': tom_res[['region', 'beta', 't', 'p', 'q']].to_dict(orient='records'),
    'action_fdr_table': action_res[['region', 'beta', 't', 'p', 'q']].to_dict(orient='records')
}
with open('/app/results.json', 'w') as f:
    json.dump(out, f, indent=2)
PY
