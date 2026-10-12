#!/bin/bash
set -euo pipefail
mkdir -p /app/output
cat > /app/solve.py <<'PYEOF'
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.spatial import ConvexHull

def gc(seq):
    return sum(1 for b in seq.upper() if b in 'GC') / float(len(seq))

def featurize(df):
    return np.column_stack([
        np.ones(len(df)),
        df['rtt_seq'].map(gc).values,
        df['pbs_seq'].map(gc).values,
        df['spacer_seq'].map(gc).values,
        df['rtt_len'].values.astype(float),
        df['edit_size'].values.astype(float),
        (df['edit_type'] == 'del').astype(float).values * df['edit_size'].values.astype(float),
        (df['edit_type'] == 'sub').astype(float).values,
        (df['edit_type'] == 'ins').astype(float).values,
        (df['pam_type'] == 'NGG').astype(float).values,
        np.abs(df['nick_offset'].values.astype(float)),
    ])

def dirichlet_mn_nll(counts, alpha):
    n = counts.sum(axis=1)
    logp = (gammaln(n + 1) - gammaln(counts + 1).sum(axis=1)
            + gammaln(n + alpha.sum(axis=1)) - gammaln(alpha.sum(axis=1))
            + (gammaln(counts + alpha) - gammaln(alpha)).sum(axis=1))
    return -float(np.sum(logp))

def fit_model(X, counts, l2=1e-4):
    K = X.shape[1]
    def obj(theta):
        B = theta.reshape(K, 3)
        eta = np.clip(X @ B, -15.0, 15.0)
        alpha = np.exp(eta)
        return dirichlet_mn_nll(counts, alpha) + l2 * float(np.sum(theta ** 2))
    res = minimize(obj, np.zeros(K * 3), method='L-BFGS-B', options={'maxiter': 800, 'ftol': 1e-8})
    return res.x.reshape(K, 3)

def dirichlet_density_logpdf(y, alpha):
    a = np.atleast_2d(alpha)
    if a.shape[1] != 3:
        a = a.reshape(-1, 3)
    return (gammaln(a.sum()) - gammaln(a).sum()
            + ((a - 1.0) * np.log(np.clip(y, 1e-12, None))).sum(axis=1))

train = pd.read_csv('/app/data/train_library1.csv')
eval_df = pd.read_csv('/app/data/eval_library2_features.csv')
X = featurize(train)
counts = train[['n_edited', 'n_unedited', 'n_indel']].values.astype(float)
obs = counts / counts.sum(axis=1, keepdims=True)
Xeval = featurize(eval_df)

n = len(train)
rng = np.random.default_rng(12345)
perm = rng.permutation(n)
ncal = 300
cal_idx = perm[:ncal]
fit_idx = perm[ncal:]
B = fit_model(X[fit_idx], counts[fit_idx])
alpha_cal = np.exp(np.clip(X[cal_idx] @ B, -15.0, 15.0))
alpha_eval = np.exp(np.clip(Xeval @ B, -15.0, 15.0))

cal_scores = -dirichlet_density_logpdf(obs[cal_idx], alpha_cal)
k = int(np.ceil((ncal + 1) * 0.90))
k = min(k, ncal)
q_hat = float(np.sort(cal_scores)[k - 1])

pred = alpha_eval / alpha_eval.sum(axis=1, keepdims=True)
pred_df = pd.DataFrame({'peg_id': eval_df['peg_id'], 'p_edited': pred[:, 0], 'p_unedited': pred[:, 1], 'p_indel': pred[:, 2]})
pred_df.to_csv('/app/output/predictions.csv', index=False)

step = 0.01
pts = []
for a in np.arange(step, 1.0, step):
    for b in np.arange(step, 1.0 - a + 1e-12, step):
        c = 1.0 - a - b
        if c >= step - 1e-12:
            pts.append([a, b, c])
G = np.asarray(pts, dtype=float)

regions = []
for i in range(len(eval_df)):
    lp = dirichlet_density_logpdf(G, alpha_eval[i])
    keep = G[lp >= -q_hat]
    if len(keep) < 3:
        mu = pred[i]
        keep = np.array([np.clip(mu + d, 1e-6, None) for d in [np.array([0.02, -0.01, -0.01]), np.array([-0.01, 0.02, -0.01]), np.array([-0.01, -0.01, 0.02])]])
        keep = keep / keep.sum(axis=1, keepdims=True)
    hull = ConvexHull(keep[:, :2])
    verts = keep[hull.vertices]
    regions.append({'peg_id': str(eval_df['peg_id'].iloc[i]), 'region': verts.tolist()})

with open('/app/output/regions.jsonl', 'w') as fh:
    for obj in regions:
        fh.write(json.dumps(obj) + '\n')

with open('/app/output/calibration.json', 'w') as fh:
    json.dump({'nominal_coverage': 0.90, 'calib_size': int(ncal), 'q_hat': q_hat}, fh)
PYEOF
python /app/solve.py
