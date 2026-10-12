#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PYEOF'
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ive
from scipy.stats import f_oneway, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

DATA, OUT = '/app/data', '/app/output'
mb = pd.read_csv(f'{DATA}/microbiome_abundance.csv', index_col=0)
mt = pd.read_csv(f'{DATA}/host_metabolomics.csv', index_col=0)
tx = pd.read_csv(f'{DATA}/host_transcriptomics.csv', index_col=0)

# --- shared latent -------------------------------------------------------
p = np.asarray(mb.values, dtype=float) + 1e-6
p = p / p.sum(axis=1, keepdims=True)
clr = np.log(p) - np.log(p).mean(axis=1, keepdims=True)
host_raw = np.hstack([mt.values, tx.values]).astype(float)
host = StandardScaler().fit_transform(host_raw)
feat = np.hstack([clr, host])
feat_names = list(mb.columns) + list(mt.columns) + list(tx.columns)
Z = PCA(n_components=8, svd_solver='full', random_state=0).fit_transform(
    StandardScaler().fit_transform(feat))
n, d = Z.shape
Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)

# --- vMF mixture EM ------------------------------------------------------
def log_const(dim, k):
    return (dim / 2 - 1) * np.log(k) - (dim / 2) * np.log(2 * np.pi) - (np.log(ive(dim / 2 - 1, k)) + k)

def kappa_from_rbar(r, dim):
    v = dim / 2 - 1
    r = min(max(r, 1e-6), 1 - 1e-6)
    fn = lambda k: ive(v + 1, k) / ive(v, k) - r
    return brentq(fn, 1e-4, 1e4, xtol=1e-8)

def vmf_em(X, K, seed=0, n_iter=200):
    n, dim = X.shape
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(X)
    mu = np.zeros((K, dim)); kap = np.full(K, 25.0)
    for c in range(K):
        m = X[km.labels_ == c].mean(0)
        mu[c] = m / (np.linalg.norm(m) + 1e-12)
    prev = -np.inf
    ll = prev
    for _ in range(n_iter):
        lp = np.empty((n, K))
        for c in range(K):
            lp[:, c] = kap[c] * (X @ mu[c]) + log_const(dim, kap[c]) - np.log(K)
        mx = lp.max(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(1))
        ll = float(lse.sum())
        R = np.exp(lp - lse[:, None])
        for c in range(K):
            s = R[:, c].sum()
            if s < 1e-10:
                continue
            ms = (R[:, c][:, None] * X).sum(0)
            nr = np.linalg.norm(ms)
            if nr < 1e-12:
                continue
            mu[c] = ms / nr
            kap[c] = kappa_from_rbar(nr / s, dim)
        if abs(ll - prev) < 1e-6 * max(1.0, abs(ll)):
            break
        prev = ll
    return mu, kap, R, ll

best = None
for K in range(1, 11):
    mu, kap, R, ll = vmf_em(Zn, K, seed=0)
    n_par = K * (d + 1) + (K - 1)
    bic = -2 * ll + n_par * np.log(n)
    if best is None or bic < best[0]:
        best = (bic, K, R)
bic_vmf, k_vmf, Rv = best
labels_vmf = Rv.argmax(1)
entropy = -(np.clip(Rv, 1e-12, 1) * np.log(np.clip(Rv, 1e-12, 1))).sum(1)

# --- Euclidean control ---------------------------------------------------
bestg = None
for K in range(1, 11):
    gm = GaussianMixture(n_components=K, covariance_type='full', n_init=3, random_state=0).fit(Z)
    if bestg is None or gm.bic(Z) < bestg[0]:
        bestg = (gm.bic(Z), K, gm.predict(Z))
k_gmm, labels_gmm = bestg[1], bestg[2]

# --- spike-and-slab style feature selection ------------------------------
ps = []
for j in range(feat.shape[1]):
    groups = [feat[labels_vmf == c, j] for c in range(k_vmf)]
    groups = [g for g in groups if g.size > 1]
    ps.append(float(f_oneway(*groups).pvalue) if len(groups) > 1 else 1.0)
qvals = multipletests(ps, method='fdr_bh')[1]
pip = np.clip(1.0 - qvals, 0.0, 1.0)

# --- conditional co-occurrence ------------------------------------------
rho = np.abs(spearmanr(clr, host).statistic)
rho = np.clip(rho[:clr.shape[1], clr.shape[1]:], 0.0, 1.0)

json.dump({
    'k_vmf': int(k_vmf),
    'k_gmm': int(k_gmm),
    'k_selection_criterion': 'BIC = -2 logL + (K*(d+1)+(K-1)) log n on the vMF mixture',
    'cluster_labels_vmf': [int(x) for x in labels_vmf],
    'cluster_labels_gmm': [int(x) for x in labels_gmm],
    'posterior_probs': [[float(x) for x in row] for row in Rv],
    'posterior_entropy': [float(x) for x in entropy],
}, open(f'{OUT}/results.json', 'w'), indent=1)

pd.DataFrame({'feature_id': feat_names, 'pip': pip}).to_csv(
    f'{OUT}/selected_features.csv', index=False)

rows = []
for a, tname in enumerate(mb.columns):
    for b, hname in enumerate(list(mt.columns) + list(tx.columns)):
        rows.append((tname, hname, float(rho[a, b])))
pd.DataFrame(rows, columns=['taxon_id', 'host_feature', 'probability']).to_csv(
    f'{OUT}/cooccurrence.csv', index=False)
print('k_vmf =', k_vmf, 'k_gmm =', k_gmm)
PYEOF
