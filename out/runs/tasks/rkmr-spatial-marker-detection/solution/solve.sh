#!/bin/bash
set -euo pipefail
cat > /app/solve.py << 'EOF'
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import os

def standardize(v):
    v = np.asarray(v, dtype=float)
    mu = np.mean(v)
    sd = np.std(v, ddof=0)
    if sd < 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sd

def main():
    data_dir = '/app/data'
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    spots = pd.read_csv(f'{data_dir}/spots.csv')
    expr = pd.read_csv(f'{data_dir}/expression.csv')
    expr = expr.set_index('spot_id').loc[spots['spot_id']].reset_index()
    gene_names = list(expr.columns[1:])
    X = expr.drop(columns=['spot_id']).values.astype(float)
    y = spots['phenotype'].values.astype(float)
    S = spots[['x', 'y']].values.astype(float)
    n, G = X.shape
    Xs = np.zeros_like(X)
    for g in range(G):
        Xs[:, g] = standardize(X[:, g])
    ys = standardize(y)
    D2 = ((S[:, None, :] - S[None, :, :])**2).sum(axis=2)
    med_dist = np.median(np.sqrt(D2))
    l_s = med_dist if med_dist > 1e-12 else 1.0
    m = min(80, n)
    kmeans = KMeans(n_clusters=m, random_state=0, n_init=10).fit(S)
    L = kmeans.cluster_centers_
    D2_mm = ((L[:, None, :] - L[None, :, :])**2).sum(axis=2)
    K_mm = np.exp(-D2_mm / (2 * l_s**2))
    K_mm += 1e-6 * np.eye(m)
    D2_nm = ((S[:, None, :] - L[None, :, :])**2).sum(axis=2)
    K_nm = np.exp(-D2_nm / (2 * l_s**2))
    L_chol = np.linalg.cholesky(K_mm)
    Z_s = K_nm @ np.linalg.inv(L_chol.T)
    def residual(v):
        v = np.asarray(v, dtype=float)
        A = Z_s.T @ Z_s + 1.0 * np.eye(Z_s.shape[1])
        alpha = np.linalg.solve(A, Z_s.T @ v)
        return v - Z_s @ alpha
    r_y = residual(ys)
    scores = np.zeros(G)
    for g in range(G):
        r_g = residual(Xs[:, g])
        D2_g = (r_g[:, None] - r_g[None, :])**2
        mad = np.median(np.abs(r_g - np.median(r_g)))
        l_g = 1.4826 * mad
        if l_g < 1e-6:
            l_g = 1e-6
        K_g = np.exp(-D2_g / (2 * l_g**2))
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        preds = np.zeros(n)
        for train_idx, test_idx in kf.split(r_g):
            K_train = K_g[np.ix_(train_idx, train_idx)]
            K_test = K_g[np.ix_(test_idx, train_idx)]
            y_train = r_y[train_idx]
            A = K_train + 1.0 * np.eye(len(train_idx))
            alpha = np.linalg.solve(A, y_train)
            preds[test_idx] = K_test @ alpha
        ss_res = np.sum((r_y - preds)**2)
        ss_tot = np.sum((r_y - np.mean(r_y))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        scores[g] = max(0.0, r2)
    pips = scores / (scores + 0.05)
    pips = np.clip(pips, 0.0, 0.999)
    order = np.lexsort((np.array(gene_names), -pips))
    sorted_genes = [gene_names[i] for i in order]
    sorted_pips = pips[order]
    ranked = pd.DataFrame({'gene': sorted_genes, 'pip': sorted_pips})
    ranked.to_csv(f'{out_dir}/ranked_pips.csv', index=False)
    panel = [g for g, p in zip(sorted_genes, sorted_pips) if p >= 0.5]
    if len(panel) < 3:
        panel = sorted_genes[:3]
    if len(panel) > 15:
        panel = panel[:15]
    with open(f'{out_dir}/panel.txt', 'w') as f:
        for g in panel:
            f.write(g + '\n')

if __name__ == '__main__':
    main()
EOF
python /app/solve.py
