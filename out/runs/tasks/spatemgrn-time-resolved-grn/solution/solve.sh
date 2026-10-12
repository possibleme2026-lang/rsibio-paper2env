#!/bin/bash
set -euo pipefail
mkdir -p /app/output
cat > /app/solve.py << 'EOF'
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso

def main():
    data_dir = '/app/data'
    expr = pd.read_csv(f'{data_dir}/expression.csv')
    coords = pd.read_csv(f'{data_dir}/coordinates.csv')
    with open(f'{data_dir}/genes.txt') as f:
        genes = [line.strip() for line in f if line.strip()]
    df = expr.merge(coords, on=['timepoint', 'spot_id'])
    results = []
    for t in [0,1,2]:
        for u in ['Q1','Q2','Q3','Q4']:
            sub = df[(df['timepoint']==t) & (df['spatial_unit']==u)]
            X = sub[genes].values
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std==0] = 1.0
            X = (X - mean) / std
            xy = sub[['x','y']].values
            N = X.shape[0]
            X_smooth = np.zeros_like(X)
            for i in range(N):
                dist = np.sqrt(((xy - xy[i])**2).sum(axis=1))
                idx = np.argsort(dist)[:6]
                X_smooth[i] = X[idx].mean(axis=0)
            if t > 0:
                sub_prev = df[(df['timepoint']==t-1) & (df['spatial_unit']==u)]
                X_prev = sub_prev[genes].values
                mean_p = X_prev.mean(axis=0)
                std_p = X_prev.std(axis=0)
                std_p[std_p==0] = 1.0
                X_prev = (X_prev - mean_p) / std_p
                xy_prev = sub_prev[['x','y']].values
                N_prev = X_prev.shape[0]
                X_prev_smooth = np.zeros_like(X_prev)
                for i in range(N_prev):
                    dist = np.sqrt(((xy_prev - xy_prev[i])**2).sum(axis=1))
                    idx = np.argsort(dist)[:6]
                    X_prev_smooth[i] = X_prev[idx].mean(axis=0)
                if N_prev != N:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(N_prev, size=N, replace=True)
                    X_prev_smooth = X_prev_smooth[idx]
                A = np.vstack([X_smooth, X_prev_smooth])
            else:
                A = X_smooth
            model = GraphicalLasso(alpha=0.1, max_iter=100, tol=1e-4)
            model.fit(A)
            Theta = model.precision_
            scores = []
            pairs = []
            for i in range(len(genes)):
                for j in range(i+1, len(genes)):
                    rho = -Theta[i,j] / np.sqrt(Theta[i,i] * Theta[j,j])
                    scores.append(abs(rho))
                    pairs.append((genes[i], genes[j]))
            scores = np.array(scores)
            max_score = scores.max() if scores.size > 0 else 0
            if max_score > 0:
                scores = scores / max_score
            else:
                scores = np.zeros_like(scores)
            for (g1, g2), score in zip(pairs, scores):
                if score > 0.05:
                    src, tgt = (g1, g2) if g1 < g2 else (g2, g1)
                    results.append({
                        'timepoint': t,
                        'spatial_unit': u,
                        'source': src,
                        'target': tgt,
                        'score': round(float(score), 6)
                    })
    out = pd.DataFrame(results)
    if len(out) > 0:
        out = out.sort_values(['timepoint', 'spatial_unit', 'score'], ascending=[True, True, False])
    out.to_csv('/app/output/edges.csv', index=False)

if __name__ == '__main__':
    main()
EOF
python /app/solve.py
