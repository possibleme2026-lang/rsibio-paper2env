import numpy as np
import pandas as pd
import os

def main():
    np.random.seed(2026)
    n = 240
    G = 100
    xs = []
    ys = []
    for i in range(20):
        for j in range(12):
            xs.append(i + 0.1 * np.random.randn())
            ys.append(j + 0.1 * np.random.randn())
    S = np.column_stack([xs, ys])
    def rbf_kernel(S, l):
        D2 = ((S[:, None, :] - S[None, :, :])**2).sum(axis=2)
        return np.exp(-D2 / (2 * l**2))
    F = []
    for l in [2.0, 4.0, 8.0]:
        K = rbf_kernel(S, l) + 1e-6 * np.eye(n)
        F.append(np.random.multivariate_normal(np.zeros(n), K))
    F = np.column_stack(F)
    X = np.zeros((n, G))
    for g in range(G):
        beta = np.random.randn(3) * 0.5
        X[:, g] = F @ beta + 0.5 * np.random.randn(n)
    y = (2.5 * np.tanh(2 * X[:, 6]) +
         2.0 * (X[:, 22]**2 - 1) +
         1.8 * np.sin(3 * X[:, 40]) +
         1.5 * np.maximum(X[:, 57], 0) +
         1.2 * (X[:, 71]**3) +
         1.0 * np.cos(2 * X[:, 88]) +
         1.0 * F[:, 0] + 0.5 * F[:, 1] +
         0.3 * np.random.randn(n))
    y = (y - y.mean()) / y.std()
    spots = pd.DataFrame({
        'spot_id': [f'S{i:03d}' for i in range(n)],
        'x': S[:, 0],
        'y': S[:, 1],
        'phenotype': y
    })
    expr = pd.DataFrame(X, columns=[f'G{g:03d}' for g in range(1, G+1)])
    expr.insert(0, 'spot_id', spots['spot_id'])
    os.makedirs('/app/data', exist_ok=True)
    spots.to_csv('/app/data/spots.csv', index=False)
    expr.to_csv('/app/data/expression.csv', index=False)

if __name__ == '__main__':
    main()
