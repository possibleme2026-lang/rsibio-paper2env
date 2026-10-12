import numpy as np
import pandas as pd
import os

def main():
    seed = 42
    rng = np.random.default_rng(seed)
    data_dir = os.path.dirname(os.path.abspath(__file__))
    T = 3
    G = 20
    genes = [f'gene_{i}' for i in range(G)]
    units = ['Q1','Q2','Q3','Q4']
    N_per_unit = {0: 50, 1: 55, 2: 60}
    K = 5
    L = rng.normal(0, 1, size=(G, K))
    mask = rng.random((G, K)) < 0.3
    L = L * mask
    spatial_effects = rng.normal(0, 1, size=(len(units), K))
    temporal_trend = np.array([1.0, 1.5, 2.0])
    records = []
    coords_records = []
    for t in range(T):
        for ui, u in enumerate(units):
            n = N_per_unit[t]
            if u == 'Q1':
                x = rng.uniform(0, 5, n)
                y = rng.uniform(0, 5, n)
            elif u == 'Q2':
                x = rng.uniform(5, 10, n)
                y = rng.uniform(0, 5, n)
            elif u == 'Q3':
                x = rng.uniform(0, 5, n)
                y = rng.uniform(5, 10, n)
            else:
                x = rng.uniform(5, 10, n)
                y = rng.uniform(5, 10, n)
            F = rng.normal(0, 1, size=(n, K))
            F += spatial_effects[ui]
            F *= temporal_trend[t]
            F += rng.normal(0, 0.5, size=(n, K))
            signal = F @ L.T
            noise = rng.normal(0, 0.5, size=(n, G))
            expr = np.log1p(np.exp(signal + noise))
            for i in range(n):
                spot_id = f't{t}_{u}_{i}'
                coords_records.append({'timepoint': t, 'spot_id': spot_id, 'x': x[i], 'y': y[i], 'spatial_unit': u})
                row = {'timepoint': t, 'spot_id': spot_id}
                for gi, g in enumerate(genes):
                    row[g] = expr[i, gi]
                records.append(row)
    expr_df = pd.DataFrame(records)
    coords_df = pd.DataFrame(coords_records)
    expr_df.to_csv(os.path.join(data_dir, 'expression.csv'), index=False)
    coords_df.to_csv(os.path.join(data_dir, 'coordinates.csv'), index=False)
    with open(os.path.join(data_dir, 'genes.txt'), 'w') as f:
        for g in genes:
            f.write(g + '\n')

if __name__ == '__main__':
    main()
