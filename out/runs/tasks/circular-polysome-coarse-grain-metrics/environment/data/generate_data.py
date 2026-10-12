import csv
import math
import os
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
T = 256
NS = [1000, 2000, 3000, 4000, 4969]

def ar_series(phi, mean, std, rng):
    x = np.empty(T)
    sig = math.sqrt(max(0.0, 1.0 - phi * phi))
    for i in range(T):
        if i == 0:
            x[i] = rng.normal()
        else:
            x[i] = phi * x[i - 1] + sig * rng.normal()
    x = (x - x.mean()) / x.std() * std + mean
    return x

def main():
    rng = np.random.default_rng(20260831)
    circ = ar_series(0.05, 90.0, 6.12, rng)
    lin = ar_series(math.exp(-1.0 / 35.0), 100.0, 17.2, rng)
    with open(os.path.join(DATA_DIR, 'chain_scaling.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['topology', 'N', 'Rg_mean'])
        for topology, nu, c in [('circular', 0.73, 0.5), ('linear', 0.62, 0.5)]:
            for n in NS:
                w.writerow([topology, n, f'{c * float(n) ** nu:.6f}'])
    with open(os.path.join(DATA_DIR, 'rg_timeseries.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['topology', 't', 'Rg'])
        for topology, series in [('circular', circ), ('linear', lin)]:
            for t, value in enumerate(series):
                w.writerow([topology, t, f'{value:.6f}'])

if __name__ == '__main__':
    main()
