#!/bin/bash
set -euo pipefail
cat > /app/solution.py << 'EOF'
import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score

df = pd.read_csv('/app/data/longitudinal_biomarkers.csv')
biomarkers = [f'biomarker_{i}' for i in range(1, 13)]
dt = 0.1
t_max = 12.0
time_grid = np.arange(0, t_max + dt/2, dt)
interp_data = {}
for pid in df['patient_id'].unique():
    pdata = df[df['patient_id'] == pid].sort_values('visit_time')
    if len(pdata) < 4:
        continue
    times = pdata['visit_time'].values
    t_min, t_max_p = times.min(), times.max()
    valid_grid = time_grid[(time_grid >= t_min) & (time_grid <= t_max_p)]
    if len(valid_grid) < 2:
        continue
    interp_data[pid] = {'times': valid_grid}
    for b in biomarkers:
        vals = pdata[b].values
        interp_vals = np.interp(valid_grid, times, vals)
        interp_data[pid][b] = interp_vals
all_samples = []
for pid, d in interp_data.items():
    for b in biomarkers:
        all_samples.append(d[b])
# Per-biomarker pooled statistics (population, ddof=0), as the
# instruction requires: pool every interpolated sample of a
# biomarker across all patients. ``all_samples`` is ordered
# patient-major / biomarker-minor, so one biomarker's samples
# sit at a constant stride of len(biomarkers). A previous fix
# padded the ragged arrays into a matrix and reduced over
# axis=0, which yields a per-TIME-COLUMN statistic (length
# max(len(series))) instead of a per-biomarker one -- that
# made ``mean`` shape (121,) while X is (n, 12), so the
# broadcast in ``(X - mean) / std`` failed.
_n_b = len(biomarkers)
mean = np.zeros(_n_b)
std = np.zeros(_n_b)
for _i in range(_n_b):
    _pool = np.concatenate([np.atleast_1d(np.asarray(a, dtype=float))
                            for a in all_samples[_i::_n_b]])
    mean[_i] = _pool.mean()
    std[_i] = _pool.std()
    if std[_i] == 0:
        std[_i] = 1.0
for pid, d in interp_data.items():
    for i, b in enumerate(biomarkers):
        d[b] = (d[b] - mean[i]) / std[i]
tau = 1.0
lag_steps = int(round(tau / dt))
C0 = np.zeros((12, 12))
Ctau = np.zeros((12, 12))
N_pairs = 0
for pid, d in interp_data.items():
    n = len(d['times'])
    if n <= lag_steps:
        continue
    X = np.array([d[b] for b in biomarkers]).T
    X_t = X[:-lag_steps]
    X_tau = X[lag_steps:]
    C0 += X_t.T @ X_t
    Ctau += X_t.T @ X_tau
    N_pairs += len(X_t)
C0 /= N_pairs
Ctau /= N_pairs
eigvals, eigvecs = eigh(Ctau, C0)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
v1 = eigvecs[:, 0]
v1 = v1 / np.sqrt(v1 @ C0 @ v1)
max_idx = np.argmax(np.abs(v1))
if v1[max_idx] < 0:
    v1 = -v1
lambda1 = eigvals[0]
lambda2 = eigvals[1] if len(eigvals) > 1 else 0.0
S = (lambda1 - lambda2) / lambda1 if lambda1 > 0 else 0.0
df['rc'] = np.nan
for pid in df['patient_id'].unique():
    pdata = df[df['patient_id'] == pid]
    X = pdata[biomarkers].values
    X_std = (X - mean) / std
    rc_vals = X_std @ v1
    df.loc[df['patient_id'] == pid, 'rc'] = rc_vals
rc_all = df['rc'].dropna().values
p1, p99 = np.percentile(rc_all, [1, 99])
bins = np.linspace(p1, p99, 21)
counts, bin_edges = np.histogram(rc_all, bins=bins)
bin_width = bin_edges[1] - bin_edges[0]
p = counts / (len(rc_all) * bin_width)
p = np.maximum(p, 1e-12)
F = -np.log(p)
F = F - np.min(F)
msd = []
for pid, d in interp_data.items():
    X = np.array([d[b] for b in biomarkers]).T
    rc_grid = X @ v1
    if len(rc_grid) > 1:
        diff = np.diff(rc_grid)**2
        msd.extend(diff)
msd = np.mean(msd)
D = msd / (2 * dt)
valid = df['rc'].notna() & df['diagnosis'].notna()
auc = roc_auc_score(df.loc[valid, 'diagnosis'], df.loc[valid, 'rc'])
result = {
    'rc_vector': v1.tolist(),
    'free_energy': F.tolist(),
    'diffusion_coefficient': float(D),
    'validation_score': float(S),
    'auc': float(auc)
}
with open('/app/submission.json', 'w') as f:
    json.dump(result, f, indent=2)
print("Solution completed.")
EOF
python /app/solution.py
