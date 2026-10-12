#!/bin/bash
set -euo pipefail
mkdir -p /app/submission
python3 << 'EOF'
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import json
import os

data_dir = '/app/data'
pop = pd.read_csv(f'{data_dir}/population_counts.csv')
growth = pd.read_csv(f'{data_dir}/growth_records.csv')
weather = pd.read_csv(f'{data_dir}/weather.csv')
bins = pd.read_csv(f'{data_dir}/bin_edges.csv')

bin_edges = bins['bin_lower'].tolist() + [bins['bin_upper'].iloc[-1]]
bin_mids = [(bins['bin_lower'].iloc[i] + bins['bin_upper'].iloc[i])/2 for i in range(len(bins))]
n_bins = len(bins)

X = np.column_stack([np.ones(len(growth)), growth['initial_size'], growth['vpd'], growth['precip']])
y = growth['growth_increment'].values
coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
g0, g_size, g_vpd, g_precip = coef
resid = y - X @ coef
sd_growth = np.std(resid)

weather_dict = {int(row['year']): (row['vpd'], row['precip']) for _, row in weather.iterrows()}

def logistic(x):
    return 1 / (1 + np.exp(-x))

def transition_matrix(s0, s_size, s_vpd, vpd_t, precip_t):
    T = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        size = bin_mids[i]
        p_surv = logistic(s0 + s_size * size + s_vpd * vpd_t)
        growth_mean = g0 + g_size * size + g_vpd * vpd_t + g_precip * precip_t
        for j in range(n_bins):
            lower = bin_edges[j]
            upper = bin_edges[j+1]
            prob = norm.cdf(upper, loc=size+growth_mean, scale=sd_growth) - \
                   norm.cdf(lower, loc=size+growth_mean, scale=sd_growth)
            T[i, j] = p_surv * prob
    return T

def recruitment_dist():
    rec_dist = np.zeros(n_bins)
    for j in range(n_bins):
        lower = bin_edges[j]
        upper = bin_edges[j+1]
        prob = norm.cdf(upper, loc=12.0, scale=2.0) - \
               norm.cdf(lower, loc=12.0, scale=2.0)
        rec_dist[j] = prob
    return rec_dist / rec_dist.sum()

rec_dist = recruitment_dist()

years = sorted(pop['year'].unique())
counts_by_year = {}
for y in years:
    df_y = pop[pop['year'] == y]
    counts_by_year[y] = df_y['count'].values

def objective(params):
    s0, s_size, s_vpd, R = params
    total_ss = 0.0
    for t in range(2000, 2019):
        obs_prev = counts_by_year[t].astype(float)
        vpd_t, precip_t = weather_dict[t]
        T = transition_matrix(s0, s_size, s_vpd, vpd_t, precip_t)
        pred = obs_prev @ T + R * rec_dist
        obs_next = counts_by_year[t+1].astype(float)
        total_ss += np.sum((pred - obs_next)**2)
    return total_ss

init_params = [-2.0, 0.05, -0.6, 30.0]
bounds = [(-5, 0), (-0.1, 0.2), (-2, 0), (0, 100)]
res = minimize(objective, init_params, bounds=bounds, method='L-BFGS-B')
s0_fit, s_size_fit, s_vpd_fit, R_fit = res.x

counts_2019 = counts_by_year[2019].astype(float)
vpd_2019, precip_2019 = weather_dict[2019]
T_2019 = transition_matrix(s0_fit, s_size_fit, s_vpd_fit, vpd_2019, precip_2019)
pred_counts_2020 = counts_2019 @ T_2019 + R_fit * rec_dist
pred_probs_2020 = pred_counts_2020 / pred_counts_2020.sum()

os.makedirs('/app/submission', exist_ok=True)
pred_df = pd.DataFrame({
    'bin_lower': bins['bin_lower'],
    'bin_upper': bins['bin_upper'],
    'predicted_count': pred_counts_2020,
    'predicted_prob': pred_probs_2020
})
pred_df.to_csv('/app/submission/predictions.csv', index=False)

params_out = {
    'growth_vpd': float(g_vpd),
    'growth_intercept': float(g0),
    'survival_vpd': float(s_vpd_fit),
    'survival_intercept': float(s0_fit),
    'recruitment': float(R_fit)
}
with open('/app/submission/params.json', 'w') as f:
    json.dump(params_out, f, indent=2)
EOF
