import numpy as np
import pandas as pd
from scipy.stats import norm

def logistic(x):
    return 1 / (1 + np.exp(-x))

def run_simulation(seed=42):
    np.random.seed(seed)
    n_years = 21
    n_bins = 20
    bin_min = 10
    bin_max = 100
    initial_N = 3000
    bin_width = (bin_max - bin_min) / n_bins
    bin_edges = np.linspace(bin_min, bin_max, n_bins+1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    true_params = {
        'g0': 0.5,
        'g_size': -0.01,
        'g_vpd': -0.25,
        'g_precip': 0.02,
        'sd_growth': 0.3,
        's0': -2.0,
        's_size': 0.05,
        's_vpd': -0.6,
        'R': 30,
        'rec_mean': 12,
        'rec_sd': 2
    }
    
    years = np.arange(2000, 2000 + n_years)
    vpd = 1.5 + 0.8 * np.sin((years - 2000) / 2) + np.random.normal(0, 0.3, n_years)
    precip = 800 + 300 * np.sin((years - 2000) / 3) + np.random.normal(0, 100, n_years)
    vpd = np.clip(vpd, 0.2, 3.0)
    precip = np.clip(precip, 200, 1500)
    
    probs = norm.pdf(bin_mids, loc=30, scale=15)
    probs = probs / probs.sum()
    initial_counts = np.random.multinomial(initial_N, probs)
    
    sizes = []
    for i, c in enumerate(initial_counts):
        sizes.extend(np.random.uniform(bin_edges[i], bin_edges[i+1], c))
    sizes = np.array(sizes)
    
    true_counts_by_year = []
    observed_counts_by_year = []
    growth_records = []
    
    for t in range(n_years):
        year = years[t]
        counts, _ = np.histogram(sizes, bins=bin_edges)
        true_counts_by_year.append(counts.copy())
        obs_counts = np.random.poisson(counts)
        observed_counts_by_year.append(obs_counts)
        
        if t == n_years - 1:
            break
        
        vpd_t = vpd[t]
        precip_t = precip[t]
        p_surv = logistic(true_params['s0'] + true_params['s_size'] * sizes + true_params['s_vpd'] * vpd_t)
        survive = np.random.rand(len(sizes)) < p_surv
        sizes = sizes[survive]
        growth_mean = (true_params['g0'] + true_params['g_size'] * sizes + 
                       true_params['g_vpd'] * vpd_t + true_params['g_precip'] * precip_t / 1000)
        growth = np.random.normal(growth_mean, true_params['sd_growth'])
        sizes = sizes + growth
        sizes = np.clip(sizes, bin_min, bin_max)
        n_rec = np.random.poisson(true_params['R'])
        rec_sizes = np.random.normal(true_params['rec_mean'], true_params['rec_sd'], n_rec)
        rec_sizes = np.clip(rec_sizes, bin_min, bin_max)
        sizes = np.concatenate([sizes, rec_sizes])
        
        counts_t = true_counts_by_year[t]
        if counts_t.sum() > 0:
            bin_indices = np.random.choice(n_bins, size=100, p=counts_t/counts_t.sum())
            for bi in bin_indices:
                size = np.random.uniform(bin_edges[bi], bin_edges[bi+1])
                growth_mean_ind = (true_params['g0'] + true_params['g_size'] * size + 
                                   true_params['g_vpd'] * vpd_t + true_params['g_precip'] * precip_t / 1000)
                growth_ind = np.random.normal(growth_mean_ind, true_params['sd_growth'])
                growth_records.append({
                    'year': int(year),
                    'initial_size': float(size),
                    'vpd': float(vpd_t),
                    'precip': float(precip_t),
                    'growth_increment': float(growth_ind)
                })
    
    counts_2019 = true_counts_by_year[19].astype(float)
    vpd_2019 = vpd[19]
    precip_2019 = precip[19]
    T = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        size = bin_mids[i]
        p_surv = logistic(true_params['s0'] + true_params['s_size'] * size + true_params['s_vpd'] * vpd_2019)
        growth_mean = (true_params['g0'] + true_params['g_size'] * size + 
                       true_params['g_vpd'] * vpd_2019 + true_params['g_precip'] * precip_2019 / 1000)
        for j in range(n_bins):
            lower = bin_edges[j]
            upper = bin_edges[j+1]
            prob = norm.cdf(upper, loc=size+growth_mean, scale=true_params['sd_growth']) - \
                   norm.cdf(lower, loc=size+growth_mean, scale=true_params['sd_growth'])
            T[i, j] = p_surv * prob
    rec_dist = np.zeros(n_bins)
    for j in range(n_bins):
        lower = bin_edges[j]
        upper = bin_edges[j+1]
        prob = norm.cdf(upper, loc=true_params['rec_mean'], scale=true_params['rec_sd']) - \
               norm.cdf(lower, loc=true_params['rec_mean'], scale=true_params['rec_sd'])
        rec_dist[j] = prob
    rec_dist = rec_dist / rec_dist.sum()
    expected_2020_counts = counts_2019 @ T + true_params['R'] * rec_dist
    true_2020_probs = expected_2020_counts / expected_2020_counts.sum()
    true_mean_size_2020 = float(np.sum(true_2020_probs * bin_mids))
    
    pop_rows = []
    for t, counts in enumerate(observed_counts_by_year):
        year = int(years[t])
        for i in range(n_bins):
            pop_rows.append({
                'year': year,
                'bin_lower': float(bin_edges[i]),
                'bin_upper': float(bin_edges[i+1]),
                'count': int(counts[i])
            })
    pop_df = pd.DataFrame(pop_rows)
    growth_df = pd.DataFrame(growth_records)
    weather_df = pd.DataFrame({
        'year': years,
        'vpd': vpd,
        'precip': precip
    })
    bin_df = pd.DataFrame({
        'bin_lower': bin_edges[:-1],
        'bin_upper': bin_edges[1:]
    })
    
    gold = {
        'true_params': true_params,
        'true_2020_probs': true_2020_probs.tolist(),
        'true_mean_size_2020': true_mean_size_2020,
        'bin_edges': bin_edges.tolist(),
        'bin_mids': bin_mids.tolist()
    }
    
    data = {
        'population_counts': pop_df,
        'growth_records': growth_df,
        'weather': weather_df,
        'bin_edges': bin_df
    }
    
    return data, gold
