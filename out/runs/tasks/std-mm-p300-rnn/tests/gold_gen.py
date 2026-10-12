import json, numpy as np, os

def simulate_block(config, W_S, W_F, W_SF, W_FS, selectivity, trials, condition):
    N = config['N']
    dt = config['dt']
    steps = int(config['T_trial'] / dt)
    tau_m = config['tau_m']
    tau_d = config['tau_d']
    U = config['U']
    U_fs = config['U_fs_attn'] if condition == 'attention' else U
    gain = config['input_gain'] * (config['attn_gain'] if condition == 'attention' else 1.0)
    sigma = config['noise_sigma_high'] if condition == 'noise' else config['noise_sigma']
    stim_start_idx = int(config['stim_start'] / dt)
    stim_end_idx = int(config['stim_end'] / dt)
    resp_start_idx = int(config['resp_start'] / dt)
    resp_end_idx = int(config['resp_end'] / dt)

    D_S = np.ones((N, N))
    D_F = np.ones((N, N))
    D_SF = np.ones((N, N))
    D_FS = np.ones((N, N))

    results = []
    for trial in trials:
        label = trial['label']
        seed = trial['seed']
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, sigma, size=(N, steps))
        u_S = np.zeros(N)
        u_F = np.zeros(N)
        x_S_record = np.zeros((N, steps))
        x_F_record = np.zeros((N, steps))
        for t in range(steps):
            x_S = np.maximum(u_S, 0)
            x_F = np.maximum(u_F, 0)
            x_S_record[:, t] = x_S
            x_F_record[:, t] = x_F
            stim = 1.0 if stim_start_idx <= t < stim_end_idx else 0.0
            I_S = gain * stim * selectivity[:, label] + noise[:, t]
            rec_S = (W_S * D_S) @ x_S
            fb = (W_FS * D_FS) @ x_F
            u_S = u_S + dt/tau_m * (-u_S + I_S + rec_S + fb)
            rec_F = (W_F * D_F) @ x_F
            ff = (W_SF * D_SF) @ x_S
            u_F = u_F + dt/tau_m * (-u_F + rec_F + ff)
            D_S = D_S + dt * ((1 - D_S)/tau_d - U * x_S[None, :] * D_S)
            D_F = D_F + dt * ((1 - D_F)/tau_d - U * x_F[None, :] * D_F)
            D_SF = D_SF + dt * ((1 - D_SF)/tau_d - U * x_S[None, :] * D_SF)
            D_FS = D_FS + dt * ((1 - D_FS)/tau_d - U_fs * x_F[None, :] * D_FS)
            D_S = np.clip(D_S, 0, 1)
            D_F = np.clip(D_F, 0, 1)
            D_SF = np.clip(D_SF, 0, 1)
            D_FS = np.clip(D_FS, 0, 1)
        mean_S = x_S_record[:, resp_start_idx:resp_end_idx].mean()
        mean_F = x_F_record[:, resp_start_idx:resp_end_idx].mean()
        mean_S_vec = x_S_record[:, resp_start_idx:resp_end_idx].mean(axis=1)
        mean_F_vec = x_F_record[:, resp_start_idx:resp_end_idx].mean(axis=1)
        time_series_S = x_S_record.mean(axis=0)
        time_series_F = x_F_record.mean(axis=0)
        results.append({
            'label': label,
            'mean_S': mean_S,
            'mean_F': mean_F,
            'mean_S_vec': mean_S_vec,
            'mean_F_vec': mean_F_vec,
            'time_series_S': time_series_S,
            'time_series_F': time_series_F
        })
    return results

def participation_ratio(vectors):
    X = np.array(vectors)
    X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    return (eigvals.sum() ** 2) / (eigvals ** 2).sum()

def compute_gold(data_dir):
    config = json.load(open(os.path.join(data_dir, 'config.json')))
    W_S = np.load(os.path.join(data_dir, 'W_S.npy'))
    W_F = np.load(os.path.join(data_dir, 'W_F.npy'))
    W_SF = np.load(os.path.join(data_dir, 'W_SF.npy'))
    W_FS = np.load(os.path.join(data_dir, 'W_FS.npy'))
    selectivity = np.load(os.path.join(data_dir, 'selectivity.npy'))
    sequences = json.load(open(os.path.join(data_dir, 'sequences.json')))
    n_discard = config['n_discard']
    cond_trials = {}
    for t in sequences:
        cond_trials.setdefault(t['condition'], []).append(t)
    cond_results = {}
    for cond in ['baseline', 'noise', 'attention']:
        cond_results[cond] = simulate_block(config, W_S, W_F, W_SF, W_FS, selectivity, cond_trials[cond], cond)
    base_res = cond_results['baseline'][n_discard:]
    base_std = [r for r in base_res if r['label'] == 0]
    base_dev = [r for r in base_res if r['label'] == 1]
    MMN_sensory = np.mean([r['mean_S'] for r in base_dev]) - np.mean([r['mean_S'] for r in base_std])
    P300_frontal = np.mean([r['mean_F'] for r in base_dev]) - np.mean([r['mean_F'] for r in base_std])
    ts_S = np.mean([r['time_series_S'] for r in base_dev], axis=0)
    ts_F = np.mean([r['time_series_F'] for r in base_dev], axis=0)
    peak_S = np.argmax(ts_S) * config['dt']
    peak_F = np.argmax(ts_F) * config['dt']
    P300_latency_delay = peak_F - peak_S
    noise_res = cond_results['noise'][n_discard:]
    threshold = (np.mean([r['mean_F'] for r in base_std]) + np.mean([r['mean_F'] for r in base_dev])) / 2
    correct = sum(1 for r in noise_res if (1 if r['mean_F'] > threshold else 0) == r['label'])
    Noise_accuracy = correct / len(noise_res)
    PR_std = participation_ratio([r['mean_F_vec'] for r in base_std])
    PR_dev = participation_ratio([r['mean_F_vec'] for r in base_dev])
    State_space_expansion = np.log(PR_dev / PR_std)
    att_res = cond_results['attention'][n_discard:]
    att_std = [r for r in att_res if r['label'] == 0]
    att_dev = [r for r in att_res if r['label'] == 1]
    att_diff = np.mean([r['mean_F'] for r in att_dev]) - np.mean([r['mean_F'] for r in att_std])
    Attention_amplification = att_diff / P300_frontal - 1.0
    return {
        'MMN_sensory': float(MMN_sensory),
        'P300_frontal': float(P300_frontal),
        'P300_latency_delay': float(P300_latency_delay),
        'Noise_accuracy': float(Noise_accuracy),
        'State_space_expansion': float(State_space_expansion),
        'Attention_amplification': float(Attention_amplification)
    }
