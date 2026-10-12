import json, numpy as np, os

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    config = {
        'N': 100,
        'dt': 1.0,
        'T_trial': 200.0,
        'stim_start': 10.0,
        'stim_end': 30.0,
        'resp_start': 30.0,
        'resp_end': 80.0,
        'tau_m': 10.0,
        'tau_d': 200.0,
        'U': 0.15,
        'U_fs_attn': 0.30,
        'input_gain': 1.5,
        'attn_gain': 1.5,
        'noise_sigma': 0.3,
        'noise_sigma_high': 0.8,
        'n_trials_per_condition': 100,
        'n_discard': 10
    }
    with open(os.path.join(data_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    N = config['N']
    g = 1.2
    p = 0.1
    def make_weight(seed):
        r = np.random.default_rng(seed)
        W = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j and r.random() < p:
                    W[i, j] = r.normal(0, g / np.sqrt(N * p))
        return W
    W_S = make_weight(1)
    W_F = make_weight(2)
    W_SF = make_weight(3)
    W_FS = make_weight(4)
    np.save(os.path.join(data_dir, 'W_S.npy'), W_S)
    np.save(os.path.join(data_dir, 'W_F.npy'), W_F)
    np.save(os.path.join(data_dir, 'W_SF.npy'), W_SF)
    np.save(os.path.join(data_dir, 'W_FS.npy'), W_FS)

    selectivity = np.zeros((N, 2))
    selectivity[:N//2, 0] = 1.0
    selectivity[N//2:, 1] = 1.0
    np.save(os.path.join(data_dir, 'selectivity.npy'), selectivity)

    conditions = ['baseline', 'noise', 'attention']
    n_trials = config['n_trials_per_condition']
    trials = []
    seq_rng = np.random.default_rng(123)
    for cond in conditions:
        for t in range(n_trials):
            label = 1 if seq_rng.random() < 0.2 else 0
            seed = int(seq_rng.integers(0, 2**31 - 1))
            trials.append({'condition': cond, 'label': label, 'seed': seed})
    with open(os.path.join(data_dir, 'sequences.json'), 'w') as f:
        json.dump(trials, f, indent=2)

if __name__ == '__main__':
    main()
