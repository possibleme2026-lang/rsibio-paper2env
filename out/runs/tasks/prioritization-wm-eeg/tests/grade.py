import numpy as np
import json
import os
from scipy.signal import butter, filtfilt, hilbert
import mne

# ---------------------------------------------------------------------------
# Reference analysis pipeline (gold standard)
# ---------------------------------------------------------------------------

def load_data():
    d = np.load('/app/data/eeg_data.npz', allow_pickle=True)
    data_pri = d['data_pri']
    data_neu = d['data_neu']
    times = d['times']
    fs = float(d['fs'])
    ch_names = d['ch_names']
    return data_pri, data_neu, times, fs, ch_names

def compute_peak_freq(data, times, fs):
    """Peak frequency of normalized Morlet power in 10-25 Hz."""
    freqs = np.arange(10, 25.5, 0.5)
    n_cycles = 7
    power = mne.time_frequency.tfr_array_morlet(data, sfreq=fs, freqs=freqs,
                                                n_cycles=n_cycles, output='power',
                                                verbose='ERROR')
    power_mean = power.mean(axis=(0, 1))  # (n_freqs, n_times)
    base_idx = np.where((times >= -0.2) & (times < 0))[0]
    base_power = power_mean[:, base_idx].mean(axis=1, keepdims=True)
    norm_power = power_mean / base_power
    time_idx = np.where((times >= 0.2) & (times <= 0.8))[0]
    avg_power = norm_power[:, time_idx].mean(axis=1)
    return freqs[np.argmax(avg_power)]

def compute_itpc(data, times, fs, freq=15.0):
    """Inter-trial phase coherence at a given frequency."""
    coeffs = mne.time_frequency.tfr_array_morlet(data, sfreq=fs, freqs=np.array([freq]),
                                                 n_cycles=7, output='complex',
                                                 verbose='ERROR')
    mean_coeff = coeffs.mean(axis=0)
    itpc = np.abs(mean_coeff)
    time_idx = np.where((times >= 0.2) & (times <= 0.8))[0]
    return itpc[:, 0, time_idx].mean()

def compute_burst_rate(data, times, fs):
    """Beta burst rate (15-25 Hz) using z-scored Hilbert envelope."""
    b, a = butter(4, [15/(fs/2), 25/(fs/2)], btype='band')
    filtered = filtfilt(b, a, data, axis=2)
    envelope = np.abs(hilbert(filtered, axis=2))
    base_idx = np.where((times >= -0.5) & (times < 0))[0]
    base_mean = envelope[:, :, base_idx].mean(axis=2, keepdims=True)
    base_std = envelope[:, :, base_idx].std(axis=2, keepdims=True)
    z = (envelope - base_mean) / base_std
    time_idx = np.where((times >= 0) & (times <= 1.5))[0]
    n_samples = len(time_idx)
    min_dur = int(0.05 * fs)
    burst_counts = []
    for tr in range(data.shape[0]):
        for ch in range(data.shape[1]):
            z_trial = z[tr, ch, time_idx]
            above = z_trial > 3
            edges = np.diff(above.astype(int))
            starts = np.where(edges == 1)[0] + 1
            ends = np.where(edges == -1)[0] + 1
            if above[0]:
                starts = np.r_[0, starts]
            if above[-1]:
                ends = np.r_[ends, n_samples]
            count = 0
            for s, e in zip(starts, ends):
                if (e - s) >= min_dur:
                    count += 1
            burst_counts.append(count)
    total_bursts = np.sum(burst_counts)
    total_time = 1.5 * data.shape[0] * data.shape[1]
    return total_bursts / total_time

def compute_self_corr(data, times, fs):
    """Topographic self-correlation (Fisher z) across time lags."""
    b, a = butter(4, [1/(fs/2), 30/(fs/2)], btype='band')
    filtered = filtfilt(b, a, data, axis=2)
    time_idx = np.where((times >= 0.2) & (times <= 0.8))[0]
    lags = [int(0.05*fs), int(0.10*fs), int(0.15*fs), int(0.20*fs)]
    z_values = []
    for tr in range(data.shape[0]):
        for t in time_idx:
            for lag in lags:
                if t + lag >= filtered.shape[2]:
                    continue
                p1 = filtered[tr, :, t]
                p2 = filtered[tr, :, t + lag]
                p1 = (p1 - p1.mean()) / p1.std()
                p2 = (p2 - p2.mean()) / p2.std()
                r = np.corrcoef(p1, p2)[0, 1]
                if r == 1.0:
                    z = 3.0
                elif r == -1.0:
                    z = -3.0
                else:
                    z = 0.5 * np.log((1 + r) / (1 - r))
                z_values.append(z)
    return np.mean(z_values)

# ---------------------------------------------------------------------------
# Grading logic
# ---------------------------------------------------------------------------

def main():
    result_path = '/app/results.json'
    if not os.path.exists(result_path):
        print('ERROR: /app/results.json not found')
        score = 0.0
    else:
        with open(result_path) as f:
            try:
                agent = json.load(f)
            except Exception as e:
                print(f'ERROR: could not parse /app/results.json: {e}')
                agent = {}
        required_keys = ['peak_freq_prioritization', 'peak_freq_neutral',
                         'itpc_15hz_prioritization', 'itpc_15hz_neutral',
                         'burst_rate_prioritization', 'burst_rate_neutral',
                         'self_corr_prioritization', 'self_corr_neutral']
        format_ok = True
        for k in required_keys:
            if k not in agent or not isinstance(agent[k], (int, float)):
                format_ok = False
                print(f'Missing or non-numeric key: {k}')
        if not format_ok:
            score = 0.0
        else:
            data_pri, data_neu, times, fs, ch_names = load_data()
            gold_peak_pri = compute_peak_freq(data_pri, times, fs)
            gold_peak_neu = compute_peak_freq(data_neu, times, fs)
            gold_itpc_pri = compute_itpc(data_pri, times, fs, freq=15.0)
            gold_itpc_neu = compute_itpc(data_neu, times, fs, freq=15.0)
            gold_burst_pri = compute_burst_rate(data_pri, times, fs)
            gold_burst_neu = compute_burst_rate(data_neu, times, fs)
            gold_self_pri = compute_self_corr(data_pri, times, fs)
            gold_self_neu = compute_self_corr(data_neu, times, fs)

            print('Gold metrics:')
            print(f'  peak_freq_pri: {gold_peak_pri}')
            print(f'  peak_freq_neu: {gold_peak_neu}')
            print(f'  itpc_15hz_pri: {gold_itpc_pri}')
            print(f'  itpc_15hz_neu: {gold_itpc_neu}')
            print(f'  burst_rate_pri: {gold_burst_pri}')
            print(f'  burst_rate_neu: {gold_burst_neu}')
            print(f'  self_corr_pri: {gold_self_pri}')
            print(f'  self_corr_neu: {gold_self_neu}')

            c1_pri = abs(agent['peak_freq_prioritization'] - gold_peak_pri) < 0.6 and 14.0 <= agent['peak_freq_prioritization'] <= 16.0
            c1_neu = abs(agent['peak_freq_neutral'] - gold_peak_neu) < 0.6 and 19.0 <= agent['peak_freq_neutral'] <= 21.0
            c1 = c1_pri and c1_neu
            print(f'Constraint 1 (peak freq): pri ok={c1_pri}, neu ok={c1_neu} -> {c1}')

            c2_diff = agent['itpc_15hz_prioritization'] > agent['itpc_15hz_neutral'] + 0.05
            c2_pri = abs(agent['itpc_15hz_prioritization'] - gold_itpc_pri) < 0.1
            c2_neu = abs(agent['itpc_15hz_neutral'] - gold_itpc_neu) < 0.1
            c2 = c2_diff and c2_pri and c2_neu
            print(f'Constraint 2 (ITPC): diff ok={c2_diff}, pri ok={c2_pri}, neu ok={c2_neu} -> {c2}')

            c3_diff = abs(agent['burst_rate_prioritization'] - agent['burst_rate_neutral']) < 0.5
            c3_pri = abs(agent['burst_rate_prioritization'] - gold_burst_pri) < 0.5
            c3_neu = abs(agent['burst_rate_neutral'] - gold_burst_neu) < 0.5
            c3 = c3_diff and c3_pri and c3_neu
            print(f'Constraint 3 (burst rate): diff ok={c3_diff}, pri ok={c3_pri}, neu ok={c3_neu} -> {c3}')

            c4_diff = agent['self_corr_neutral'] > agent['self_corr_prioritization'] + 0.1
            c4_pri = abs(agent['self_corr_prioritization'] - gold_self_pri) < 0.2
            c4_neu = abs(agent['self_corr_neutral'] - gold_self_neu) < 0.2
            c4 = c4_diff and c4_pri and c4_neu
            print(f'Constraint 4 (self-corr): diff ok={c4_diff}, pri ok={c4_pri}, neu ok={c4_neu} -> {c4}')

            c5 = format_ok
            print(f'Constraint 5 (format): {c5}')

            constraints = [c1, c2, c3, c4, c5]
            score = sum(constraints) / len(constraints)

    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': score, 'reward': score}, f)
    print(f'Final score: {score}')

if __name__ == '__main__':
    main()
