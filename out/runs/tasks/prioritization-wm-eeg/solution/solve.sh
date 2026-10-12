#!/bin/bash
set -euo pipefail

python - <<'PY'
import numpy as np
import json
import mne
from scipy.signal import butter, filtfilt, hilbert

d = np.load('/app/data/eeg_data.npz', allow_pickle=True)
data_pri = d['data_pri']
data_neu = d['data_neu']
times = d['times']
fs = float(d['fs'])

def peak_freq(data):
    freqs = np.arange(10, 25.5, 0.5)
    power = mne.time_frequency.tfr_array_morlet(data, sfreq=fs, freqs=freqs,
                                                n_cycles=7, output='power',
                                                verbose='ERROR')
    power_mean = power.mean(axis=(0,1))
    base_idx = np.where((times >= -0.2) & (times < 0))[0]
    base_power = power_mean[:, base_idx].mean(axis=1, keepdims=True)
    norm_power = power_mean / base_power
    time_idx = np.where((times >= 0.2) & (times <= 0.8))[0]
    avg_power = norm_power[:, time_idx].mean(axis=1)
    return freqs[np.argmax(avg_power)]

def itpc_15(data):
    coeffs = mne.time_frequency.tfr_array_morlet(data, sfreq=fs, freqs=np.array([15.0]),
                                                 n_cycles=7, output='complex',
                                                 verbose='ERROR')
    mean_coeff = coeffs.mean(axis=0)
    itpc = np.abs(mean_coeff)
    time_idx = np.where((times >= 0.2) & (times <= 0.8))[0]
    return itpc[:, 0, time_idx].mean()

def burst_rate(data):
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

def self_corr(data):
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

results = {
    'peak_freq_prioritization': float(peak_freq(data_pri)),
    'peak_freq_neutral': float(peak_freq(data_neu)),
    'itpc_15hz_prioritization': float(itpc_15(data_pri)),
    'itpc_15hz_neutral': float(itpc_15(data_neu)),
    'burst_rate_prioritization': float(burst_rate(data_pri)),
    'burst_rate_neutral': float(burst_rate(data_neu)),
    'self_corr_prioritization': float(self_corr(data_pri)),
    'self_corr_neutral': float(self_corr(data_neu))
}

with open('/app/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
PY
