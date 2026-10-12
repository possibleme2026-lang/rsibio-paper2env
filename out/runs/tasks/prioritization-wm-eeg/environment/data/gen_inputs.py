import numpy as np
import pandas as pd
import os

np.random.seed(42)

fs = 250
n_trials = 120
n_channels = 32
tmin, tmax = -0.5, 2.0
n_times = int((tmax - tmin) * fs) + 1
times = np.linspace(tmin, tmax, n_times)

ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8',
            'P7','P3','Pz','P4','P8','O1','Oz','O2','FC1','FC2','CP1','CP2',
            'FT9','FT10','TP9','TP10','F1','F2','C1','C2']

pattern1 = np.random.randn(n_channels)
pattern1 /= np.linalg.norm(pattern1)
pattern2 = np.random.randn(n_channels)
pattern2 /= np.linalg.norm(pattern2)

def pink_noise(n, fs):
    white = np.random.randn(n)
    f = np.fft.rfftfreq(n, 1/fs)
    f[0] = f[1] if len(f) > 1 else 1
    F = np.fft.rfft(white)
    F = F / np.sqrt(f)
    pink = np.fft.irfft(F, n)
    return pink

def generate_condition(cond):
    data = np.zeros((n_trials, n_channels, n_times))
    for tr in range(n_trials):
        for ch in range(n_channels):
            noise = pink_noise(n_times, fs)
            noise = noise / np.std(noise) * 1.0
            data[tr, ch, :] = noise
        alpha_phase = np.random.rand() * 2 * np.pi
        alpha_amp = 0.2 * (1 + 0.5 * np.random.randn())
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * times + alpha_phase)
        data[tr, :, :] += alpha[None, :] * pattern1[:, None] * 0.5
        erp = 4.0 * np.exp(-((times - 0.2)**2) / (2 * 0.05**2))
        data[tr, :, :] += erp[None, :] * pattern1[:, None]
        if cond == 'prioritization':
            freq = 15.0
            amp = 4.0
            rot_freq = 2.0
        else:
            freq = 20.0
            amp = 4.0
            rot_freq = 0.0
        phase = np.zeros(n_times)
        phase[times < 0] = np.random.rand() * 2 * np.pi
        phase[times >= 0] = 0.0
        osc = amp * np.sin(2 * np.pi * freq * times + phase)
        env = np.zeros(n_times)
        env[times >= 0] = np.exp(-((times[times >= 0] - 0.4)**2) / (2 * 0.25**2))
        osc = osc * env
        if rot_freq > 0:
            theta = 2 * np.pi * rot_freq * times
            pat = np.cos(theta)[None, :] * pattern1[:, None] + np.sin(theta)[None, :] * pattern2[:, None]
        else:
            pat = pattern1[:, None] * np.ones((1, n_times))
        data[tr, :, :] += osc[None, :] * pat
        n_bursts = np.random.poisson(1.0)
        for _ in range(n_bursts):
            burst_onset = np.random.uniform(0.1, 1.5)
            burst_env = np.exp(-((times - burst_onset)**2) / (2 * (0.1/4)**2))
            burst_phase = np.random.rand() * 2 * np.pi
            burst = 2.0 * np.sin(2 * np.pi * 20 * times + burst_phase) * burst_env
            data[tr, :, :] += burst[None, :] * pattern1[:, None]
    return data

data_pri = generate_condition('prioritization')
data_neu = generate_condition('neutral')

outdir = os.path.dirname(os.path.abspath(__file__))
np.savez_compressed(os.path.join(outdir, 'eeg_data.npz'),
                    data_pri=data_pri,
                    data_neu=data_neu,
                    times=times,
                    ch_names=ch_names,
                    fs=fs)

events = []
for cond, n in [('prioritization', n_trials), ('neutral', n_trials)]:
    for tr in range(n):
        events.append({'trial': tr, 'condition': cond, 'cue_onset_sample': int((0 - tmin) * fs)})
pd.DataFrame(events).to_csv(os.path.join(outdir, 'events.csv'), index=False)

with open(os.path.join(outdir, 'README.md'), 'w') as f:
    f.write('# Synthetic EEG dataset\n\n')
    f.write('This dataset mimics Experiment 2 of the prioritization working memory study.\n')
    f.write('Two conditions: prioritization and neutral.\n')
    f.write('Data shape: (120 trials, 32 channels, 626 time points), fs=250 Hz.\n')
    f.write('Time range: -0.5 to 2.0 s relative to cue onset.\n')
    f.write('See instruction for analysis details.\n')
