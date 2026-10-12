#!/usr/bin/env python
import os
import numpy as np
import pandas as pd

SEED = 20260729
rng = np.random.default_rng(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, 'eeg'), exist_ok=True)

n_subjects = 10
subjects = [f'sub-{i:02d}' for i in range(1, n_subjects + 1)]
tasks = ['emotion', 'sex']
congruencies = ['congruent', 'incongruent']
channels = [f'EEG{i:02d}' for i in range(1, 17)]
times = np.linspace(-100, 500, 128)
traj_times = np.linspace(0, 1, 101)

pattern = np.zeros(16)
pattern[[4, 5, 6, 7]] = [1.0, -1.0, 1.0, -1.0]

def gaussian(t, mu, sigma):
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)

subj_rt = rng.normal(0, 30, n_subjects)
subj_dev = rng.normal(0, 0.02, n_subjects)
subj_eeg_amp = rng.normal(1.0, 0.15, n_subjects)

events = []
traj_rows = []

for si, subj in enumerate(subjects):
    n_trials = 200
    eeg = np.zeros((n_trials, 16, 128), dtype=np.float32)
    tid = 0
    for task in tasks:
        for cong in congruencies:
            for rep in range(50):
                latent = rng.normal(0, 1)
                base_rt = 700 if task == 'emotion' else 680
                rt_cong = 0 if cong == 'congruent' else (80 if task == 'emotion' else 30)
                rt = base_rt + rt_cong + subj_rt[si] + rng.normal(0, 40)
                base_dev = 0.05
                dev_cong = 0 if cong == 'congruent' else (0.15 if task == 'emotion' else 0.05)
                dev = base_dev + dev_cong + subj_dev[si] + 0.03 * latent + rng.normal(0, 0.02)
                dev = float(max(0.0, dev))
                if task == 'emotion':
                    amp = 0.0 if cong == 'congruent' else 2.0 * subj_eeg_amp[si]
                    peak_lat = 200.0
                else:
                    amp = 0.0 if cong == 'congruent' else 0.8 * subj_eeg_amp[si]
                    peak_lat = 300.0
                amp_trial = amp * (1.0 + 0.3 * latent)
                for t_idx, t in enumerate(times):
                    basis = gaussian(t, peak_lat, 50.0)
                    signal = amp_trial * pattern * basis
                    noise = rng.normal(0, 1.0, 16)
                    eeg[tid, :, t_idx] = signal + noise
                events.append({
                    'subject_id': subj,
                    'trial_id': tid,
                    'task': task,
                    'congruency': cong,
                    'rt_ms': float(rt),
                    'response': 'left' if rng.random() < 0.5 else 'right',
                    'accuracy': 1
                })
                x = np.linspace(0, 1, 101)
                bump = np.exp(-0.5 * ((x - 0.4) / 0.15) ** 2)
                y = dev * bump + rng.normal(0, 0.01, 101)
                for ti in range(101):
                    traj_rows.append({
                        'subject_id': subj,
                        'trial_id': tid,
                        'task': task,
                        'congruency': cong,
                        'time': float(traj_times[ti]),
                        'x': float(x[ti]),
                        'y': float(y[ti])
                    })
                tid += 1
    np.savez_compressed(os.path.join(BASE, 'eeg', f'{subj}.npz'),
                        eeg=eeg, times=times,
                        trial_ids=np.arange(n_trials),
                        channels=np.array(channels))

pd.DataFrame(events).to_csv(os.path.join(BASE, 'events.csv'), index=False)
pd.DataFrame(traj_rows).to_csv(os.path.join(BASE, 'mouse_trajectories.csv'), index=False)
with open(os.path.join(BASE, 'subjects.txt'), 'w') as f:
    f.write('\n'.join(subjects))
print('Synthetic data generated.')
