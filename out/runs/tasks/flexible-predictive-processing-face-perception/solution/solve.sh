#!/bin/bash
set -euo pipefail
mkdir -p /app/output
cat > /app/solve.py <<'PY'
import json, os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

DATA = '/app/data'
OUT = '/app/output/results.json'

def load_data():
    events = pd.read_csv(os.path.join(DATA, 'events.csv'))
    traj = pd.read_csv(os.path.join(DATA, 'mouse_trajectories.csv'))
    subjects = [line.strip() for line in open(os.path.join(DATA, 'subjects.txt')) if line.strip()]
    eeg = {}
    for subj in subjects:
        npz = np.load(os.path.join(DATA, 'eeg', f'{subj}.npz'))
        eeg[subj] = {'eeg': npz['eeg'], 'times': npz['times'], 'trial_ids': npz['trial_ids']}
    return events, traj, subjects, eeg

def compute_mad(traj):
    return traj.groupby(['subject_id','trial_id']).apply(
        lambda g: np.max(np.abs(g['y'].values))
    ).reset_index(name='mad')

def compute_behavior(events, traj, subjects):
    mad = compute_mad(traj)
    df = events.merge(mad, on=['subject_id','trial_id'])
    results = {}
    subj_diffs = {}
    for measure in ['rt_ms','mad']:
        subj_diffs[measure] = {}
        for task in ['emotion','sex']:
            diffs = []
            for subj in subjects:
                sub = df[(df.subject_id==subj) & (df.task==task)]
                con = sub[sub.congruency=='congruent'][measure].mean()
                inc = sub[sub.congruency=='incongruent'][measure].mean()
                diffs.append(inc - con)
            diffs = np.array(diffs)
            t, p = stats.ttest_1samp(diffs, 0)
            subj_diffs[measure][task] = diffs
            results.setdefault(measure, {})[task] = {
                'mean_diff': float(np.mean(diffs)),
                't_stat': float(t),
                'p_value': float(p)
            }
        diff = subj_diffs[measure]['emotion'] - subj_diffs[measure]['sex']
        t, p = stats.ttest_1samp(diff, 0)
        results[measure]['task_difference'] = {
            'mean_diff': float(np.mean(diff)),
            't_stat': float(t),
            'p_value': float(p)
        }
    return results

def compute_rsa(eeg, events, subjects):
    results = {}
    for task in ['emotion','sex']:
        d_all = []
        for subj in subjects:
            npz = eeg[subj]
            eeg_data = npz['eeg']
            times = npz['times']
            sub_events = events[events.subject_id==subj]
            task_events = sub_events[sub_events.task==task]
            con_idx = task_events[task_events.congruency=='congruent']['trial_id'].values
            inc_idx = task_events[task_events.congruency=='incongruent']['trial_id'].values
            eeg_z = (eeg_data - eeg_data.mean(axis=0, keepdims=True)) / (eeg_data.std(axis=0, keepdims=True) + 1e-8)
            mean_con = eeg_z[con_idx].mean(axis=0)
            mean_inc = eeg_z[inc_idx].mean(axis=0)
            var_con = eeg_z[con_idx].var(axis=0, ddof=1)
            var_inc = eeg_z[inc_idx].var(axis=0, ddof=1)
            pooled_var = (var_con + var_inc) / 2.0
            diff = mean_inc - mean_con
            d = np.sqrt(np.sum(diff**2 / (pooled_var + 1e-8), axis=0))
            d_all.append(d)
        d_mean = np.mean(d_all, axis=0)
        peak_idx = int(np.argmax(d_mean))
        results[task] = {
            'peak_amplitude': float(d_mean[peak_idx]),
            'peak_latency_ms': float(times[peak_idx])
        }
    return results

def _decode_subject_task(eeg_data, times, task_events):
    labels = (task_events['congruency']=='incongruent').astype(int).values
    trial_ids = task_events['trial_id'].values
    X = eeg_data[trial_ids]
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    n_times = X.shape[2]
    accs = np.zeros(n_times)
    for t in range(n_times):
        Xt = X[:, :, t]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        preds = np.zeros(len(labels))
        for train_idx, test_idx in skf.split(Xt, labels):
            clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
            clf.fit(Xt[train_idx], labels[train_idx])
            preds[test_idx] = clf.predict(Xt[test_idx])
        accs[t] = balanced_accuracy_score(labels, preds)
    return accs

def compute_decoding(eeg, events, subjects):
    results = {}
    for task in ['emotion','sex']:
        acc_all = []
        last_times = None
        for subj in subjects:
            npz = eeg[subj]
            last_times = npz['times']
            sub_events = events[events.subject_id==subj]
            task_events = sub_events[sub_events.task==task]
            accs = _decode_subject_task(npz['eeg'], npz['times'], task_events)
            acc_all.append(accs)
        acc_mean = np.mean(acc_all, axis=0)
        peak_idx = int(np.argmax(acc_mean))
        results[task] = {
            'peak_accuracy': float(acc_mean[peak_idx]),
            'peak_latency_ms': float(last_times[peak_idx])
        }
    return results

def compute_fusion(eeg, events, traj, subjects, decoding_results):
    mad = compute_mad(traj)
    df = events.merge(mad, on=['subject_id','trial_id'])
    results = {}
    for task in ['emotion','sex']:
        peak_lat = decoding_results[task]['peak_latency_ms']
        rs = []
        for subj in subjects:
            npz = eeg[subj]
            eeg_data = npz['eeg']
            times = npz['times']
            t_idx = int(np.argmin(np.abs(times - peak_lat)))
            sub_events = events[events.subject_id==subj]
            task_events = sub_events[sub_events.task==task]
            labels = (task_events['congruency']=='incongruent').astype(int).values
            trial_ids = task_events['trial_id'].values
            X = eeg_data[trial_ids]
            X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
            Xt = X[:, :, t_idx]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            dec_vals = np.zeros(len(labels))
            for train_idx, test_idx in skf.split(Xt, labels):
                clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
                clf.fit(Xt[train_idx], labels[train_idx])
                dec_vals[test_idx] = clf.decision_function(Xt[test_idx])
            subj_df = df[(df.subject_id==subj) & (df.task==task)]
            mad_vals = subj_df.set_index('trial_id').loc[trial_ids, 'mad'].values
            r, _ = stats.pearsonr(dec_vals, mad_vals)
            rs.append(r)
        zs = np.arctanh(np.clip(rs, -0.999, 0.999))
        mean_r = np.tanh(np.mean(zs))
        results[task] = {'mean_r': float(mean_r)}
    results['difference'] = float(results['emotion']['mean_r'] - results['sex']['mean_r'])
    return results

def main():
    events, traj, subjects, eeg = load_data()
    behavior = compute_behavior(events, traj, subjects)
    rsa = compute_rsa(eeg, events, subjects)
    decoding = compute_decoding(eeg, events, subjects)
    fusion = compute_fusion(eeg, events, traj, subjects, decoding)
    out = {'behavior': behavior, 'rsa': rsa, 'decoding': decoding, 'fusion': fusion}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()
PY
python /app/solve.py
