#!/usr/bin/env python
import json, os, sys, math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

DATA = '/app/data'
SUBMISSION = '/app/output/results.json'
REPORT = '/logs/verifier/report.json'
REWARD = '/logs/verifier/reward.json'

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

def reference():
    events, traj, subjects, eeg = load_data()
    behavior = compute_behavior(events, traj, subjects)
    rsa = compute_rsa(eeg, events, subjects)
    decoding = compute_decoding(eeg, events, subjects)
    fusion = compute_fusion(eeg, events, traj, subjects, decoding)
    return {'behavior': behavior, 'rsa': rsa, 'decoding': decoding, 'fusion': fusion}

def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol

def check_finite(obj):
    if isinstance(obj, dict):
        return all(check_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(check_finite(v) for v in obj)
    if isinstance(obj, (int, float)):
        return math.isfinite(float(obj))
    return False

def main():
    checks = []
    if not os.path.exists(SUBMISSION):
        print('FAIL: submission not found')
        write_reward(0.0)
        return
    try:
        sub = json.load(open(SUBMISSION))
    except Exception as e:
        print(f'FAIL: invalid JSON: {e}')
        write_reward(0.0)
        return
    ref = reference()
    schema_ok = True
    try:
        for k in ['behavior','rsa','decoding','fusion']:
            if k not in sub: schema_ok = False
        for m in ['rt','mad']:
            for t in ['emotion','sex','task_difference']:
                if t not in sub['behavior'][m]: schema_ok = False
                for f in ['mean_diff','t_stat','p_value']:
                    if f not in sub['behavior'][m][t]: schema_ok = False
        for task in ['emotion','sex']:
            for key in ['rsa','decoding']:
                fields = ['peak_amplitude','peak_latency_ms'] if key=='rsa' else ['peak_accuracy','peak_latency_ms']
                for f in fields:
                    if f not in sub[key][task]: schema_ok = False
        for t in ['emotion','sex']:
            if 'mean_r' not in sub['fusion'][t]: schema_ok = False
        if 'difference' not in sub['fusion']: schema_ok = False
    except Exception:
        schema_ok = False
    checks.append(('schema', schema_ok))
    finite_ok = False
    try:
        finite_ok = check_finite(sub)
    except Exception:
        finite_ok = False
    checks.append(('finite_numbers', finite_ok))
    try:
        rt = sub['behavior']['rt']
        ref_rt = ref['behavior']['rt']
        ok = (close(rt['emotion']['mean_diff'], ref_rt['emotion']['mean_diff'], 10.0) and
              close(rt['sex']['mean_diff'], ref_rt['sex']['mean_diff'], 10.0) and
              close(rt['task_difference']['mean_diff'], ref_rt['task_difference']['mean_diff'], 10.0) and
              rt['emotion']['mean_diff'] > rt['sex']['mean_diff'] and
              rt['emotion']['p_value'] < 0.05 and rt['sex']['p_value'] < 0.05 and
              rt['task_difference']['p_value'] < 0.05)
    except Exception:
        ok = False
    checks.append(('behavior_rt', ok))
    try:
        mad = sub['behavior']['mad']
        ref_mad = ref['behavior']['mad']
        ok = (close(mad['emotion']['mean_diff'], ref_mad['emotion']['mean_diff'], 0.03) and
              close(mad['sex']['mean_diff'], ref_mad['sex']['mean_diff'], 0.03) and
              close(mad['task_difference']['mean_diff'], ref_mad['task_difference']['mean_diff'], 0.03) and
              mad['emotion']['mean_diff'] > mad['sex']['mean_diff'] and
              mad['emotion']['p_value'] < 0.05 and mad['sex']['p_value'] < 0.05 and
              mad['task_difference']['p_value'] < 0.05)
    except Exception:
        ok = False
    checks.append(('behavior_mad', ok))
    try:
        rsa = sub['rsa']
        ref_rsa = ref['rsa']
        ok = (close(rsa['emotion']['peak_amplitude'], ref_rsa['emotion']['peak_amplitude'], 0.5) and
              close(rsa['sex']['peak_amplitude'], ref_rsa['sex']['peak_amplitude'], 0.5) and
              rsa['emotion']['peak_amplitude'] > rsa['sex']['peak_amplitude'] and
              100 <= rsa['emotion']['peak_latency_ms'] <= 400 and
              100 <= rsa['sex']['peak_latency_ms'] <= 400)
    except Exception:
        ok = False
    checks.append(('rsa_amplitude', ok))
    try:
        rsa = sub['rsa']
        ref_rsa = ref['rsa']
        ok = (close(rsa['emotion']['peak_latency_ms'], ref_rsa['emotion']['peak_latency_ms'], 30.0) and
              close(rsa['sex']['peak_latency_ms'], ref_rsa['sex']['peak_latency_ms'], 30.0) and
              rsa['emotion']['peak_latency_ms'] < rsa['sex']['peak_latency_ms'])
    except Exception:
        ok = False
    checks.append(('rsa_latency', ok))
    try:
        dec = sub['decoding']
        ref_dec = ref['decoding']
        ok = (close(dec['emotion']['peak_accuracy'], ref_dec['emotion']['peak_accuracy'], 0.1) and
              close(dec['sex']['peak_accuracy'], ref_dec['sex']['peak_accuracy'], 0.1) and
              dec['emotion']['peak_accuracy'] > dec['sex']['peak_accuracy'] and
              dec['emotion']['peak_accuracy'] > 0.6 and dec['sex']['peak_accuracy'] > 0.6)
    except Exception:
        ok = False
    checks.append(('decoding_accuracy', ok))
    try:
        dec = sub['decoding']
        ref_dec = ref['decoding']
        ok = (close(dec['emotion']['peak_latency_ms'], ref_dec['emotion']['peak_latency_ms'], 30.0) and
              close(dec['sex']['peak_latency_ms'], ref_dec['sex']['peak_latency_ms'], 30.0) and
              dec['emotion']['peak_latency_ms'] < dec['sex']['peak_latency_ms'])
    except Exception:
        ok = False
    checks.append(('decoding_latency', ok))
    try:
        fus = sub['fusion']
        ref_fus = ref['fusion']
        ok = (close(fus['emotion']['mean_r'], ref_fus['emotion']['mean_r'], 0.1) and
              close(fus['sex']['mean_r'], ref_fus['sex']['mean_r'], 0.1) and
              close(fus['difference'], ref_fus['difference'], 0.1) and
              fus['emotion']['mean_r'] > fus['sex']['mean_r'] and
              fus['emotion']['mean_r'] > 0.15 and
              fus['difference'] > 0.05)
    except Exception:
        ok = False
    checks.append(('fusion', ok))
    try:
        pvals = [
            sub['behavior']['rt']['emotion']['p_value'],
            sub['behavior']['rt']['sex']['p_value'],
            sub['behavior']['rt']['task_difference']['p_value'],
            sub['behavior']['mad']['emotion']['p_value'],
            sub['behavior']['mad']['sex']['p_value'],
            sub['behavior']['mad']['task_difference']['p_value'],
        ]
        ok = all(0 <= p <= 1 for p in pvals)
    except Exception:
        ok = False
    checks.append(('p_value_range', ok))
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    score = passed / total if total else 0.0
    report = {
        'score': float(score),
        'passed': passed,
        'total': total,
        'checks': [{'name': name, 'passed': bool(ok)} for name, ok in checks],
        'reference': ref,
        'submission': sub
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    with open(REWARD, 'w') as f:
        json.dump({'score': float(score), 'reward': float(score)}, f)
    print(json.dumps(report, indent=2))

def write_reward(score):
    os.makedirs(os.path.dirname(REWARD), exist_ok=True)
    with open(REWARD, 'w') as f:
        json.dump({'score': float(score), 'reward': float(score)}, f)

if __name__ == '__main__':
    main()
