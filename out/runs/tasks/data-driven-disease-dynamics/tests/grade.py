import json, os, sys, numpy as np, pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score

def compute_gold():
    df = pd.read_csv('/app/data/longitudinal_biomarkers.csv')
    biomarkers = [f'biomarker_{i}' for i in range(1, 13)]
    dt = 0.1
    t_max = 12.0
    time_grid = np.arange(0, t_max + dt/2, dt)
    interp_data = {}
    for pid in df['patient_id'].unique():
        pdata = df[df['patient_id'] == pid].sort_values('visit_time')
        if len(pdata) < 4:
            continue
        times = pdata['visit_time'].values
        t_min, t_max_p = times.min(), times.max()
        valid_grid = time_grid[(time_grid >= t_min) & (time_grid <= t_max_p)]
        if len(valid_grid) < 2:
            continue
        interp_data[pid] = {'times': valid_grid}
        for b in biomarkers:
            vals = pdata[b].values
            interp_vals = np.interp(valid_grid, times, vals)
            interp_data[pid][b] = interp_vals
    all_samples = []
    for pid, d in interp_data.items():
        for b in biomarkers:
            all_samples.append(d[b])
    # Per-biomarker pooled statistics (population, ddof=0), as the
    # instruction requires: pool every interpolated sample of a
    # biomarker across all patients. ``all_samples`` is ordered
    # patient-major / biomarker-minor, so one biomarker's samples
    # sit at a constant stride of len(biomarkers). A previous fix
    # padded the ragged arrays into a matrix and reduced over
    # axis=0, which yields a per-TIME-COLUMN statistic (length
    # max(len(series))) instead of a per-biomarker one -- that
    # made ``mean`` shape (121,) while X is (n, 12), so the
    # broadcast in ``(X - mean) / std`` failed.
    _n_b = len(biomarkers)
    mean = np.zeros(_n_b)
    std = np.zeros(_n_b)
    for _i in range(_n_b):
        _pool = np.concatenate([np.atleast_1d(np.asarray(a, dtype=float))
                                for a in all_samples[_i::_n_b]])
        mean[_i] = _pool.mean()
        std[_i] = _pool.std()
        if std[_i] == 0:
            std[_i] = 1.0
    for pid, d in interp_data.items():
        for i, b in enumerate(biomarkers):
            d[b] = (d[b] - mean[i]) / std[i]
    tau = 1.0
    lag_steps = int(round(tau / dt))
    C0 = np.zeros((12, 12))
    Ctau = np.zeros((12, 12))
    N_pairs = 0
    for pid, d in interp_data.items():
        n = len(d['times'])
        if n <= lag_steps:
            continue
        X = np.array([d[b] for b in biomarkers]).T
        X_t = X[:-lag_steps]
        X_tau = X[lag_steps:]
        C0 += X_t.T @ X_t
        Ctau += X_t.T @ X_tau
        N_pairs += len(X_t)
    C0 /= N_pairs
    Ctau /= N_pairs
    eigvals, eigvecs = eigh(Ctau, C0)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    v1 = eigvecs[:, 0]
    v1 = v1 / np.sqrt(v1 @ C0 @ v1)
    max_idx = np.argmax(np.abs(v1))
    if v1[max_idx] < 0:
        v1 = -v1
    lambda1 = eigvals[0]
    lambda2 = eigvals[1] if len(eigvals) > 1 else 0.0
    S = (lambda1 - lambda2) / lambda1 if lambda1 > 0 else 0.0
    df['rc'] = np.nan
    for pid in df['patient_id'].unique():
        pdata = df[df['patient_id'] == pid]
        X = pdata[biomarkers].values
        X_std = (X - mean) / std
        rc_vals = X_std @ v1
        df.loc[df['patient_id'] == pid, 'rc'] = rc_vals
    rc_all = df['rc'].dropna().values
    p1, p99 = np.percentile(rc_all, [1, 99])
    bins = np.linspace(p1, p99, 21)
    counts, bin_edges = np.histogram(rc_all, bins=bins)
    bin_width = bin_edges[1] - bin_edges[0]
    p = counts / (len(rc_all) * bin_width)
    p = np.maximum(p, 1e-12)
    F = -np.log(p)
    F = F - np.min(F)
    msd = []
    for pid, d in interp_data.items():
        X = np.array([d[b] for b in biomarkers]).T
        rc_grid = X @ v1
        if len(rc_grid) > 1:
            diff = np.diff(rc_grid)**2
            msd.extend(diff)
    msd = np.mean(msd)
    D = msd / (2 * dt)
    valid = df['rc'].notna() & df['diagnosis'].notna()
    auc = roc_auc_score(df.loc[valid, 'diagnosis'], df.loc[valid, 'rc'])
    return {
        'rc_vector': v1,
        'free_energy': F,
        'diffusion_coefficient': float(D),
        'validation_score': float(S),
        'auc': float(auc),
        'C0': C0
    }

def main():
    gold = compute_gold()
    report = {}
    score = 0.0
    total_constraints = 8
    passed = 0
    submission_path = '/app/submission.json'
    if not os.path.exists(submission_path):
        report['constraint_1'] = {'passed': False, 'reason': 'submission.json not found'}
        write_result(0.0, report)
        return
    try:
        with open(submission_path) as f:
            sub = json.load(f)
    except Exception as e:
        report['constraint_1'] = {'passed': False, 'reason': f'Invalid JSON: {e}'}
        write_result(0.0, report)
        return
    required_keys = ['rc_vector', 'free_energy', 'diffusion_coefficient', 'validation_score', 'auc']
    if not all(k in sub for k in required_keys):
        report['constraint_1'] = {'passed': False, 'reason': f'Missing keys. Found: {list(sub.keys())}'}
        write_result(0.0, report)
        return
    report['constraint_1'] = {'passed': True}
    passed += 1
    rc = np.array(sub['rc_vector'])
    if len(rc) != 12:
        report['constraint_2'] = {'passed': False, 'reason': f'rc_vector length {len(rc)} != 12'}
    else:
        C0 = gold['C0']
        norm_sq = rc @ C0 @ rc
        sign_ok = rc[np.argmax(np.abs(rc))] > 0
        if abs(norm_sq - 1.0) < 0.05 and sign_ok:
            report['constraint_2'] = {'passed': True, 'norm_sq': float(norm_sq)}
            passed += 1
        else:
            report['constraint_2'] = {'passed': False, 'reason': f'norm_sq={norm_sq:.4f}, sign_ok={sign_ok}'}
    if len(rc) == 12:
        gold_v = gold['rc_vector']
        C0 = gold['C0']
        num = abs(rc @ C0 @ gold_v)
        den = np.sqrt((rc @ C0 @ rc) * (gold_v @ C0 @ gold_v))
        cos_sim = num / den if den > 0 else 0
        if cos_sim > 0.99:
            report['constraint_3'] = {'passed': True, 'cos_sim': float(cos_sim)}
            passed += 1
        else:
            report['constraint_3'] = {'passed': False, 'cos_sim': float(cos_sim)}
    else:
        report['constraint_3'] = {'passed': False, 'reason': 'rc_vector length mismatch'}
    fe = np.array(sub['free_energy'])
    if len(fe) != 20:
        report['constraint_4'] = {'passed': False, 'reason': f'free_energy length {len(fe)} != 20'}
    else:
        min_fe = np.min(fe)
        if abs(min_fe) < 1e-3:
            report['constraint_4'] = {'passed': True, 'min_fe': float(min_fe)}
            passed += 1
        else:
            report['constraint_4'] = {'passed': False, 'reason': f'min_fe={min_fe}'}
    gold_fe = gold['free_energy']
    if len(fe) == 20:
        mae = np.mean(np.abs(fe - gold_fe))
        corr = np.corrcoef(fe, gold_fe)[0, 1] if np.std(fe) > 0 and np.std(gold_fe) > 0 else 0
        if mae < 0.2 and corr > 0.95:
            report['constraint_5'] = {'passed': True, 'mae': float(mae), 'corr': float(corr)}
            passed += 1
        else:
            report['constraint_5'] = {'passed': False, 'mae': float(mae), 'corr': float(corr)}
    else:
        report['constraint_5'] = {'passed': False, 'reason': 'free_energy length mismatch'}
    try:
        D = float(sub['diffusion_coefficient'])
        gold_D = gold['diffusion_coefficient']
        rel_err = abs(D - gold_D) / abs(gold_D) if gold_D != 0 else abs(D)
        if rel_err < 0.05 and D > 0:
            report['constraint_6'] = {'passed': True, 'D': D, 'gold_D': gold_D, 'rel_err': float(rel_err)}
            passed += 1
        else:
            report['constraint_6'] = {'passed': False, 'D': D, 'gold_D': gold_D, 'rel_err': float(rel_err)}
    except Exception as e:
        report['constraint_6'] = {'passed': False, 'reason': str(e)}
    try:
        S = float(sub['validation_score'])
        gold_S = gold['validation_score']
        if abs(S - gold_S) < 0.01 and 0 <= S <= 1:
            report['constraint_7'] = {'passed': True, 'S': S, 'gold_S': gold_S}
            passed += 1
        else:
            report['constraint_7'] = {'passed': False, 'S': S, 'gold_S': gold_S}
    except Exception as e:
        report['constraint_7'] = {'passed': False, 'reason': str(e)}
    try:
        auc = float(sub['auc'])
        gold_auc = gold['auc']
        if abs(auc - gold_auc) < 0.01 and 0 <= auc <= 1:
            report['constraint_8'] = {'passed': True, 'auc': auc, 'gold_auc': gold_auc}
            passed += 1
        else:
            report['constraint_8'] = {'passed': False, 'auc': auc, 'gold_auc': gold_auc}
    except Exception as e:
        report['constraint_8'] = {'passed': False, 'reason': str(e)}
    score = passed / total_constraints
    report['score'] = score
    report['passed'] = passed
    report['total'] = total_constraints
    write_result(score, report)

def write_result(score, report):
    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump({'score': score, 'reward': score}, f)
    with open('/logs/verifier/report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
