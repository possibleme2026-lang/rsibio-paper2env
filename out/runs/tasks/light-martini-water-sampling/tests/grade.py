import json, os, sys, math
import numpy as np
import MDAnalysis as mda
from scipy.optimize import curve_fit

DATA = '/app/data'
ANSWER = '/app/answer.json'
REWARD = '/logs/verifier/reward.json'
REPORT = '/logs/verifier/report.json'

def load_answer():
    if not os.path.exists(ANSWER):
        return None
    try:
        with open(ANSWER) as f:
            return json.load(f)
    except Exception:
        return None

def parse_mdp(path):
    dt = None
    nst = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('dt'):
                dt = float(line.split('=')[1].strip())
            if line.startswith('nstxout-compressed'):
                nst = int(line.split('=')[1].strip())
    return dt, nst

def get_water_mass(itp_path):
    with open(itp_path) as f:
        for line in f:
            if line.strip().startswith('1 W'):
                parts = line.split()
                if len(parts) >= 7:
                    return float(parts[6])
    return None

def compute_idp_tau(traj_path, top_path, frame_time_ns):
    u = mda.Universe(top_path, traj_path)
    n_frames = len(u.trajectory)
    R = np.zeros(n_frames)
    for i, ts in enumerate(u.trajectory):
        pos = u.atoms.positions
        R[i] = np.linalg.norm(pos[-1] - pos[0])
    R = R - np.mean(R)
    acf = np.correlate(R, R, mode='full')[n_frames-1:]
    acf = acf / acf[0]
    idx = np.where(acf < 0.05)[0]
    if len(idx) == 0:
        max_lag = n_frames - 1
    else:
        max_lag = idx[0]
    if max_lag < 3:
        max_lag = min(10, n_frames-1)
    lags = np.arange(max_lag + 1) * frame_time_ns
    y = acf[:max_lag+1]
    def exp_func(t, tau):
        return np.exp(-t / tau)
    try:
        popt, _ = curve_fit(exp_func, lags[1:], y[1:], p0=[10.0], maxfev=10000)
        tau = popt[0]
    except Exception:
        mask = y > 0
        if np.sum(mask) > 2:
            slope, _ = np.polyfit(lags[mask], np.log(y[mask]), 1)
            tau = -1.0 / slope
        else:
            tau = np.nan
    return tau

def unwrap_positions(pos, box):
    n_frames, n_atoms, _ = pos.shape
    unwrapped = np.zeros_like(pos)
    unwrapped[0] = pos[0]
    for t in range(1, n_frames):
        d = pos[t] - pos[t-1]
        d = d - np.round(d / box) * box
        unwrapped[t] = unwrapped[t-1] + d
    return unwrapped

def compute_lipid_D(traj_path, top_path, frame_time_ps):
    u = mda.Universe(top_path, traj_path)
    sel = u.select_atoms('name P')
    n_frames = len(u.trajectory)
    n_atoms = len(sel)
    pos = np.zeros((n_frames, n_atoms, 3))
    for i, ts in enumerate(u.trajectory):
        pos[i] = sel.positions
    if u.dimensions is not None:
        box = np.array([u.dimensions[0], u.dimensions[1], 0.0])
    else:
        box = np.array([400.0, 400.0, 0.0])
    unwrapped = unwrap_positions(pos, box)
    msd = np.zeros(n_frames)
    for lag in range(n_frames):
        disp = unwrapped[lag:] - unwrapped[:-lag]
        msd[lag] = np.mean(np.sum(disp[:,:,:2]**2, axis=2))
    t = np.arange(n_frames) * frame_time_ps
    mask = (t >= 10000) & (t <= 100000)
    if np.sum(mask) < 2:
        mask = (t >= 1000) & (t <= 50000)
    coeff = np.polyfit(t[mask], msd[mask], 1)
    D = coeff[0] / 4.0
    return D

def compute_min_nonbonded_distance(traj_path, top_path):
    u = mda.Universe(top_path, traj_path)
    n_atoms = len(u.atoms)
    min_dist = np.inf
    for ts in u.trajectory:
        pos = u.atoms.positions
        for i in range(n_atoms):
            for j in range(i+2, n_atoms):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
    return min_dist / 10.0

def main():
    ans = load_answer()
    if ans is None:
        score = 0.0
        report = {'error': 'missing or invalid answer.json'}
        os.makedirs(os.path.dirname(REWARD), exist_ok=True)
        with open(REWARD, 'w') as f:
            json.dump({'score': score, 'reward': score}, f)
        with open(REPORT, 'w') as f:
            json.dump(report, f)
        print(json.dumps(report))
        return

    idp_std = os.path.join(DATA, 'idp/standard')
    idp_light = os.path.join(DATA, 'idp/light')
    bil_std = os.path.join(DATA, 'bilayer/standard')
    bil_light = os.path.join(DATA, 'bilayer/light')

    mass_std = get_water_mass(os.path.join(idp_std, 'water.itp'))
    mass_light = get_water_mass(os.path.join(idp_light, 'water.itp'))
    dt, nst = parse_mdp(os.path.join(idp_std, 'md.mdp'))
    timestep_fs = dt * 1000.0
    frame_time_ps = dt * nst

    tau_std = compute_idp_tau(os.path.join(idp_std, 'traj.xtc'), os.path.join(idp_std, 'topol.gro'), frame_time_ps / 1000.0)
    tau_light = compute_idp_tau(os.path.join(idp_light, 'traj.xtc'), os.path.join(idp_light, 'topol.gro'), frame_time_ps / 1000.0)
    speedup = tau_std / tau_light if tau_light > 0 else 0.0

    D_std = compute_lipid_D(os.path.join(bil_std, 'traj.xtc'), os.path.join(bil_std, 'topol.gro'), frame_time_ps)
    D_light = compute_lipid_D(os.path.join(bil_light, 'traj.xtc'), os.path.join(bil_light, 'topol.gro'), frame_time_ps)
    enhancement = (D_light - D_std) / D_std * 100.0 if D_std > 0 else 0.0

    min_dist = compute_min_nonbonded_distance(os.path.join(idp_light, 'traj.xtc'), os.path.join(idp_light, 'topol.gro'))

    gold_decision = (speedup >= 1.5) and (enhancement >= 10.0) and (min_dist > 0.4)

    checks = {}
    required_keys = ['water_mass_standard_amu', 'water_mass_light_amu', 'timestep_fs', 'frame_time_ps',
                     'idp_tau_standard_ns', 'idp_tau_light_ns', 'idp_speedup_ratio',
                     'lipid_D_standard_A2_per_ps', 'lipid_D_light_A2_per_ps',
                     'lipid_diffusion_enhancement_percent', 'min_nonbonded_distance_light_nm',
                     'light_water_recommended']
    checks['json_keys'] = all(k in ans for k in required_keys)

    checks['masses_timestep'] = (
        abs(ans.get('water_mass_standard_amu', -1) - 72.0) < 0.1 and
        abs(ans.get('water_mass_light_amu', -1) - 20.0) < 0.1 and
        abs(ans.get('timestep_fs', -1) - 20.0) < 0.1 and
        abs(ans.get('frame_time_ps', -1) - 100.0) < 1.0
    )

    ans_speedup = ans.get('idp_speedup_ratio', -1)
    checks['idp_speedup'] = (ans_speedup >= 1.5) and (abs(ans_speedup - speedup) / speedup < 0.15)

    ans_enh = ans.get('lipid_diffusion_enhancement_percent', -1)
    checks['lipid_enhancement'] = (ans_enh >= 10.0) and (abs(ans_enh - enhancement) / max(enhancement, 1e-6) < 0.20)

    ans_min = ans.get('min_nonbonded_distance_light_nm', -1)
    checks['min_distance'] = (ans_min > 0.4) and (abs(ans_min - min_dist) < 0.1)

    ans_dec = ans.get('light_water_recommended', None)
    checks['decision'] = (ans_dec == gold_decision)

    total = len(checks)
    passed = sum(1 for v in checks.values() if v)
    score = passed / total

    report = {
        'checks': checks,
        'passed': passed,
        'total': total,
        'gold': {
            'mass_std': mass_std,
            'mass_light': mass_light,
            'timestep_fs': timestep_fs,
            'frame_time_ps': frame_time_ps,
            'tau_std_ns': tau_std,
            'tau_light_ns': tau_light,
            'speedup': speedup,
            'D_std': D_std,
            'D_light': D_light,
            'enhancement': enhancement,
            'min_dist_nm': min_dist,
            'decision': gold_decision
        },
        'answer': ans
    }

    os.makedirs(os.path.dirname(REWARD), exist_ok=True)
    with open(REWARD, 'w') as f:
        json.dump({'score': score, 'reward': score}, f)
    with open(REPORT, 'w') as f:
        json.dump(report, f)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
