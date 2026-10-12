#!/bin/bash
set -euo pipefail
python - <<'PY'
import json, numpy as np, MDAnalysis as mda
from scipy.optimize import curve_fit

DATA = '/app/data'
ans = {}

def parse_mdp(path):
    dt = nst = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('dt'):
                dt = float(line.split('=')[1].strip())
            if line.startswith('nstxout-compressed'):
                nst = int(line.split('=')[1].strip())
    return dt, nst

def get_water_mass(path):
    with open(path) as f:
        for line in f:
            if line.strip().startswith('1 W'):
                return float(line.split()[6])
    return None

ans['water_mass_standard_amu'] = get_water_mass(f'{DATA}/idp/standard/water.itp')
ans['water_mass_light_amu'] = get_water_mass(f'{DATA}/idp/light/water.itp')
dt, nst = parse_mdp(f'{DATA}/idp/standard/md.mdp')
ans['timestep_fs'] = dt * 1000.0
ans['frame_time_ps'] = dt * nst
frame_time_ns = ans['frame_time_ps'] / 1000.0

def compute_tau(traj, top):
    u = mda.Universe(top, traj)
    n = len(u.trajectory)
    R = np.zeros(n)
    for i, ts in enumerate(u.trajectory):
        R[i] = np.linalg.norm(u.atoms.positions[-1] - u.atoms.positions[0])
    R -= R.mean()
    acf = np.correlate(R, R, mode='full')[n-1:]
    acf /= acf[0]
    idx = np.where(acf < 0.05)[0]
    max_lag = idx[0] if len(idx) > 0 else n-1
    if max_lag < 3:
        max_lag = min(10, n-1)
    lags = np.arange(max_lag+1) * frame_time_ns
    y = acf[:max_lag+1]
    def exp_func(t, tau):
        return np.exp(-t/tau)
    try:
        popt, _ = curve_fit(exp_func, lags[1:], y[1:], p0=[10.0], maxfev=10000)
        return popt[0]
    except Exception:
        mask = y > 0
        slope, _ = np.polyfit(lags[mask], np.log(y[mask]), 1)
        return -1.0/slope

tau_std = compute_tau(f'{DATA}/idp/standard/traj.xtc', f'{DATA}/idp/standard/topol.gro')
tau_light = compute_tau(f'{DATA}/idp/light/traj.xtc', f'{DATA}/idp/light/topol.gro')
ans['idp_tau_standard_ns'] = tau_std
ans['idp_tau_light_ns'] = tau_light
ans['idp_speedup_ratio'] = tau_std / tau_light

def unwrap(pos, box):
    n, a, _ = pos.shape
    unw = np.zeros_like(pos)
    unw[0] = pos[0]
    for t in range(1, n):
        d = pos[t] - pos[t-1]
        d = d - np.round(d / box) * box
        unw[t] = unw[t-1] + d
    return unw

def compute_D(traj, top):
    u = mda.Universe(top, traj)
    sel = u.select_atoms('name P')
    n = len(u.trajectory)
    pos = np.zeros((n, len(sel), 3))
    for i, ts in enumerate(u.trajectory):
        pos[i] = sel.positions
    box = np.array([u.dimensions[0], u.dimensions[1], 0.0]) if u.dimensions is not None else np.array([400.0, 400.0, 0.0])
    unw = unwrap(pos, box)
    msd = np.zeros(n)
    for lag in range(n):
        disp = unw[lag:] - unw[:-lag]
        msd[lag] = np.mean(np.sum(disp[:,:,:2]**2, axis=2))
    t = np.arange(n) * ans['frame_time_ps']
    mask = (t >= 10000) & (t <= 100000)
    if mask.sum() < 2:
        mask = (t >= 1000) & (t <= 50000)
    coeff = np.polyfit(t[mask], msd[mask], 1)
    return coeff[0] / 4.0

D_std = compute_D(f'{DATA}/bilayer/standard/traj.xtc', f'{DATA}/bilayer/standard/topol.gro')
D_light = compute_D(f'{DATA}/bilayer/light/traj.xtc', f'{DATA}/bilayer/light/topol.gro')
ans['lipid_D_standard_A2_per_ps'] = D_std
ans['lipid_D_light_A2_per_ps'] = D_light
ans['lipid_diffusion_enhancement_percent'] = (D_light - D_std) / D_std * 100.0

def min_dist(traj, top):
    u = mda.Universe(top, traj)
    n_atoms = len(u.atoms)
    min_d = np.inf
    for ts in u.trajectory:
        pos = u.atoms.positions
        for i in range(n_atoms):
            for j in range(i+2, n_atoms):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    min_d = d
    return min_d / 10.0

ans['min_nonbonded_distance_light_nm'] = min_dist(f'{DATA}/idp/light/traj.xtc', f'{DATA}/idp/light/topol.gro')

ans['light_water_recommended'] = (ans['idp_speedup_ratio'] >= 1.5) and (ans['lipid_diffusion_enhancement_percent'] >= 10.0) and (ans['min_nonbonded_distance_light_nm'] > 0.4)

with open('/app/answer.json', 'w') as f:
    json.dump(ans, f, indent=2)
PY
