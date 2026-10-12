import os, numpy as np, MDAnalysis as mda

def ensure_dirs():
    for sys in ['idp/standard', 'idp/light', 'bilayer/standard', 'bilayer/light']:
        os.makedirs(f'/app/data/{sys}', exist_ok=True)

def write_itp(path, mass):
    with open(path, 'w') as f:
        f.write('[ atoms ]\n')
        f.write('; nr type resnr residue atom cgnr charge mass\n')
        f.write(f'1 W 1 W W 1 0.0 {mass}\n')

def write_mdp(path):
    with open(path, 'w') as f:
        f.write('dt = 0.02\n')
        f.write('nsteps = 10000000\n')
        f.write('nstxout-compressed = 5000\n')

def write_readme():
    content = '''# Light Martini Water Analysis

This dataset contains coarse-grained Martini MD trajectories for:
- an intrinsically disordered protein (IDP) chain (20 BB beads)
- a POPC lipid bilayer (512 lipids, 3 beads each: P, C1, C2)

Each system was simulated with standard water (72 amu) and light water (20 amu).
Trajectory frame spacing is given by dt * nstxout-compressed in md.mdp.
The IDP bond equilibrium length is not required for the task.

Headgroup bead for lipid diffusion is named 'P'.

## Files
- idp/standard/{topol.gro, traj.xtc, water.itp, md.mdp}
- idp/light/...
- bilayer/standard/...
- bilayer/light/...

## Task
See the main instruction.
'''
    with open('/app/data/README.md', 'w') as f:
        f.write(content)

def generate_idp_frames(seed, tau_ns, n_frames=2000, n_beads=20, bond=3.8):
    rng = np.random.default_rng(seed)
    dt = 0.1
    mean = 18.0
    sigma = 2.5
    R = np.zeros(n_frames)
    R[0] = mean + sigma * rng.normal()
    alpha = np.exp(-dt / tau_ns)
    for i in range(1, n_frames):
        R[i] = mean + alpha * (R[i-1] - mean) + sigma * np.sqrt(1 - alpha**2) * rng.normal()
    R = np.clip(R, 12.0, 26.0)
    pool = []
    for _ in range(20000):
        coords = np.zeros((n_beads, 3))
        for i in range(1, n_beads):
            vec = rng.normal(size=3)
            vec /= np.linalg.norm(vec)
            coords[i] = coords[i-1] + bond * vec
        R0 = np.linalg.norm(coords[-1] - coords[0])
        if R0 < 5.0:
            continue
        dists = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=2)
        for i in range(n_beads):
            for j in range(max(0,i-1), min(n_beads,i+2)):
                dists[i,j] = np.inf
        min_d = np.min(dists)
        pool.append((R0, min_d, coords))
    frames = []
    for target_R in R:
        best = None
        best_diff = np.inf
        for R0, min_d, coords in pool:
            scale = target_R / R0
            if 0.8 < scale < 1.5 and min_d * scale > 4.0:
                diff = abs(scale - 1.0)
                if diff < best_diff:
                    best = coords * scale
                    best_diff = diff
        if best is None:
            R0, min_d, coords = pool[0]
            best = coords * (target_R / R0)
        frames.append(best)
    return np.array(frames)

def write_idp_traj(base, frames, box_size=1000.0):
    n_frames, n_atoms, _ = frames.shape
    u = mda.Universe.empty(n_atoms, n_residues=n_atoms,
                           atom_resindex=np.arange(n_atoms),
                           residue_segindex=np.zeros(n_atoms, dtype=int),
                           trajectory=True)
    u.add_TopologyAttr('name', ['BB'] * n_atoms)
    u.add_TopologyAttr('resname', ['ALA'] * n_atoms)
    u.add_TopologyAttr('resid', np.arange(1, n_atoms+1))
    u.add_TopologyAttr('segid', ['A'] * n_atoms)
    u.dimensions = np.array([box_size, box_size, box_size, 90, 90, 90], dtype=np.float32)
    u.atoms.positions = frames[0]
    u.atoms.write(os.path.join(base, 'topol.gro'))
    with mda.Writer(os.path.join(base, 'traj.xtc'), n_atoms) as W:
        for i in range(n_frames):
            u.atoms.positions = frames[i]
            W.write(u.atoms)

def generate_bilayer_frames(seed, D, n_frames=2000, n_lipids=512, box=400.0):
    rng = np.random.default_rng(seed)
    dt = 100.0
    n_per_side = int(np.ceil(np.sqrt(n_lipids)))
    spacing = box / n_per_side
    coords = []
    for i in range(n_per_side):
        for j in range(n_per_side):
            if len(coords) >= n_lipids:
                break
            x = i * spacing + rng.uniform(-2, 2)
            y = j * spacing + rng.uniform(-2, 2)
            coords.append([x, y])
    coords = np.array(coords)
    head_pos = coords
    frames = np.zeros((n_frames, n_lipids*3, 3))
    step = np.sqrt(2 * D * dt)
    for t in range(n_frames):
        if t > 0:
            head_pos += step * rng.normal(size=(n_lipids, 2))
            head_pos = head_pos % box
        p = np.zeros((n_lipids, 3))
        p[:, :2] = head_pos
        p[:, 2] = 50.0
        c1 = p.copy()
        c1[:, 2] = 45.0
        c1[:, :2] += rng.normal(0, 0.5, size=(n_lipids, 2))
        c2 = p.copy()
        c2[:, 2] = 40.0
        c2[:, :2] += rng.normal(0, 0.5, size=(n_lipids, 2))
        frame = np.zeros((n_lipids*3, 3))
        frame[0::3] = p
        frame[1::3] = c1
        frame[2::3] = c2
        frames[t] = frame
    return frames, box

def write_bilayer_traj(base, frames, box):
    n_frames, n_atoms, _ = frames.shape
    n_lipids = n_atoms // 3
    u = mda.Universe.empty(n_atoms, n_residues=n_lipids,
                           atom_resindex=np.repeat(np.arange(n_lipids), 3),
                           residue_segindex=np.zeros(n_lipids, dtype=int),
                           trajectory=True)
    names = []
    resnames = []
    resids = []
    for i in range(n_lipids):
        names.extend(['P', 'C1', 'C2'])
        resnames.extend(['POPC']*3)
        resids.extend([i+1]*3)
    u.add_TopologyAttr('name', names)
    u.add_TopologyAttr('resname', resnames)
    u.add_TopologyAttr('resid', resids)
    u.add_TopologyAttr('segid', ['MEMB']*n_atoms)
    u.dimensions = np.array([box, box, 100.0, 90, 90, 90], dtype=np.float32)
    u.atoms.positions = frames[0]
    u.atoms.write(os.path.join(base, 'topol.gro'))
    with mda.Writer(os.path.join(base, 'traj.xtc'), n_atoms) as W:
        for i in range(n_frames):
            u.atoms.positions = frames[i]
            W.write(u.atoms)

def main():
    ensure_dirs()
    write_readme()
    frames_std = generate_idp_frames(seed=42, tau_ns=20.0)
    write_idp_traj('/app/data/idp/standard', frames_std)
    write_itp('/app/data/idp/standard/water.itp', 72.0)
    write_mdp('/app/data/idp/standard/md.mdp')
    frames_light = generate_idp_frames(seed=43, tau_ns=8.0)
    write_idp_traj('/app/data/idp/light', frames_light)
    write_itp('/app/data/idp/light/water.itp', 20.0)
    write_mdp('/app/data/idp/light/md.mdp')
    frames_bil_std, box = generate_bilayer_frames(seed=44, D=0.001)
    write_bilayer_traj('/app/data/bilayer/standard', frames_bil_std, box)
    write_itp('/app/data/bilayer/standard/water.itp', 72.0)
    write_mdp('/app/data/bilayer/standard/md.mdp')
    frames_bil_light, box = generate_bilayer_frames(seed=45, D=0.00116)
    write_bilayer_traj('/app/data/bilayer/light', frames_bil_light, box)
    write_itp('/app/data/bilayer/light/water.itp', 20.0)
    write_mdp('/app/data/bilayer/light/md.mdp')

if __name__ == '__main__':
    main()
