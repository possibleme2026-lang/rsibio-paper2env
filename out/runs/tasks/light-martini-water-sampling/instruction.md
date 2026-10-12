# Light Martini Water Sampling Analysis

You are provided with coarse-grained MD trajectories from Martini simulations of an intrinsically disordered protein (IDP) and a POPC lipid bilayer, each simulated with standard water (72 amu) and light water (20 amu). Your task is to implement a validation pipeline to quantify sampling speedup and diffusion enhancement, and decide if light water meets acceptance criteria.

## Inputs
- `/app/data/README.md`
- `/app/data/idp/standard/{topol.gro,traj.xtc,water.itp,md.mdp}`
- `/app/data/idp/light/{topol.gro,traj.xtc,water.itp,md.mdp}`
- `/app/data/bilayer/standard/{topol.gro,traj.xtc,water.itp,md.mdp}`
- `/app/data/bilayer/light/{topol.gro,traj.xtc,water.itp,md.mdp}`

## Task
1. Parse `water.itp` and `md.mdp` to extract water bead masses and timestep. Compute frame time as `dt * nstxout-compressed`.
2. For IDP trajectories, compute end-to-end distance R(t) (distance between first and last BB bead). Compute autocorrelation function, fit a single exponential to ACF up to the first lag where ACF < 0.05. Report tau_std and tau_light in ns, and speedup = tau_std / tau_light.
3. For bilayer trajectories, select headgroup bead `P`. Unwrap positions with PBC, compute 2D MSD, fit linear region between 10 ns and 100 ns to MSD = 4 D t. Report D_std and D_light in Å²/ps, and enhancement = (D_light - D_std)/D_std * 100.
4. For the light IDP trajectory, compute the minimum distance between any two non-bonded BB beads (index difference > 1) over all frames. Report in nm.
5. Decide if light water is recommended: true if speedup >= 1.5, diffusion enhancement >= 10%, and min non-bonded distance > 0.4 nm.

## Constraints
1. JSON must be at `/app/answer.json` and contain all required keys (see Output).
2. `water_mass_standard_amu` = 72.0, `water_mass_light_amu` = 20.0, `timestep_fs` = 20.0, `frame_time_ps` = 100.0 (within 1% tolerance).
3. IDP speedup ratio must be within 15% of the value recomputed by the grader from the trajectories, and >= 1.5.
4. Lipid diffusion enhancement must be within 20% of the value recomputed by the grader, and >= 10.0%.
5. `min_nonbonded_distance_light_nm` must be > 0.4 and within 0.1 nm of the value recomputed by the grader.
6. `light_water_recommended` must match the boolean condition in Task step 5 using your reported values.

## Output
Write a JSON file to `/app/answer.json` with the following keys and example values:
```json
{
  "water_mass_standard_amu": 72.0,
  "water_mass_light_amu": 20.0,
  "timestep_fs": 20.0,
  "frame_time_ps": 100.0,
  "idp_tau_standard_ns": 19.5,
  "idp_tau_light_ns": 7.8,
  "idp_speedup_ratio": 2.5,
  "lipid_D_standard_A2_per_ps": 0.001,
  "lipid_D_light_A2_per_ps": 0.00116,
  "lipid_diffusion_enhancement_percent": 16.0,
  "min_nonbonded_distance_light_nm": 0.45,
  "light_water_recommended": true
}
```

## Scoring
6 constraints equally weighted. Grader recomputes gold from trajectories and checks each constraint. Score = fraction of constraints passed.
