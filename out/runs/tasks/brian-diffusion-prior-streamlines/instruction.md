# BRIAN: diffusion-informed streamline propagation of the MRI-visible glioblastoma core

You are re-implementing the core of the BRIAN simulator. In the original study, patient glioblastoma masks from UPENN-GBM are transferred onto healthy Human Connectome Project host brains through the MNI152 template, diffusion orientation distribution functions (dODFs) are reconstructed on each host, and stochastic streamline propagation (MRtrix3 iFOD2 style) produces volumetric occupancy maps that are fitted per tumour to the reference mask under a three-stage curriculum over a constrained five-metric objective. The fitted propagation is then compared against (a) a direction-blind isotropic null and (b) a single-tensor dODF that replaces the multi-shell SHORE dODF while the tracking framework is held fixed.

This task reproduces that workflow end-to-end on a reduced, fully synthetic cohort. Everything you need is already on disk; nothing must be downloaded.

## Inputs

All arrays live on a 40 x 40 x 40 grid with 2 mm isotropic voxels.

- `/app/data/sphere.npy` - float32 (60, 3) unit vectors on the sphere. **Use exactly these directions** whenever you need to sample a dODF.
- `/app/data/hosts/<host>/dwi.nii.gz` - float32 (40, 40, 40, 45) diffusion-weighted signal.
- `/app/data/hosts/<host>/bvals.txt` - 45 b-values in s/mm^2 (3 x b=0, 20 x b=1000, 22 x b=2000).
- `/app/data/hosts/<host>/bvecs.txt` - 45 unit gradient directions, zero rows where b=0.
- `/app/data/hosts/<host>/brain_mask.nii.gz` - uint8 brain mask; streamlines must remain inside it.
- `/app/data/hosts/<host>/wm_mask.nii.gz` - uint8 white-matter mask; streamlines terminate as soon as they leave it.
- `/app/data/tumours/<host>_<tumour>_ref.nii.gz` - uint8 reference mask (the MRI-visible tumour core) in host space. This is the fitting target.
- `/app/data/protocol.json` - parameter bounds, the curriculum schedule, the exact metric definitions and the objective weights. Read it; the grader uses the same file.

Hosts are `host01` and `host02`; tumours are `A`, `B`, `C`. The six evaluation pairings are listed in `protocol.json["pairings"]` (`host01_A`, `host01_B`, `host01_C`, `host02_A`, `host02_B`, `host02_C`).

## Task

1. **Reconstruct dODFs.** For each host, build a gradient table from `bvals.txt`/`bvecs.txt` and fit a multi-shell SHORE model (`dipy.reconst.shore.ShoreModel`, `radial_order=4`, `zeta=700`) to `dwi.nii.gz`. Evaluate the fitted ODF on the 60 directions of `sphere.npy`, clip negatives to zero and normalise each voxel to a probability distribution. Also fit a single-tensor model (`dipy.reconst.dti.TensorModel`) and evaluate its ODF on the same directions the same way.

2. **Implement stochastic streamline propagation.** Seeds are drawn uniformly (with replacement) from the voxels of `erode(ref_mask, 2 iterations, 6-connected structuring element) AND wm_mask`. The initial direction of a streamline is a direction sampled from the dODF at its seed voxel. Then, at each step: look up the dODF in the voxel containing the current position, restrict the 60 sphere directions to those within `angular_threshold` degrees of the previous direction, renormalise, and sample the next direction from that restricted distribution; advance the position by `step_size` voxels along the sampled direction. A streamline terminates when it leaves the WM mask or when the total traversed length exceeds `max_length`. The occupancy map is the set of voxels hit by `np.round(position)` at any time, including the seed voxels.

3. **Fit the four propagation parameters per pairing** (`step_size`, `angular_threshold`, `max_length`, `n_seeds`) with the three-stage curriculum given in `protocol.json` (stage 1: random search over the full bounds; stage 2: random search over bounds shrunk to `best * (1 +/- 0.35)` intersected with the global bounds; stage 3: the same with `best * (1 +/- 0.12)`). The stage objective is the weighted combination of the five metrics defined below, evaluated against that pairing's reference mask.

4. **Final run and metrics.** Re-run the propagation with the fitted parameters and a fixed RNG seed to produce the reported occupancy map and streamlines, then compute the five metrics for that pairing.

5. **Isotropic null.** With the *same* fitted parameters (`step_size`, `angular_threshold`, `max_length`, `n_seeds`) and the same seed voxels, replace the sampled dODF distribution by a uniform distribution over the sphere (still restricted by `angular_threshold`) and record the resulting occupancy map and metrics.

6. **Single-tensor dODF.** Repeat the directional propagation with the fitted parameters but using the single-tensor dODF instead of the SHORE dODF, and record the occupancy map and metrics.

7. **Write every artefact** described in `## Output`.

## Metric definitions (use exactly these)

For binary masks `A` (occupancy) and `B` (reference) on the same grid: `inter = |A AND B|`.

- `dice = 2 * inter / (|A| + |B|)`
- `precision = inter / |A|`
- `recall = inter / |B|`
- `surface(X) = X AND NOT erode(X, one iteration with the 6-connected structuring element)`
- `surface_dice`: let `S_A`, `S_B` be the two surfaces, `d_A` the Euclidean distance transform to `S_B` and `d_B` the distance transform to `S_A`; then `surface_dice = 0.5 * (mean(d_A[S_A] <= 2) + mean(d_B[S_B] <= 2))`
- `hausdorff_agreement = max(0, 1 - min(p95(concat(d_A[S_A], d_B[S_B])), 6) / 6)`, i.e. the 95th percentile of the combined surface distances, capped at 6 voxels.
- `objective = 0.30*dice + 0.20*hausdorff_agreement + 0.20*surface_dice + 0.15*precision + 0.15*recall`

All five metric values must lie in `[0, 1]`.

## Output

Write everything under `/app/output`.

- `/app/output/occupancy/<pairing_id>.nii.gz` - uint8 {0,1}, SHORE + directional occupancy.
- `/app/output/occupancy_iso/<pairing_id>.nii.gz` - uint8 {0,1}, isotropic null occupancy.
- `/app/output/occupancy_tensor/<pairing_id>.nii.gz` - uint8 {0,1}, single-tensor directional occupancy.
- `/app/output/streamlines/<pairing_id>.npz` - `points` float32 (M, 3) voxel coordinates concatenated over streamlines and `offsets` int32 (K+1,), so streamline `k` is `points[offsets[k]:offsets[k+1]]`. Only streamlines with at least 5 points need to be stored. Consecutive points must be exactly `step_size` apart (dead streamlines must be truncated, not padded).
- `/app/output/params.json` - `{"pairings": [{"pairing_id": "host01_A", "host": "host01", "step_size": 0.61, "angular_threshold": 24.8, "max_length": 17.6, "n_seeds": 512, "objective": 0.71}, ...]}`
- `/app/output/curriculum.json` - `{"pairings": [{"pairing_id": "host01_A", "stages": [{"stage": 1, "n_evals": 20, "bounds": {"step_size": [0.4, 1.2], "angular_threshold": [8.0, 75.0], "max_length": [8.0, 30.0], "n_seeds": [300.0, 900.0]}, "best_objective": 0.61}, {"stage": 2, ...}, {"stage": 3, ...}]}, ...]}`
- `/app/output/metrics.json` - `{"pairings": [{"pairing_id": "host01_A", "dice": 0.78, "hausdorff_agreement": 0.66, "surface_dice": 0.61, "precision": 0.80, "recall": 0.76}, ...], "median_dice": 0.77, "pooled_dice": 0.76}`
- `/app/output/comparison.json` - `{"directional": {"dice": ..., "hausdorff_agreement": ..., "surface_dice": ..., "precision": ..., "recall": ...}, "isotropic": {...same five keys...}, "improved_metrics": ["dice", ...]}` where the first two dictionaries hold the median metric over the six pairings.
- `/app/output/comparison_dodf.json` - `{"shore": {...five median metrics...}, "tensor": {...five median metrics...}}`

## Constraints

1. `sphere.npy` must be the only direction set used for ODF sampling; every occupancy map must have shape `(40, 40, 40)` and contain only the values 0 and 1, and at least 95% of its non-zero voxels must fall inside the host `brain_mask`.
2. The reported per-pairing `dice`, `precision`, `recall`, `surface_dice` and `hausdorff_agreement` in `metrics.json` must agree with the grader's independent recomputation from your occupancy map to within 0.03 absolute.
3. Per-pairing Dice against the reference must be at least 0.45, and the median Dice over the six pairings must be at least 0.55.
4. The reported occupancy map must not be a copy of the reference mask: it must differ from `ref` in more than 0.4% of voxels and its Dice with `ref` must be below 0.99.
5. `streamlines/<pairing_id>.npz` must be geometrically consistent with `occupancy/<pairing_id>.nii.gz`: every streamline has at least 5 points, every consecutive step length equals the fitted `step_size` to within 20%, every turn angle is at most `angular_threshold + 5` degrees, every point lies inside `brain_mask`, and rasterising the stored points (`np.round`) reproduces the occupancy map with Dice at least 0.75.
6. `curriculum.json` must record exactly three stages per pairing whose total search-box width (sum over the four parameters of `hi - lo`) strictly decreases from stage 1 to stage 3.
7. All fitted parameters must lie inside `protocol.json["parameter_bounds"]`, and `n_seeds` must be an integer in `[300, 900]`.
8. The isotropic null must be genuinely worse: the median Dice of `occupancy_iso` against the reference must be at least 0.05 below the median Dice of the directional `occupancy`, and at least three of the five directional median metrics must exceed their isotropic counterparts.
9. The single-tensor variant must actually have been run: its median Dice must be at least 0.45 and, for at least three of the six pairings, the SHORE and single-tensor occupancy maps must differ (Dice between them below 0.995).

## Scoring

The grader recomputes every metric from your written artefacts and awards partial credit as a weighted sum of eleven independently checked constraint groups: file presence (0.06), occupancy map validity (0.08), per-pairing Dice floor and reported/recomputed agreement (0.16), median Dice floor (0.10), agreement of the four remaining metrics (0.10), anti-copy checks (0.10), streamline validity and rasterisation consistency (0.14), curriculum tightening (0.08), parameter bounds (0.06), isotropic-null comparison (0.07) and single-tensor comparison (0.05). A per-constraint breakdown is printed to stdout; the final scalar is written to `/logs/verifier/reward.json`.
