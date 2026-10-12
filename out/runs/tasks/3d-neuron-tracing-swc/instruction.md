# 3D Neuron Tracing from Fluorescence Microscopy

You are a computational neuroscientist reconstructing neuronal morphology from a 3D fluorescence microscopy volume. The volume contains a single neuron with a soma and branching neurites. Your goal is to produce a standard SWC reconstruction that matches the hidden gold-standard morphology.

## Inputs
- `/app/data/volume.npy`: a 128×128×64 float32 array (values 0–1) representing a noisy, low-contrast 3D fluorescence microscopy volume. Voxel size is 0.2 µm isotropic.
- `/app/data/metadata.json`: JSON with `voxel_size_um` (`[0.2, 0.2, 0.2]`), `volume_shape` (`[128, 128, 64]`), and `seed_voxel` (the soma seed point in voxel coordinates).
- `/app/data/gen_inputs.py`: the deterministic generator script used to create the volume. The seed used to generate the data is **not** present in the image; do not attempt to recover it. Use the image itself.

## Task
Reconstruct the full neuronal morphology from the volume. You must:
1. Preprocess and threshold the volume to isolate the neurites.
2. Compute a 3D skeleton (e.g., using `skimage.morphology.skeletonize_3d`).
3. Build a graph from the skeleton, handle loops/noise, and trace from the provided soma seed.
4. Estimate the radius at each reconstructed node (e.g., from a Euclidean distance transform).
5. Convert voxel coordinates to micrometers and write a valid SWC file.

## Constraints
1. **SWC format**: The output must be a valid SWC file. Each non-comment line must contain exactly 7 whitespace-separated columns: `id type x y z radius parent`. IDs must be unique positive integers. There must be exactly one root node with `parent = -1` and `type = 1`. Every non-root node's `parent` must exist and be strictly smaller than its own `id`. No duplicate IDs.
2. **Connected tree**: The reconstruction must form a single connected tree rooted at the soma. All nodes must be reachable from the root via parent links. No cycles or disconnected components.
3. **Root proximity**: The root node's coordinates must be within 1.5 µm of the provided seed point (converted to microns using `voxel_size_um`).
4. **Path length**: The total path length of your reconstruction (sum of Euclidean edge lengths in µm) must be within ±30% of the hidden gold-standard total path length.
5. **Node accuracy**: The node F1 score against the hidden gold-standard nodes must be at least 0.7 at a tolerance of 2.0 µm (nearest-neighbor matching in 3D).
6. **Branch-point accuracy**: The branch-point F1 score (nodes with degree ≥ 3) must be at least 0.6 at a tolerance of 3.0 µm.
7. **Radius accuracy**: For matched nodes, the mean absolute error (MAE) of radii must be ≤ 0.5 µm.
8. **Physical bounds**: All coordinates must lie within the physical volume: x, y ∈ [0, 25.6] µm and z ∈ [0, 12.8] µm.

## Output
Write your reconstruction to `/app/output/trace.swc` in standard SWC format. Example row:
```
1 1 10.2400 12.8000 5.1200 0.6000 -1
2 3 10.4800 12.9600 5.2800 0.5600 1
```

## Scoring
The grader checks the 8 constraints above. Each constraint contributes equally to a score in [0, 1]. For continuous metrics (F1, MAE, path-length ratio, root distance), partial credit is awarded linearly up to the threshold. The final `reward` is the mean constraint score. The grader writes `{"score": <float>, "reward": <float>}` to `/logs/verifier/reward.json`.
