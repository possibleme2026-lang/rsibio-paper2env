# Physics-Guided Reconstruction of 3D Membrane Curvature from Noisy Volumetric Images

You are given a set of simulated 3D electron microscopy (EM) volumes of cellular membranes. Each volume is a 97×97×97 voxel grid with isotropic 1.0 nm spacing. The intensity is high near the membrane surface and low in the background, with additive Gaussian noise. Your task is to reconstruct the membrane surface for each volume and compute geometric curvature statistics.

## Inputs
- `/app/data/volumes/vol_00.npy` ... `vol_04.npy`: float32 arrays of shape (97, 97, 97). The physical coordinate axes are x, y, z, spanning -48 to +48 nm with 1.0 nm voxel size.
- `/app/data/metadata.csv`: CSV with columns `volume_id,voxel_size_nm,roi_expr`. The `roi_expr` is a Python-evaluable expression in `x`, `y`, `z` (in nm) that defines the region of interest for computing summary metrics. For example, `abs(z)<=8` means only include mesh faces whose centroid has |z| <= 8 nm.

## Task
For each volume, reconstruct the membrane surface as a continuous triangulated mesh. Then compute, over the region of interest (ROI) specified in metadata.csv:
1. Surface area in nm².
2. Area-weighted mean of the absolute mean curvature, in nm⁻¹.
3. Area-weighted mean of the absolute Gaussian curvature, in nm⁻².

For the volume `vol_03`, the membrane is a catenoid-like neck. Detect the neck constriction and report its center and principal curvatures in `/app/data/results/neck_metrics.json`.

## Constraints
1. **Output files**: Write `/app/data/results/curvature_summary.csv` with exactly columns `volume_id,surface_area_nm2,mean_abs_H_nm_inv,mean_abs_K_nm_inv2` and one row for each of the 5 volumes. Write `/app/data/results/neck_metrics.json` as a JSON object with keys `volume_id`, `neck_center_nm` (list of 3 floats), `kappa1_nm_inv`, `kappa2_nm_inv`, `anisotropy_index`.
2. **ROI compliance**: Metrics in the CSV must be computed only over mesh faces whose centroid satisfies the `roi_expr` for that volume. Use area weighting for H and K.
3. **Surface area accuracy**: Relative error vs the hidden ground truth must be <= 12% for each volume.
4. **Mean absolute mean curvature accuracy**: Relative error vs hidden ground truth must be <= 18% for each volume.
5. **Mean absolute Gaussian curvature accuracy**: Relative error vs hidden ground truth must be <= 25% for each volume.
6. **Neck detection**: For `vol_03`, the reported `neck_center_nm` must be within 4 nm of the true neck center. The principal curvatures must have opposite signs, and their magnitudes must be within 25% of the true catenoid values (|κ| ≈ 0.125 nm⁻¹). The anisotropy index `|κ1-κ2|/(|κ1|+|κ2|)` must be >= 0.7.

## Output
- `/app/data/results/curvature_summary.csv` example row:
```
volume_id,surface_area_nm2,mean_abs_H_nm_inv,mean_abs_K_nm_inv2
vol_00,12345.67,0.045,0.0021
```
- `/app/data/results/neck_metrics.json` example:
```json
{
  "volume_id": "vol_03",
  "neck_center_nm": [0.0, 0.0, 0.0],
  "kappa1_nm_inv": 0.11,
  "kappa2_nm_inv": -0.11,
  "anisotropy_index": 1.0
}
```

## Scoring
The grader compares your CSV and JSON to hidden ground truth generated from the clean (noise-free) volumes. Score is the average of 5 equally weighted components:
- Correct file formats and valid values (0.2)
- Surface area relative error <= 12% for each volume (0.2, partial credit per volume)
- Mean abs H relative error <= 18% for each volume (0.2, partial credit per volume)
- Mean abs K relative error <= 25% for each volume (0.2, partial credit per volume)
- Neck metrics: center, opposite signs, magnitudes, anisotropy (0.2, partial credit per sub-check)
The final score is written to `/logs/verifier/reward.json` as a JSON number.
