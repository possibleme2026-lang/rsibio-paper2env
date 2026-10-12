# ConfDock: Atom-specific Uncertainty Quantification for Molecular Docking

You are given a calibration set of 120 protein-ligand complexes with true crystal poses and predicted poses from four docking methods (`vina`, `gnina`, `diffdock`, `equibind`). A pretrained graph neural network (GNN) provides per-atom quantile estimates of the 90th percentile of Euclidean error for each predicted pose. Your task is to implement the split conformal calibration step of ConfDock to produce atom-specific 90% prediction intervals (radii in Angstroms) for a held-out test set.

## Inputs

- `/app/data/calibration/` contains the calibration data:
  - `proteins/CPX0000.pdb`, ... : protein pocket structures (PDB format).
  - `ligands/CPX0000_true.sdf`, ... : true ligand poses (SDF format).
  - `poses/CPX0000_vina.sdf`, `CPX0000_gnina.sdf`, `CPX0000_diffdock.sdf`, `CPX0000_equibind.sdf`, ... : predicted poses per docking method (SDF format).
  - `gnn_quantiles.csv` : columns `complex_id,docking_method,atom_id,gnn_quantile`.
- The hidden test set will be provided at runtime to your script as a directory with the same structure but **without** `ligands/` (no true poses). It will contain `proteins/`, `poses/`, and `gnn_quantiles.csv`.

## Task

Write a Python script at `/app/solution/predict.py` that:

1. Reads the calibration data from `/app/data/calibration/`.
2. For each calibration atom (complex, method, atom_id), computes the Euclidean error between the true pose and the predicted pose, and forms the nonconformity score: `score = error / gnn_quantile`.
3. Implements group-conditional (Mondrian) split conformal calibration. Use groups defined by `(docking_method, element)`, where `element` is the atom's chemical element from the SDF. For each group, compute the finite-sample corrected `1-alpha` quantile of scores (with `alpha=0.1`). Handle groups with fewer than 10 calibration atoms by falling back to the method-level quantile, then to the global quantile.
4. For a given test directory, reads the predicted poses and `gnn_quantiles.csv`, determines each atom's group, and computes the prediction radius: `radius = q_group * gnn_quantile`.
5. Writes a CSV to the path specified by `--output` with columns `complex_id,docking_method,atom_id,radius` (one row per test atom).

The script must accept command-line arguments `--input <test_dir>` and `--output <output_csv>`.

## Constraints

1. The output CSV must have exactly the columns `complex_id,docking_method,atom_id,radius` and must contain exactly one row per test atom (matching the hidden test set keys).
2. All radii must be finite, positive, and <= 10.0 Angstroms.
3. Overall empirical coverage on the hidden test set must be >= 0.90 (target 90% coverage).
4. Per-docking-method coverage must be >= 0.85 for each of the four methods.
5. Mean interval radius must be <= 0.70 times the mean radius of standard split conformal prediction (global quantile) on the same hidden test set (i.e., at least 30% reduction in mean width).
6. Within each docking method, the Pearson correlation between the output radius and the provided `gnn_quantile` must be >= 0.5.

## Output

`/app/solution/predict.py` and, when run, a CSV like:

```
complex_id,docking_method,atom_id,radius
TEST0000,vina,1,0.85
TEST0000,vina,2,1.12
...
```

## Scoring

The grader runs your script on a hidden test set and checks constraints 1-6. Score = fraction of constraints passed. Partial credit is given. The hidden test set is generated deterministically with a different seed than the calibration set.
