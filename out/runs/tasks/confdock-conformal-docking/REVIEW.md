# Review notes

- Data is synthetic but realistic: 120 calibration complexes and 80 hidden test complexes, each with 20-40 ligand atoms, 300 protein pocket atoms, and four docking methods (vina, gnina, diffdock, equibind). Ligands and poses are written in SDF format; pockets in PDB format.
- The GNN quantile estimates are provided as a CSV; the agent implements the split conformal calibration step of ConfDock. Group-conditional calibration by (docking_method, element) is required to achieve the width reduction and per-method coverage constraints.
- Shortcut probe: the hidden test set is generated with a different seed and is not visible to the agent. Hardcoding calibration results will fail because the test complexes are new and the script must generalize. The grader recomputes coverage and width against hidden true coordinates.
- Tool chain: RDKit is used for SDF parsing, NumPy/SciPy/Pandas for computation, and scikit-learn is available for any additional modeling. The container installs these packages via pip.
- Known limitations: the synthetic error model uses Gaussian perturbations with method- and element-specific biases to emulate the GNN's miscalibration. Real docking errors may have more complex dependencies, but the conformal calibration logic remains the same.
