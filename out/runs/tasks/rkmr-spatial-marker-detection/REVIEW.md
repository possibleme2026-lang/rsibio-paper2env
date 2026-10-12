# Review notes

- Data is synthetic spatial transcriptomics with 240 spots and 100 genes, generated deterministically at build time by environment/data/make_inputs.py (seed 2026). The generator is removed from the final image to prevent shortcut access to true markers.
- Shortcut probe considered: univariate correlation or linear regression would be confounded by spatial factors; the spatial kernel adjustment and nonlinear kernel regression are required to recover the causal markers.
- Tool chain: numpy, pandas, scikit-learn, scipy. No domain-specific binary needed; the method is implemented in Python.
- Known limitation: the task is a simplified RKMR-lite; it does not implement the full spike-and-slab variational inference of the original paper, but preserves the core spatial kernel, nonlinear relevance, and PIP calibration steps.
