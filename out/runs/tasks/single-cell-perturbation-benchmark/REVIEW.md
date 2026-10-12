# Review notes

- Data is synthetic but realistic: 500 control cells, 200 perturbed cells, 300 genes, 6 perturbations for task1 and 5 for task2; generated deterministically with seed 42 for visible data and seed 2024 for hidden data.
- Shortcut probe: the grader runs the agent's evaluate.py on hidden data generated with a protected seed, so hardcoded outputs fail.
- Tool chain: numpy, scipy, statsmodels, scikit-learn, anndata for h5ad I/O.
- Energy distance on PCA is a proxy for distributional similarity; DE uses Welch's t-test and BH FDR as specified.
