# Review notes

- Data is synthetically generated with a deterministic seed (42) by environment/data/gen_inputs.py, which is removed from the final image. The generator simulates realistic scRNA-seq counts (8000 cells x 300 genes), BCR/TCR repertoire tables, and clinical titers with latent High/Low responder effects.
- Shortcut probe: the gold standard is not present in the environment. The grader regenerates the pristine data from the protected seed in tests/gold_gen.py and computes the reference analysis, then compares the agent's output. Reading the environment data alone does not reveal the expected scores or DE genes.
- Tool chain: Python 3.11, numpy, pandas, scipy, scikit-learn, statsmodels. The task requires implementing normalization, z-scoring, Wilcoxon tests, BH correction, Shannon entropy, and Pearson correlations.
- Known limitations: The synthetic data is an approximation; real scRNA-seq data would have more genes and cells. The reference pipeline defines GNG4+ Tfh cells by a percentile threshold, which may not perfectly match biological definitions. The grader tolerates small numerical differences.
