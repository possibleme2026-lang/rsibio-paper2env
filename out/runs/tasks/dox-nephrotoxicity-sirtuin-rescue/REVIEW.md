# Review notes

- Data are synthetic and generated deterministically with seed 42 by environment/data/gen_inputs.py. The generator is deleted in the Dockerfile after producing the input files, so the agent cannot inspect the true effect sizes.
- Shortcut probe: the generator script contains the true group-level effects. It is removed from the final image; only the generated CSV/JSON files are visible. The agent must implement the full analysis pipeline to recover the results.
- Tool chain: Python with numpy, pandas, scipy, statsmodels. No domain-specific binaries are required; the analysis is statistical and reproducible with these libraries.
- Known limitations: the data are simulated with simplified negative-binomial and lognormal models. Real multi-omics data would have batch effects, missing values, and more complex variance structures. The task is designed to test the integrative analysis workflow rather than biological discovery.
