# Review notes

- Synthetic event-level EEG spectral power data generated to mimic the statistical structure of the paper's dataset (48,000 trials, 80 subjects). Real dataset not publicly available; synthetic data preserves effect sizes and correlation structure.
- Shortcut probe: Agent could try to output constants from the paper (theta positive, alpha negative) without computing from data. Grader checks per-subject coefficients and correlations against reference computed from the actual data, so constants fail.
- Tool chain: Python with numpy, pandas, scipy, statsmodels. Uses statsmodels.Logit with cluster-robust SE for group-level model, matching standard practice.
- Known limitation: Gamma effect set to zero for group-level non-significance; per-subject gamma coefficients are noise but required for schema completeness. The correlation for gamma is computed but not scored.
