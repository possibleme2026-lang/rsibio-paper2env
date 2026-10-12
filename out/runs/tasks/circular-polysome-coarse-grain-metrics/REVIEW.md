# Review notes

- Real MD trajectories cannot be redistributed; synthetic coarse-grained data with matching schema (chain scaling and Rg time series) are generated deterministically from a fixed seed.
- Shortcut probe: hidden gold is not stored in environment; the verifier regenerates the original data and metrics from the seed, so data tampering or reading /app/output.json does not reveal gold.
- Known limitations: the task omits gyration-tensor dshape and the fluctuation-informed crossover model; tau is a discrete finite-time estimator and synthetic parameters are intentionally close to, but not identical to, the paper.
