# Circular vs linear polysome coarse-grain metrics

Synthetic coarse-grained data stand in for the MD trajectories described in the paper. The files /app/environment/data/chain_scaling.csv and /app/environment/data/rg_timeseries.csv are generated at image build time. If they are missing, run: python /app/environment/data/generate_data.py.

Compute the following quantities and write them to /app/output.json as a JSON object:
- nu_circular, nu_linear: ordinary least-squares slope of log(Rg_mean) versus log(N) for each topology in chain_scaling.csv.
- rel_fluct_circular, rel_fluct_linear: population std(Rg) / mean(Rg) over the corresponding time series in rg_timeseries.csv.
- tau_circular, tau_linear: normalized autocorrelation time of Rg(t). Subtract the mean; rho_k = sum_t y_t y_{t+k} / sum_t y_t^2; tau is the first lag k (dt = 1.0) with rho_k <= exp(-1), or T-1 if no crossing occurs.
- tau_ratio: tau_linear / tau_circular.

Output keys must be exactly: nu_circular, nu_linear, rel_fluct_circular, rel_fluct_linear, tau_circular, tau_linear, tau_ratio. All values must be JSON numbers.

Scoring: a hidden grader regenerates the original synthetic data from a fixed seed, computes the same metrics, and returns score = max(0, 1 - 5 * normalized RMSE), where normalized RMSE is the root-mean-square relative error over the seven metrics.

Constraints: no network access; do not read or modify /tests; write /app/output.json during the task. The grader writes /logs/verifier/reward.json.
