# Data-driven disease dynamics: optimal reaction coordinate and free energy landscape

You are working on a computational biology challenge inspired by the paper "Data Driven Disease Dynamics Models". The paper introduces a framework to describe disease progression as a diffusion on a free energy landscape using optimal reaction coordinates (RCs). Your task is to implement the core pipeline on a synthetic longitudinal clinical dataset.

## Inputs
- `/app/data/longitudinal_biomarkers.csv`: A CSV file with columns: `patient_id`, `visit_time`, `biomarker_1` ... `biomarker_12`, `diagnosis`. The data contains 200 patients, each with 5-15 visits at irregular times between 0 and 12 years. `diagnosis` is 0/1.
- `/app/data/gen_inputs.py`: The deterministic generator script used to create the dataset (for reproducibility).

## Task
Write a script (e.g., `/app/solution.py`) that performs the following steps and writes the results to `/app/submission.json`.

1. **Preprocessing**: For each patient with at least 4 visits, linearly interpolate all 12 biomarkers onto a uniform time grid from `t=0` to `t=12` with step `dt=0.1` years. Only interpolate within the patient's observed time range (`min_time` to `max_time`). Discard patients with fewer than 4 visits.
2. **Standardization**: Compute the mean and standard deviation (population, `ddof=0`) of each biomarker across all interpolated samples (pooled over all patients). Standardize all interpolated samples and all original visit samples using these statistics.
3. **TICA**: Use a lag time `tau = 1.0` year (i.e., `lag_steps = 10` on the uniform grid). For each patient, form time-lagged pairs `(x(t), x(t+tau))` from the standardized interpolated data. Pool all pairs. Compute the mean covariance matrices `C0 = <x(t) x(t)^T>` and `Ctau = <x(t) x(t+tau)^T>`. Solve the generalized eigenvalue problem `Ctau v = lambda C0 v`. Take the eigenvector `v1` corresponding to the largest eigenvalue `lambda1`. Normalize it so that `v1^T C0 v1 = 1`. Apply the sign convention: the coefficient with the largest absolute value must be positive.
4. **Reaction coordinate**: For each original visit, project the standardized biomarker values onto `v1` to obtain `rc`.
5. **Free energy**: Using all `rc` values from all visits, compute a histogram with 20 equally spaced bins between the 1st and 99th percentiles of `rc`. Compute the probability density `p_i = count_i / (N * bin_width)`. The free energy is `F_i = -ln(p_i)`, shifted so that `min(F_i) = 0`. Output the 20 `F_i` values in order of increasing `rc`.
6. **Diffusion coefficient**: On the uniform grid, compute the mean squared displacement of `rc` over one step (`dt=0.1`): `D = mean( (rc(t+dt) - rc(t))^2 ) / (2 * dt)` over all patients and all valid time points.
7. **Validation score**: Using the generalized eigenvalues sorted in descending order (`lambda1 >= lambda2 >= ...`), compute the spectral gap ratio `S = (lambda1 - lambda2) / lambda1`. If only one eigenvalue exists, `S = 1.0`.
8. **AUC**: Compute the ROC AUC of using `rc` to classify `diagnosis` (0/1) on all original visits.

## Constraints
1. The output file `/app/submission.json` must be valid JSON with exactly the keys: `rc_vector` (list of 12 floats), `free_energy` (list of 20 floats), `diffusion_coefficient` (float), `validation_score` (float), `auc` (float).
2. `rc_vector` must be normalized such that `v^T C0 v = 1` (tolerance 0.05) and must satisfy the sign convention (largest absolute coefficient positive).
3. `free_energy` must be a list of exactly 20 floats, shifted so that its minimum is 0 (tolerance 1e-3).
4. `diffusion_coefficient` must be positive and within 5% relative error of the gold value computed by the reference implementation.
5. `validation_score` must be in [0, 1] and within 0.01 absolute error of the gold value.
6. `auc` must be in [0, 1] and within 0.01 absolute error of the gold value.
7. `free_energy` must match the gold profile with mean absolute error < 0.2 and Pearson correlation > 0.95.
8. `rc_vector` must match the gold direction in the C0 metric: `abs(v @ C0 @ gold_v) / sqrt((v @ C0 @ v) * (gold_v @ C0 @ gold_v)) > 0.99`.

## Output
Write `/app/submission.json` with these keys. Example structure:
```json
{
  "rc_vector": [0.12, -0.05, ...],
  "free_energy": [0.0, 0.5, ...],
  "diffusion_coefficient": 0.034,
  "validation_score": 0.72,
  "auc": 0.85
}
```

## Scoring
The grader recomputes the gold standard using the reference implementation on the same input data. Each of the 8 constraints above is checked independently. The final score is the fraction of constraints passed (partial credit). The score and reward are written to `/logs/verifier/reward.json`.
