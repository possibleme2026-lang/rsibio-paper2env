# Spectral Correlates of Encoding Distinguish Good from Poor Learners

## Context
You are analyzing event-level EEG spectral power data from a free recall experiment. Each row represents a word presentation (encoding event) with the subject, list, serial position, word, log-transformed power in theta (4-8 Hz), alpha (8-12 Hz), and gamma (30-80 Hz) bands, and whether the word was later recalled. The goal is to reproduce the key analyses linking subsequent memory effects (SMEs) to individual differences in learning ability.

## Inputs
- `/app/data/encoding_events.csv`: 48,000 rows, 80 subjects, 30 lists per subject, 20 words per list. Columns: `subject_id`, `list_id`, `serial_position` (1-20), `word`, `theta_power`, `alpha_power`, `gamma_power`, `recalled` (0/1).

## Task
Implement a complete analysis pipeline that:
1. Fits a group-level logistic regression predicting `recalled` from `theta_power`, `alpha_power`, `gamma_power`, `serial_position`, and `serial_position^2`, using cluster-robust standard errors clustered by `subject_id`. Extract coefficients, standard errors, z-values, and p-values for all predictors.
2. For each subject, fits a separate logistic regression predicting `recalled` from the same predictors (theta, alpha, gamma, serial_position, serial_position^2). Extract the coefficients for theta, alpha, and gamma. These are the per-subject SMEs.
3. Computes each subject's recall accuracy as the proportion of recalled words.
4. Computes the Pearson correlation between subject accuracy and the absolute value of each per-subject SME coefficient (theta, alpha, gamma). Report r and p for each band.
5. Writes all results to `/app/output/results.json` in the exact schema below.

## Constraints
1. The group-level model must use cluster-robust standard errors clustered by `subject_id` (e.g., using `statsmodels` with `cov_type='cluster'` and `cov_kwds={'groups': data['subject_id']}`).
2. The group-level model must include `serial_position` and `serial_position^2` as covariates. The output must report coefficients for `theta_power`, `alpha_power`, `gamma_power`, `serial_position`, and `serial_position_sq`.
3. In the group-level model, the theta coefficient must be positive and its p-value < 0.05; the alpha coefficient must be negative and its p-value < 0.05; the gamma p-value must be > 0.05.
4. The per-subject analysis must include all 80 subjects. For each subject, report the theta, alpha, and gamma coefficients from the per-subject logistic regression.
5. The correlation between accuracy and absolute theta SME must be negative (r < 0). The same for absolute alpha SME (r < 0). Report r and p for both.
6. All numeric values in the output must be JSON numbers (not strings). Round coefficients to 6 decimal places, standard errors to 6 decimal places, z-values to 4 decimal places, and p-values to 6 decimal places.

## Output
Write `/app/output/results.json` with this structure:

```json
{
  "group_level": {
    "theta_power": {"coef": 0.451234, "se": 0.023456, "z": 19.2345, "p": 0.000000},
    "alpha_power": {"coef": -0.563210, "se": 0.024567, "z": -22.9123, "p": 0.000000},
    "gamma_power": {"coef": 0.002345, "se": 0.021234, "z": 0.1104, "p": 0.912345},
    "serial_position": {"coef": 0.123456, "se": 0.012345, "z": 10.0000, "p": 0.000000},
    "serial_position_sq": {"coef": -0.005678, "se": 0.000567, "z": -10.0000, "p": 0.000000}
  },
  "per_subject": [
    {"subject_id": 1, "theta": 0.4321, "alpha": -0.5432, "gamma": 0.0123},
    ...
  ],
  "correlations": {
    "theta": {"r": -0.5432, "p": 0.000001},
    "alpha": {"r": -0.6123, "p": 0.000000},
    "gamma": {"r": 0.0123, "p": 0.912345}
  }
}
```

## Scoring
The grader checks the following 10 constraints independently. Score = (number of passed checks) / 10.
1. Output file exists and is valid JSON.
2. Group-level section has all required predictors with coef, se, z, p.
3. Theta coefficient > 0 and p < 0.05.
4. Alpha coefficient < 0 and p < 0.05.
5. Gamma p > 0.05.
6. Group-level coefficients match reference within ±0.05.
7. Per-subject section contains exactly 80 subjects with theta, alpha, gamma coefficients.
8. Per-subject coefficients for 5 randomly selected subjects match reference within ±0.1.
9. Correlation theta r < 0 and alpha r < 0.
10. Correlation values match reference within ±0.1.

Partial credit is proportional to the number of constraints satisfied.
