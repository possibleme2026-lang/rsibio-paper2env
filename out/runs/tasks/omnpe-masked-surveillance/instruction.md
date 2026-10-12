# Amortised parameter recovery from masked longitudinal surveillance data

## Context

You are rebuilding the inference pipeline of an observation-masked neural posterior estimation (OM-NPE) study. Sentinel surveillance of an arboviral outbreak across six French Polynesian archipelagos is stored on a common weekly grid (52 weeks), but each archipelago reported only in some weeks. Two things must be produced: (a) a mechanistic effective-SEIR fit per archipelago that uses **only the weeks that were actually reported**, and (b) a fast amortised surrogate that maps cheap surveillance summary statistics straight to the basic reproduction number R0, so that alternative monitoring schedules can be evaluated without refitting a model.

## Inputs

- `/app/data/surveillance.csv` — 312 rows, columns `region,week,cases`. A **blank `cases` cell means that archipelago did not report that week** (this is the observation mask). Regions: `Tahiti`, `Moorea`, `Bora-Bora`, `Raiatea`, `Nuku_Hiva`, `Tubuai`.
- `/app/data/regions.csv` — columns `region,population`.
- `/app/data/simulated_library.csv` — 250 synthetic regions generated from the same effective-SEIR + masking process, with known R0. Columns: `sim_id,peak_week,log1p_peak,log1p_total,log1p_case_rate,obs_frac,late_frac,log10_N,R0`.

## Task

### Step 1 — Fit the effective SEIR model per region

Integrate, for each region with population `N`,

```
dS/dt = -beta*S*I/N
dE/dt =  beta*S*I/N - sigma*E
dI/dt =  sigma*E - gamma*I
dR/dt =  gamma*I
```

with `S(0)=N-1, E(0)=0, I(0)=1, R(0)=0`. The model prediction for week `w` (w = 0..51) is `I(w)`. Estimate `(beta, sigma, gamma)` by least squares **restricted to the weeks where `cases` is not blank** — masked weeks must not contribute to the loss or to any reported error. Report `R0 = beta/gamma`.

### Step 2 — Build the amortised surrogate

Train a regressor on `simulated_library.csv` that maps the seven surveillance summary statistics below to R0, and ship it with an executable `/app/output/amortized_predict.py`. When called as

```
python /app/output/amortized_predict.py --features /path/to/features.json
```

it must print a single JSON object `{"R0": <number>}` on stdout. Training may be done at solve time (for example via an extra `--train` flag); only the `--features` interface is exercised by the grader.

Summary statistics for a region with observed (unmasked) weekly counts `y`, observed week indices `idx`, population `N`, and `total = sum(y)`:

- `peak_week` = (week index of the maximum observed count) / 51
- `log1p_peak` = ln(1 + max(y))
- `log1p_total` = ln(1 + total)
- `log1p_case_rate` = ln(1 + total / N)
- `obs_frac` = (number of observed weeks) / 52
- `late_frac` = sum(y over weeks 26..51) / (sum(y over weeks 0..25) + sum(y over weeks 26..51) + 1)
- `log10_N` = log10(N)

The grader will hand your predictor this feature vector for previously unseen synthetic regions and compare the predicted R0 with the true value.

## Constraints

1. `/app/output/fits.csv` must have exactly the columns `region,beta,sigma,gamma,R0,fit_rmse` and exactly one row per surveillance region (6 rows), with the region labels spelled as in `regions.csv`.
2. Every fitted value must be admissible: `0.05 <= beta <= 25`, `0.1 <= sigma <= 12`, `0.1 <= gamma <= 12`, `0.6 <= R0 <= 6.0`, `abs(R0 - beta/gamma) <= 0.001*R0`, and `fit_rmse >= 0`.
3. The fitted `R0` of every region must recover the (hidden) true generating value: full credit within 15% relative error, half credit within 30%.
4. `/app/output/predictions.csv` must have exactly the columns `region,week,predicted_cases`, with 52 consecutive integer weeks 0..51 for each of the six regions. For every region the normalised RMSE against the reported weeks, `RMSE / mean(reported)`, must be below `0.8 * CV(reported)` where `CV = std/mean` of the reported weeks. The `fit_rmse` reported in `fits.csv` must agree with the recomputed RMSE within 25%.
5. `/app/output/pooled.json` must be a JSON object with a numeric `pooled_R0` and `pooled_R0_ci` (two numbers `lo, hi`) satisfying `lo <= pooled_R0 <= hi`, `hi > lo`, `0.6 <= pooled_R0 <= 6.0`, and `pooled_R0` within +/-0.2 of the range spanned by the six per-region R0 values.
6. `/app/output/amortized_predict.py` must answer a `--features` query in under 8 seconds, and its predictions on 20 held-out synthetic regions must be accurate: full credit within 15% relative error, 0.7x credit within 25%, 0.35x credit within 40%, otherwise zero.
7. All numeric output must be finite (no `NaN`, `inf`, or empty cells), and no output file may be written outside `/app/output/`.

## Output

- `/app/output/fits.csv`

```
region,beta,sigma,gamma,R0,fit_rmse
Tahiti,2.887,,,...
```

(e.g. a row `Tahiti,2.8875,2.10,1.05,2.7500,123.4`)

- `/app/output/predictions.csv`

```
region,week,predicted_cases
Tahiti,0,1.0
Tahiti,1,1.4
```

- `/app/output/pooled.json` — e.g. `{"pooled_R0": 2.13, "pooled_R0_ci": [1.78, 2.51]}`
- `/app/output/amortized_predict.py` — the amortised predictor (plus whatever model artefact it needs, also under `/app/output/`).

## Scoring

`tests/grade.py` evaluates seven independent constraint groups and writes a partial-credit score to `/logs/verifier/reward.json`. Weights: `fits_format` 0.10, `parameter_ranges` 0.10, `r0_recovery` 0.25, `trajectory_fit` 0.10, `pooled_posterior` 0.10, `amortised_generalisation` 0.30, `amortised_runtime` 0.05. The per-region and per-holdout breakdown is printed to stdout and stored in `/logs/verifier/report.json`.
