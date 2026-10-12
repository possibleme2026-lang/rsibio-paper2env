# Integrated Bayesian IPM for Size-Structured Forest Dynamics

You are working as a computational ecologist. A long-term forest demographic dataset has been simulated (or collected) from a size-structured tree population. The data include annual population counts by size class and individual tree growth records with annual climate covariates. Your goal is to fit an integrated integral projection model (IPM) that combines both data sources to infer underlying demographic rates and predict the size distribution for the year 2020.

The true data-generating process is a stochastic, size-structured IPM with survival, growth, and recruitment. Population counts are available for 2000–2019; individual growth records are available for 2000–2019. Weather covariates (vapor pressure deficit, VPD; precipitation) are provided for 2000–2020. The year 2020 population distribution is not provided; you must predict it.

## Inputs
All files are in `/app/data/`.
- `population_counts.csv`: 400 rows. Columns: `year`, `bin_lower`, `bin_upper`, `count`. Years 2000–2019, 20 size bins from 10 to 100 cm.
- `growth_records.csv`: 2000 rows. Columns: `year`, `initial_size`, `vpd`, `precip`, `growth_increment`. Individual annual growth increments (cm) with initial size (cm), VPD (kPa), and precipitation (mm) for the starting year.
- `weather.csv`: 21 rows. Columns: `year`, `vpd`, `precip`. Weather for 2000–2020. Use 2019 weather to transition to 2020.
- `bin_edges.csv`: 20 rows. Columns: `bin_lower`, `bin_upper`. Size bin definitions.

## Task
1. Explore the data and understand the size structure, climate variation, and growth patterns.
2. Fit a growth model from `growth_records.csv`. The growth increment is a linear function of initial size, VPD, and precipitation (plus noise). Estimate the coefficient of VPD on growth (`growth_vpd`).
3. Construct a dynamical IPM that projects the population one year forward. The IPM should include:
   - Survival probability as a logistic function of size and VPD.
   - Growth kernel that maps size to next-year size using the fitted growth model.
   - Recruitment as a fixed number of new individuals per year, with a size distribution centered at 12 cm (SD 2 cm).
4. Fit the survival and recruitment parameters to the population counts using the IPM. You may use Bayesian or optimization-based inference. Use all available years (2000–2019) to constrain the model.
5. Predict the expected size distribution for 2020 starting from the observed 2019 counts and the 2019 weather.
6. Write your results to `/app/submission/predictions.csv` and `/app/submission/params.json`.

## Constraints
1. `predictions.csv` must have exactly 20 rows, one per size bin, sorted by `bin_lower` ascending. Columns must be `bin_lower,bin_upper,predicted_count,predicted_prob`. `predicted_count` and `predicted_prob` must be non-negative. The sum of `predicted_prob` must be 1.0 ± 0.01.
2. `params.json` must be valid JSON with exactly these keys: `growth_vpd`, `growth_intercept`, `survival_vpd`, `survival_intercept`, `recruitment`. All values must be finite numbers.
3. Estimated `growth_vpd` must be negative and within 0.08 of the true simulated value.
4. The predicted 2020 size distribution must have RMSE ≤ 0.02 against the hidden true expected distribution (normalized probabilities).
5. The predicted 2020 mean size (using bin midpoints) must be within 3.0 cm of the true mean size.
6. The predicted 2020 size distribution must have Pearson correlation ≥ 0.85 with the hidden true expected distribution.
7. Estimated `recruitment` must be within 20% of the true annual recruitment rate.
8. Estimated `survival_vpd` must be negative and within 0.3 of the true value.

## Output
- `/app/submission/predictions.csv`:
  ```
  bin_lower,bin_upper,predicted_count,predicted_prob
  10.0,14.5,12.3,0.0123
  14.5,19.0,25.6,0.0256
  ...
  ```
- `/app/submission/params.json`:
  ```json
  {
    "growth_vpd": -0.25,
    "growth_intercept": 0.5,
    "survival_vpd": -0.6,
    "survival_intercept": -2.0,
    "recruitment": 30.0
  }
  ```

## Scoring
The grader checks constraints 1–8 independently. Each constraint contributes a weight: format (0.10), params format (0.10), growth_vpd (0.20), RMSE (0.20), mean size (0.10), correlation (0.10), recruitment (0.10), survival_vpd (0.10). The final score is the sum of weights for passed constraints. Partial credit is given. The score is written to `/logs/verifier/reward.json`.
