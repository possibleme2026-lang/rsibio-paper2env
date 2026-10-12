# Ambulatory light dosimetry: chest and wrist placement bias

You are a computational biologist analysing a multi-site ambulatory light dosimetry study. Ten-second melanopic equivalent daylight illuminance (MEDI) was measured concurrently at the glasses, chest, and wrist. Your task is to implement the paper's placement-bias analysis pipeline on the provided synthetic-but-realistic dataset and write a machine-readable summary.

## Inputs
- `/app/data/dosimetry.csv`: 829,440 rows (12 participants x 8 days x 8640 10-s samples). Columns: `datetime`, `participant_id`, `site`, `participant_day`, `context`, `quality_flag`, `glasses_MEDI`, `chest_MEDI`, `wrist_MEDI`. Invalid samples have `quality_flag=1` and all three MEDI values set to -1.
- `/app/data/gen_inputs.py`: deterministic generator that produced the CSV (for provenance; do not modify).

## Task
1. Load the data. Define a valid sample as `quality_flag == 0` and all three MEDI values > 0.
2. Exclude any participant-day with fewer than 80% of the expected 8640 valid samples (< 6912). Report the excluded participant-day IDs sorted lexicographically. All subsequent analyses use only included participant-days and valid samples.
3. Time-resolved placement error: for each placement (chest, wrist) and each context, compute the percentage error `100 * (placement - glasses) / glasses` for every valid sample. Report the median and mean of these errors per context.
4. Daily metrics: for each included participant-day and each placement, compute the following metrics from the valid samples, sorted by datetime:
   - `mean_log_MEDI`: mean of `log10(MEDI + 1)`.
   - `median_MEDI`: median of MEDI.
   - `TAT_250`: fraction of samples with MEDI >= 250.
   - `TAT_1000`: fraction of samples with MEDI >= 1000.
   - `max_MEDI`: maximum MEDI.
   - `CoM_hour`: circular center of mass of exposure in hours: convert each sample's time-of-day `h` to angle `2*pi*h/24`; weight by MEDI; compute `atan2(mean(MEDI*sin(theta)), mean(MEDI*cos(theta))) * 24/(2*pi) mod 24`.
   - `IV`: intradaily variability of `log10(MEDI + 1)`: `sqrt( n * sum_{t=2}^n (x_t - x_{t-1})^2 / ((n-1) * sum_{t=1}^n (x_t - mean(x))^2) )`. If n < 2 or variance is zero, set IV to 0.
5. For each metric and placement, compute absolute bias versus glasses across included participant-days:
   - For all metrics except `CoM_hour`: if glasses metric is 0 and placement metric is 0, bias = 0; if glasses metric is 0 but placement metric is not, skip that day for that metric; otherwise `abs(100 * (placement - glasses) / glasses)`.
   - For `CoM_hour`: `min(|placement - glasses|, 24 - |placement - glasses|)`.
   Report the median absolute bias, maximum absolute bias, and number of days used (`n_days`).
6. Flag metrics whose median absolute bias exceeds 5 for non-timing metrics or 0.5 for `CoM_hour`. Report sorted lists per placement.

## Constraints
1. Valid samples must satisfy `quality_flag == 0` and all three MEDI > 0.
2. Excluded participant-days are exactly those with < 6912 valid samples; include them sorted lexicographically.
3. Context errors must use included days only and be computed per placement and context for all six contexts: `commute`, `home_evening`, `indoor_day`, `indoor_night`, `outdoor_day`, `outdoor_night`.
4. Daily metrics must use only valid samples within included participant-days, sorted by datetime. Use the exact metric definitions above.
5. Bias for `CoM_hour` is circular absolute hour difference. All other biases are absolute percentage differences. Days with zero reference metric are handled as specified.
6. The output JSON must contain exactly the keys shown below. Round all floating-point numbers to 4 decimal places.
7. `metrics_exceeding_threshold` lists must be sorted alphabetically.
8. Do not modify `/app/data/dosimetry.csv`; write the result to `/app/output/placement_analysis.json`.

## Output
Write a single JSON object to `/app/output/placement_analysis.json` with this structure (values are illustrative):
```json
{
  "excluded_participant_days": ["P01_D01", "P02_D02"],
  "valid_sample_count": 700000,
  "context_errors": {
    "chest": {
      "commute": {"median_pct_error": -8.1234, "mean_pct_error": -8.2345},
      "home_evening": {"median_pct_error": -12.0, "mean_pct_error": -12.1},
      "indoor_day": {"median_pct_error": -10.1, "mean_pct_error": -10.2},
      "indoor_night": {"median_pct_error": -22.0, "mean_pct_error": -22.1},
      "outdoor_day": {"median_pct_error": -4.1, "mean_pct_error": -4.2},
      "outdoor_night": {"median_pct_error": -24.0, "mean_pct_error": -24.1}
    },
    "wrist": {
      "commute": {"median_pct_error": -32.0, "mean_pct_error": -32.1},
      "home_evening": {"median_pct_error": -38.0, "mean_pct_error": -38.1},
      "indoor_day": {"median_pct_error": -35.0, "mean_pct_error": -35.1},
      "indoor_night": {"median_pct_error": -40.0, "mean_pct_error": -40.1},
      "outdoor_day": {"median_pct_error": -55.0, "mean_pct_error": -55.1},
      "outdoor_night": {"median_pct_error": -45.0, "mean_pct_error": -45.1}
    }
  },
  "daily_metric_bias": {
    "chest": {
      "CoM_hour": {"median_abs_bias": 0.12, "max_abs_bias": 0.45, "n_days": 88},
      "IV": {"median_abs_bias": 4.2, "max_abs_bias": 15.3, "n_days": 88},
      "TAT_1000": {"median_abs_bias": 2.1, "max_abs_bias": 20.0, "n_days": 88},
      "TAT_250": {"median_abs_bias": 3.0, "max_abs_bias": 25.0, "n_days": 88},
      "max_MEDI": {"median_abs_bias": 5.1, "max_abs_bias": 12.0, "n_days": 88},
      "mean_log_MEDI": {"median_abs_bias": 4.0, "max_abs_bias": 10.0, "n_days": 88},
      "median_MEDI": {"median_abs_bias": 5.5, "max_abs_bias": 18.0, "n_days": 88}
    },
    "wrist": {
      "CoM_hour": {"median_abs_bias": 0.25, "max_abs_bias": 0.8, "n_days": 88},
      "IV": {"median_abs_bias": 8.0, "max_abs_bias": 30.0, "n_days": 88},
      "TAT_1000": {"median_abs_bias": 10.0, "max_abs_bias": 50.0, "n_days": 88},
      "TAT_250": {"median_abs_bias": 12.0, "max_abs_bias": 60.0, "n_days": 88},
      "max_MEDI": {"median_abs_bias": 15.0, "max_abs_bias": 40.0, "n_days": 88},
      "mean_log_MEDI": {"median_abs_bias": 10.0, "max_abs_bias": 25.0, "n_days": 88},
      "median_MEDI": {"median_abs_bias": 18.0, "max_abs_bias": 84.0, "n_days": 88}
    }
  },
  "metrics_exceeding_threshold": {
    "chest": ["max_MEDI", "median_MEDI"],
    "wrist": ["IV", "TAT_1000", "TAT_250", "max_MEDI", "mean_log_MEDI", "median_MEDI"]
  }
}
```

## Scoring
The grader recomputes the gold summary from a hidden deterministic copy of the dataset. Score is the fraction of 8 independently verified checks passed: (1) excluded days, (2) valid sample count, (3) chest context errors, (4) wrist context errors, (5) chest daily metric bias, (6) wrist daily metric bias, (7) chest threshold list, (8) wrist threshold list. Numeric checks use tolerance 0.01; lists must match exactly. Partial credit is awarded per check.
