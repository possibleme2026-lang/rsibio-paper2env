# Reproducing the 18F-FDG micro-PET/CT chemoradiation-resistance analysis in orthotopic HNSCC models

## Background

A preclinical imaging study of chemoradiation resistance in head and neck squamous cell carcinoma (HNSCC) used three orthotopic murine models: MOC1 and MOC2 (HPV-negative) and MLM1 (HPV-positive). Therapy started once the tumour exceeded 50 mm3 and consisted of cisplatin (5 mg/kg) plus image-guided radiation (8 Gy) on days 0 and 7. Tumours were measured by caliper on days 0, 3, 7, 11 and 14; every mouse also received an 18F-FDG micro-PET/CT scan on day 14 with manual tumour segmentation. Treatment response was scored from the day-11 relative tumour size, and survival was followed to a humane endpoint.

You are given the raw cohort table and the PET/CT region-of-interest measurements, and nothing else. Rebuild the complete quantitative pipeline that the study used and write one machine-readable result file. In the published report the day-14 PET/CT tumour volume and SUVmax were the features most strongly associated with survival; whether SUVmean carries prognostic information is for your analysis to decide.

## Inputs

Both files live in `/app/data` and were produced by the deterministic generator `/app/data/gen_inputs.py` (seed 20260803).

`/app/data/mouse_cohort.csv` - 121 rows, one per treated mouse:
- `mouse_id` (e.g. `MOC1-07`), `model` (`MOC1` | `MOC2` | `MLM1`), `treatment`
- `day0_volume_mm3`, `day3_volume_mm3`, `day7_volume_mm3`, `day11_volume_mm3`, `day14_volume_mm3`: caliper tumour volumes in mm3
- `survival_days`: days from treatment start to death or last follow-up
- `event`: 1 = death observed, 0 = censored

`/app/data/pet_ct_roi.csv` - one row per mouse for the day-14 scan:
- `mouse_id`, `scan_day`, `body_weight_g`, `injected_dose_mbq`, `uptake_time_min`
- `roi_volume_mm3`: manually segmented PET/CT tumour volume on day 14
- `roi_activity_mean_kbq_ml`, `roi_activity_max_kbq_ml`: mean and maximum 18F-FDG activity concentration in the tumour ROI

## Task

Work through the following steps and write `/app/output/results.json`.

1. Join the two tables on `mouse_id` (all 121 mice must survive the join) and derive the PET metrics with the standard body-weight SUV normalisation: `SUV = C_kBq_per_mL * body_weight_g / (1000 * injected_dose_mbq)`. `suv_max` is built from `roi_activity_max_kbq_ml` and `suv_mean` from `roi_activity_mean_kbq_ml`.
2. Score day-11 treatment response: `relative_volume_day11 = day11_volume_mm3 / day0_volume_mm3`; a mouse is a responder iff `relative_volume_day11 < 1.0`.
3. Growth trajectories: reshape the five caliper volumes into long format and fit linear mixed-effects models on `ln(volume_mm3)` with a random intercept per mouse, fitted by maximum likelihood (REML = False) with `day` as a continuous covariate. Report the likelihood-ratio statistic and p-value for (a) the `model x day` interaction (full `log_volume ~ C(model) * day`, null `log_volume ~ C(model) + day`, df = 2) and (b) the `responder x day` interaction (null `log_volume ~ responder + day`, df = 1).
4. Survival by model: Kaplan-Meier median survival per model and the multi-group log-rank p-value across the three models.
5. Survival by response: log-rank p-value between responders and non-responders.
6. Cox proportional-hazards models (Wald statistics, 95% CI on the log-hazard scale):
   - univariable, one covariate at a time: `suv_max`, `suv_mean`, `roi_volume_mm3`;
   - multivariable: `suv_max` + `roi_volume_mm3`, together with the null partial log-likelihood (intercept-only partial likelihood, i.e. minus the sum over event times of the log number at risk), the full partial log-likelihood, the likelihood-ratio statistic with df = 2, and Harrell's concordance index of the fitted model.
7. Mann-Whitney U tests (two-sided) comparing `suv_max` and `suv_mean` between responders and non-responders.

## Constraints

1. Response scoring is exact: responder iff `day11_volume_mm3 / day0_volume_mm3 < 1.0` (strictly less; no median split, no alternative cut-off).
2. SUV must be computed as `SUV = C_kBq_per_mL * body_weight_g / (1000 * injected_dose_mbq)`; `suv_max` uses the maximum ROI concentration, `suv_mean` the mean ROI concentration.
3. The tumour-volume covariate in every Cox model is the day-14 PET/CT segmented volume `roi_volume_mm3`, not a caliper volume and not a log transform; hazard ratios are reported per 1 mm3.
4. The mixed models must use maximum likelihood (never REML), a random intercept per mouse, and `day` as a continuous variable; each reported LRT p-value must equal `chi2.sf(lr_stat, df)` for the df stated above.
5. Every numeric value in `results.json` must be a JSON number (no strings, no nulls, no NaN/Infinity). p-values must lie in [0, 1]; hazard ratios must be strictly positive and satisfy `hr_ci_low < hr < hr_ci_high`.
6. Self-consistency is mandatory: `hr = exp(coef)`, `hr_ci_low/high = exp(coef -/+ 1.959964 * se)`, `z = coef / se`, the Cox p-value `= 2 * Phi(-|z|)`, and `lr_p = chi2.sf(lr_stat, df)`. An entry whose components contradict each other scores zero even if the point estimates look plausible.
7. All 121 mice must be analysed; do not drop rows, do not modify the input files, and do not regenerate the cohort with a different seed.
8. Risk direction must be internally consistent: if a Cox p-value is below 0.05 the corresponding 95% CI must exclude 1.
9. If a Kaplan-Meier median is not reached within the observed follow-up of a group, report the largest observed `survival_days` in that group instead.
10. Report at least 6 significant digits for every estimate - the grader compares against values recomputed from the data.

## Output

Write a single JSON object to `/app/output/results.json` using exactly these keys. The numbers below are illustrative placeholders and must be replaced by your own results:

```json
{
  "n_mice": 121,
  "n_by_model": {"MOC1": 40, "MOC2": 41, "MLM1": 40},
  "n_responders": 43,
  "n_non_responders": 78,
  "median_relative_volume_day11": {"responder": 0.83, "non_responder": 1.42},
  "growth_model": {
    "model_by_day": {"lr_stat": 38.2, "df": 2, "p": 5.1e-09},
    "response_by_day": {"lr_stat": 121.7, "df": 1, "p": 2.3e-28}
  },
  "survival": {
    "logrank_p_across_models": 3.4e-06,
    "logrank_p_response_groups": 0.00021,
    "median_survival_days": {"MOC1": 31.0, "MOC2": 22.0, "MLM1": 37.0}
  },
  "cox_univariable": {
    "suv_max": {"coef": 0.41, "se": 0.12, "hr": 1.51, "hr_ci_low": 1.19, "hr_ci_high": 1.91, "z": 3.42, "p": 0.00063},
    "suv_mean": {"coef": 0.18, "se": 0.12, "hr": 1.20, "hr_ci_low": 0.95, "hr_ci_high": 1.51, "z": 1.50, "p": 0.13},
    "roi_volume_mm3": {"coef": 0.0074, "se": 0.0013, "hr": 1.0074, "hr_ci_low": 1.0048, "hr_ci_high": 1.0101, "z": 5.69, "p": 1.3e-08}
  },
  "cox_multivariable": {
    "suv_max": {"coef": 0.31, "se": 0.14, "hr": 1.36, "hr_ci_low": 1.03, "hr_ci_high": 1.80, "z": 2.21, "p": 0.027},
    "roi_volume_mm3": {"coef": 0.0061, "se": 0.0014, "hr": 1.0061, "hr_ci_low": 1.0034, "hr_ci_high": 1.0089, "z": 4.36, "p": 1.3e-05},
    "ll_null": -412.6,
    "ll_full": -390.1,
    "lr_stat": 45.0,
    "lr_df": 2,
    "lr_p": 1.6e-10,
    "concordance_index": 0.74
  },
  "mannwhitney_p": {"suv_max": 0.0042, "suv_mean": 0.21}
}
```

## Scoring

The grader recomputes every quantity from the protected seed and awards graded partial credit per check group: schema completeness (0.05), cohort counts and responder classification (0.08), response-group medians (0.07), model x day LRT (0.10), responder x day LRT (0.08), log-rank across models (0.08), log-rank across response groups (0.07), Kaplan-Meier medians (0.07), univariable Cox block (0.15), multivariable Cox block including the likelihood-ratio test (0.15), concordance index (0.05) and Mann-Whitney tests (0.05). Each group is scored as the fraction of its sub-checks passed, with a 5% relative tolerance on coefficients, ~15% on p-values and 0.02 absolute on the concordance index. Final score = weighted fraction of checks passed.
