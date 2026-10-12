# Cerebellar substrates of explicit Theory of Mind and nonverbal action prediction in early childhood

You are a computational neuroimaging analyst working on a developmental study of social cognition. The study collected structural MRI and behavioral data from 80 children aged 3-4 years. Your goal is to reproduce the key analysis linking cerebellar gray matter volume to two dissociable components of social cognition: explicit Theory of Mind (ToM) and nonverbal action prediction.

## Inputs

All data are in `/app/data/`:

- `cerebellar_gmv.csv`: 80 rows (subjects) x 25 columns. First column `subject_id`; remaining 24 columns are regional gray matter volumes (mm³) for cerebellar lobules/regions.
- `behavior.csv`: `subject_id`, `age_months`, `sex` (M/F), `site` (1/2/3), `explicit_tom_score` (0–20), `action_prediction_accuracy` (0–1), `tiv_mm3` (total intracranial volume in mm³).
- `cortical_network_scores.csv`: `subject_id` and 7 z-scored cerebral network scores: `Visual`, `Somatomotor`, `DorsalAttention`, `VentralAttention`, `Limbic`, `Frontoparietal`, `DefaultMode`.
- `atlas_labels.csv`: `region_id`, `region_name`, `lobule`, `hemisphere`, `prior_network` (prior biological annotation: `ToM`, `ActionObservation`, `Other`).

## Task

Perform the following analysis steps and write your results to `/app/results.json`.

1. **Data preparation**: Merge the three data files on `subject_id`. Verify there are no missing values. Do not drop any subjects.

2. **Regional association analysis**: For each of the 24 cerebellar regions, fit an ordinary least squares (OLS) linear model separately for each behavioral measure (`explicit_tom_score` and `action_prediction_accuracy`). The model formula is:
   `behavior ~ GMV_region + age_months + C(sex) + C(site) + tiv_mm3`
   Extract the coefficient (`beta`), t-statistic (`t`), and p-value (`p`) for the `GMV_region` term.

3. **Multiple comparison correction**: For each behavioral measure, apply Benjamini-Hochberg false discovery rate (FDR) correction across the 24 regions. Select regions with q < 0.05. These are your significant region sets.

4. **Dissociation test**: Compute the intersection size between the significant ToM regions and significant action prediction regions.

5. **Cross-validated prediction**: Using the significant regions for each measure as features, evaluate out-of-sample prediction with 5-fold cross-validation. Standardize the features (z-score) before cross-validation. Fit a partial least squares (PLS) regression with 2 components on the training folds, predict the test fold, and compute the coefficient of determination (R²). Report the mean R² across the 5 folds.

6. **Network correlation**: For the significant ToM regions, compute each subject's mean z-scored GMV across those regions. Compute the Pearson correlation between this mean and the `DefaultMode` network score. For the significant action prediction regions, compute the mean z-scored GMV and correlate with the `VentralAttention` network score. Report the correlation coefficient `r` and p-value `p`.

7. **Output**: Write `/app/results.json` with the following structure:
```json
{
  "tom_regions": ["<region_name>", ...],
  "action_regions": ["<region_name>", ...],
  "dissociation_intersection": 0,
  "tom_cv_r2": 0.42,
  "action_cv_r2": 0.38,
  "tom_network_correlation": {"network": "DefaultMode", "r": 0.55, "p": 0.0001},
  "action_network_correlation": {"network": "VentralAttention", "r": 0.48, "p": 0.0002},
  "tom_fdr_table": [
    {"region": "<region_name>", "beta": 1.2, "t": 4.5, "p": 0.00001, "q": 0.0001},
    ... all 24 regions ...
  ],
  "action_fdr_table": [
    {"region": "<region_name>", "beta": 1.1, "t": 4.2, "p": 0.00002, "q": 0.0002},
    ... all 24 regions ...
  ]
}
```

## Constraints

1. The output must be valid JSON at `/app/results.json` with all keys shown above.
2. For explicit ToM, the significant set must include at least one Crus II region (left and/or right) and must NOT include any region from lobule VIIB or the posterior vermis (Vermis_VI, Vermis_VII, Vermis_VIII).
3. For action prediction, the significant set must include at least one VIIB region and at least one posterior vermis region (Vermis_VI, Vermis_VII, or Vermis_VIII), and must NOT include any Crus II region.
4. The intersection of the two significant sets must be empty (dissociation_intersection == 0).
5. Both cross-validated R² values must be > 0.15.
6. The ToM network correlation must be with `DefaultMode`, with r > 0.3 and p < 0.05.
7. The action network correlation must be with `VentralAttention`, with r > 0.3 and p < 0.05.
8. The FDR tables must contain all 24 regions for each behavioral measure, with p and q values matching your OLS and BH FDR computations.

## Scoring

The grader will recompute the reference analysis on the same data and check each constraint individually. Partial credit is given as the fraction of constraints passed. The reward is the same as the score.
