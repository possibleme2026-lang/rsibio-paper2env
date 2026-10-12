# Reproduce the Fontan EV Score (FES) for Fontan-associated liver disease

## Background

Fontan-associated liver disease (FALD) is a near-universal complication of the Fontan palliation, driven by chronic venous hypertension and progressive hepatic fibrosis. Liver biopsy is impractical for serial surveillance and the serological indices developed for viral hepatitis (APRI, FIB-4) perform poorly in congestive hepatopathy. The Fontan EV Score (FES) instead combines time post-surgery with a panel of circulating small-extracellular-vesicle (sEV) miRNAs inside a proportional-odds (ordinal) logistic regression, and was reported to reach AUC 0.876 for moderate and 0.963 for severe fibrosis in a held-out cohort.

You have been handed the processed cohort of a translational ovine Fontan model: longitudinal serum sEV small-RNA sequencing counts, paired clinical chemistry, elastography-derived fibrosis grading, and a pre-specified train/test split. Re-implement the full analysis chain — index calculation, preprocessing, penalised-regression feature selection, ordinal model fitting, held-out scoring and the benchmark comparison against APRI and FIB-4 — and emit the model artefacts and metrics that the paper reports.

## Inputs

* `/app/data/clinical.csv` — 58 longitudinal serum samples from 19 sheep (Fontan n=13, Sham n=6) with columns `sample_id, sheep_id, group, months_post_surgery, age_years, ast_u_l, alt_u_l, platelets_10e9_l, liver_stiffness_kpa, fibrosis_grade, split`.
  `fibrosis_grade` is the ordinal reference standard (0 = no fibrosis, 1 = mild, 2 = moderate, 3 = severe); `split` is `train` or `test`.
* `/app/data/mirna_counts.csv` — raw sEV small-RNA sequencing counts, 58 rows x 180 miRNA columns, first column `sample_id`. Rows are in exactly the same order as `clinical.csv`.
* `/app/data/gen_inputs.py` — the deterministic generator (seed 20260731) that produced both files; for provenance only, not part of the analysis.

## Task

1. **Clinical indices.** For every sample compute APRI and FIB-4 from the raw clinical chemistry (AST upper limit of normal = 40 U/L):
   * `APRI = (AST / 40.0) / platelets_10e9_l * 100`
   * `FIB4 = (age_years * AST) / (platelets_10e9_l * sqrt(ALT))`
2. **Preprocessing.** Retain all miRNAs with a raw count >= 10 in at least 30% of the 58 samples (i.e. in >= 17.4 samples). Library-size normalise the retained matrix to counts per million (CPM) using *only the retained miRNAs* as the library, then apply `log2(CPM + 1)`.
3. **Feature selection.** On the training split only: z-score standardise every retained miRNA (sample standard deviation, ddof = 1) and fit an L1-penalised logistic regression `sklearn.linear_model.LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=10000, random_state=0)` for the binary outcome `fibrosis_grade >= 2`. Start at `C = 0.5` and divide C by ten (0.5, 0.05, 0.005, 0.0005) until at least 11 miRNAs carry a non-zero coefficient. Rank non-zero miRNAs by decreasing |coefficient| (ties broken by ascending miRNA name) and keep the top 11; report the panel sorted alphabetically.
4. **Ordinal model.** Fit a proportional-odds logistic regression (`statsmodels.miscmodels.ordinal_model.OrderedModel(..., distr='logit')`, optimiser `method='bfgs'`) on the training split with features `[log1p(months_post_surgery)] + panel miRNAs` using the *unstandardised* `log2(CPM+1)` values, and the ordinal outcome `fibrosis_grade` in {0,1,2,3}.
5. **Scoring.** For every sample (train and test) compute the FES from the fitted coefficients `beta` and strictly increasing thresholds `t0 < t1 < t2` with linear predictor `eta = beta . x`:
   * `P(grade <= j) = sigmoid(t_j - eta)`
   * `FES_moderate = P(grade >= 2) = 1 - sigmoid(t1 - eta)`
   * `FES_severe = P(grade >= 3) = 1 - sigmoid(t2 - eta)`
6. **Benchmark.** On the test split only, compute six ROC AUCs: `FES_moderate` and APRI for `grade >= 2`, and `FES_severe` and FIB-4 (and APRI) for `grade >= 3`.
7. **Artefacts.** Write the five output files described below.

## Constraints

1. `clinical_indices.csv` holds exactly the 58 `sample_id`s in the same order as `clinical.csv`, with `apri` and `fib4` matching the stated formulas to a relative tolerance of 1e-4 (report at least 6 decimals).
2. `normalized_expression.csv` has the same 58 rows (same order) and *exactly* the retained miRNA set — no dropped miRNAs, no extra columns — and every cell matches `log2(CPM+1)` computed over the retained miRNAs to an absolute tolerance of 1e-5.
3. `panel.json` reports a panel of exactly 11 unique miRNA names, all of them retained by the step-2 filter, together with the `c_used` value actually used (one of 0.5, 0.05, 0.005, 0.0005).
4. `model.json` reports coefficients for exactly 12 features (`log1p_months` and the 11 panel miRNAs, no others, no intercept) and three strictly increasing thresholds.
5. `predictions.csv` contains all 58 samples with `fes_moderate` and `fes_severe` finite and inside [0, 1], and `fes_severe <= fes_moderate` for every sample.
6. The reported probabilities must be reproducible from your own `model.json`, `normalized_expression.csv` and `clinical.csv` (max absolute deviation <= 1e-3); the grader recomputes them independently.
7. The FES must discriminate on the held-out test split: AUC >= 0.80 for moderate (grade >= 2) and >= 0.85 for severe (grade >= 3) yields full credit; lower bands receive graded partial credit.
8. The six AUCs in `metrics.json` must agree within 0.02 with values recomputed from your own predictions and from the raw clinical chemistry.
9. All five artefacts must be written under `/app/outputs/` with exactly the schemas below (no NaN, no unnamed index columns, no extra files required).
10. The pipeline must be deterministic: re-running it on the same inputs must reproduce identical numbers (fix all random seeds; `liblinear` and the BFGS optimiser are deterministic given the specified settings).

## Output

All files go into `/app/outputs/`.

* `clinical_indices.csv` — header `sample_id,apri,fib4`, 58 rows in input order. Example row: `OV01_03m,0.417534,0.091273`
* `normalized_expression.csv` — first column `sample_id` (unnamed CSV index), then one column per retained miRNA, values `log2(CPM+1)` rounded to 6 decimals, 58 rows in input order.
* `panel.json` — `{"panel": ["miR-125a-5p", ... 11 names sorted alphabetically ...], "c_used": 0.5}`
* `model.json` — `{"coefficients": {"log1p_months": -0.42, "miR-125a-5p": 1.31, ...}, "thresholds": [t0, t1, t2]}`
* `predictions.csv` — header `sample_id,split,fes_moderate,fes_severe`, 58 rows in input order. Example row: `OV01_03m,test,0.741328,0.105772`
* `metrics.json` — `{"test_n": 23, "auc_moderate_fes": 0.93, "auc_severe_fes": 0.97, "auc_moderate_apri": 0.61, "auc_severe_apri": 0.66, "auc_moderate_fib4": 0.72, "auc_severe_fib4": 0.75}` (`test_n` = number of held-out samples; the exact values are for illustration only).

## Scoring

The grader re-implements the reference pipeline from `/app/data` and compares your artefacts. Ten independently graded checks are combined by weight: APRI values (0.07), FIB-4 values (0.07), normalised expression matrix (0.12), panel overlap with the reference elastic-net panel (0.12), ordinal coefficients plus thresholds against an independent refit (0.12), prediction schema/monotonicity (0.08), model-to-prediction self-consistency (0.12), moderate AUC band (0.10), severe AUC band (0.10) and metrics self-consistency (0.10). Most checks award a fraction proportional to the number of rows, cells or coefficients that pass, so partial progress is rewarded; the final score is the weighted sum in [0, 1].
