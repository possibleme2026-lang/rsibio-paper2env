# Longitudinal Functional Connectivity Analysis in SCA7

You are a computational neuroscientist analyzing resting-state fMRI network-to-network functional connectivity (FC) from a longitudinal cohort of spinocerebellar ataxia type 7 (SCA7) patients and healthy controls. The data are in `/app/data/`. Each subject has three visits (0, 1, 2 corresponding to 0, 12, 24 months). FC values are Fisher z-transformed network-to-network connectivity between 8 cortical and cerebellar networks.

## Inputs
- `/app/data/subjects.csv`: subject_id, group, age, sex.
- `/app/data/visits.csv`: subject_id, visit (0,1,2), months, age_at_visit, group, group_code, MMSE, MoCA, SARA.
- `/app/data/fc_longitudinal.csv`: subject_id, visit, edge_id, network_i, network_j, fc_z. Contains 28 unique edges across 96 sessions.
- `/app/data/edges.csv`: list of the 28 edge IDs.
- `/app/data/gen_inputs.py`: deterministic generator (seed 20260215) used to create the data.

## Task
Perform the following analyses and write all outputs to `/app/output/`:

1. **Longitudinal Group×Visit interaction**: For each of the 28 edges, fit a linear mixed-effects model with fixed effects `group_code`, `visit` (0,1,2), and their interaction, plus a random intercept per subject. Use `fc_z` as the outcome. Extract the interaction coefficient (`beta_group_visit`), standard error, t-statistic, and p-value. Apply Benjamini-Hochberg FDR correction across the 28 edges. Sort results by p-value ascending. Mark edges with FDR < 0.05 as significant.

2. **Clinical associations**: For the baseline visit (visit=0), compute the Visual–DefaultMode dysconnectivity score as the negative of the Visual–DefaultMode `fc_z` value (`dysconnectivity = -fc_z`). Compute Pearson correlations between this score and MMSE, MoCA, and SARA. Report r, two-sided p-value, and n.

3. **Classification**: Using baseline visit (visit=0) data, train a classifier to distinguish SCA7 from controls using all 28 edge FC features. Use a pipeline of `StandardScaler` followed by `LogisticRegression(penalty='l2', C=0.1, solver='liblinear', random_state=0, max_iter=1000)`. Evaluate with `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)`. For each subject, record the out-of-fold predicted group (0=Control, 1=SCA7) and the predicted probability of SCA7. Compute accuracy, precision, recall, F1, and the confusion matrix. Fit the same pipeline on all baseline data and extract the logistic regression coefficients for all 28 edges; rank edges by absolute coefficient (descending) to identify top features.

4. **Reproducible script**: Write `/app/output/analysis.py` that takes `--input_dir` (default `/app/data`) and `--output_dir` (default `/app/output`) and runs the entire analysis, producing all output files below.

## Constraints
1. `interaction_results.csv` must contain exactly 28 rows (one per edge), columns: `edge_id`, `network_i`, `network_j`, `beta_group_visit`, `se`, `t`, `p_value`, `p_fdr`, `significant`. Sorted by `p_value` ascending. `significant` is boolean.
2. The Visual–DefaultMode and Somatomotor–Cerebellar edges must have `p_fdr < 0.05` and `beta_group_visit < 0` (greater decline in SCA7 relative to controls).
3. `clinical_associations.csv` must contain three rows (MMSE, MoCA, SARA) with columns `score`, `edge_id`, `r`, `p_value`, `n`. The dysconnectivity score must be negatively correlated with MMSE and MoCA (r < 0, p < 0.05) and positively correlated with SARA (r > 0, p < 0.05).
4. `predictions.csv` must have 32 rows (baseline subjects) with columns `subject_id`, `true_group`, `predicted_group`, `probability_SCA7`. `true_group` and `predicted_group` use 0=Control, 1=SCA7. `probability_SCA7` is in [0,1].
5. `classification_metrics.json` must contain keys `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `cv_folds`. Accuracy and F1 must be >= 0.80. Reported metrics must match metrics recomputed from `predictions.csv` within 0.02.
6. `top_features.csv` must list all 28 edges ranked by absolute logistic regression coefficient (descending) with columns `edge_id`, `coefficient`, `abs_coefficient`, `rank`. The top 5 must include Visual–DefaultMode and Somatomotor–Cerebellar.
7. `analysis.py` must exist, accept `--input_dir` and `--output_dir`, and reproduce the outputs when run in a clean directory.

## Output
Write files to `/app/output/`:
- `interaction_results.csv`
- `clinical_associations.csv`
- `classification_metrics.json`
- `predictions.csv`
- `top_features.csv`
- `analysis.py`

Example row for `interaction_results.csv`:
`Visual-DefaultMode,Visual,DefaultMode,-0.0612,0.018,-3.40,0.0012,0.012,True`
(Note: values are illustrative; compute from data.)

Example row for `predictions.csv`:
`SCA7_01,1,1,0.92`

## Scoring
The grader checks each constraint independently and awards partial credit: score = (number of passed constraints) / (total constraints). Constraints include file existence, schema, numeric tolerances, biological sign expectations, and reproducibility.
