# Predicting Long-Term Stability of CHO Cells from Epigenetic Marks

## Background
Chinese hamster ovary (CHO) cells are the dominant production platform for recombinant protein biopharmaceuticals. However, their long-term stability under perfusion culture is variable and difficult to predict. Previous work has shown that epigenetic changes—specifically chromatin modification levels and DNA methylation—are associated with stability. Your task is to build a machine learning model that predicts whether a CHO clone will remain stable and highly productive over long-term culture, using only epigenetic features.

## Inputs
- `/app/data/train.csv`: 1,000 training samples. Columns: `sample_id`, 350 epigenetic features (e.g., `GENE01_H3K4me3`), `batch_id`, `passage_number`, and the binary target `stable` (1 = stable/highly productive, 0 = unstable).
- `/app/data/test_features.csv`: 400 test samples. Same `sample_id` and 350 epigenetic feature columns. The `stable` column and metadata columns (`batch_id`, `passage_number`) are NOT present.
- `/app/data/feature_metadata.csv`: maps each feature to its gene and epigenetic mark.
- `/app/data/README.md`: additional data details.

The data are synthetic but designed to mimic real CHO epigenetic profiles. Some feature values are missing (approx. 3%). The training metadata columns (`batch_id`, `passage_number`) are not available at test time and must not be used for prediction.

## Task
1. Load and explore the data. Handle missing values and any necessary scaling.
2. Train a machine learning model (the paper found multilayer perceptrons to work best, but you may use any model) to predict `stable` from the epigenetic features. Use proper validation to avoid overfitting.
3. Predict the stability label and the probability of stable for each of the 400 test samples.
4. Compute feature importances (e.g., permutation importance or SHAP values) to identify which epigenetic marks drive predictions.
5. Save your predictions and feature importances to the specified output paths.

## Constraints
1. `predictions.csv` must contain exactly 400 rows plus a header, with columns `sample_id`, `predicted_label`, `probability_stable`. The `sample_id` values must exactly match those in `test_features.csv`.
2. `predicted_label` must be integer 0 or 1. `probability_stable` must be a float in [0, 1]. No missing values.
3. The F1 score for the positive class (stable = 1) on the hidden test labels must be >= 0.50.
4. The Matthews correlation coefficient (MCC) must be >= 0.15.
5. Recall for the positive class must be >= 0.55.
6. Precision for the positive class must be >= 0.35.
7. The fraction of predicted positive labels must be between 0.15 and 0.45.
8. `feature_importance.csv` must contain columns `feature` and `importance`, with at least 20 rows. At least 4 of the true causal epigenetic features must appear in your top 20 features by importance.
9. The top 20 features in `feature_importance.csv` must not include the metadata columns `batch_id` or `passage_number`.

## Output
- `/app/output/predictions.csv` with columns `sample_id,predicted_label,probability_stable`. Example:
  ```
  sample_id,predicted_label,probability_stable
  TEST_0001,1,0.87
  TEST_0002,0,0.12
  ```
- `/app/output/feature_importance.csv` with columns `feature,importance`. Example:
  ```
  feature,importance
  GENE12_H3K4me3,0.152
  GENE05_5mC,0.134
  ```

## Scoring
The grader checks the 9 constraints above. Score is the fraction of constraints passed. Partial credit is given for each satisfied constraint.
