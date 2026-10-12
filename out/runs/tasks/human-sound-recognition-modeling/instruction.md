# Predicting Human Recognition Accuracy for Environmental Sounds

## Context
Human recognition of environmental sounds is critical for daily life, but computational models of this ability remain underdeveloped. In this task, you will build a model that predicts human recognition accuracy (proportion of correct identifications) for environmental sound clips. The dataset contains acoustic features and metadata for clips that vary by source category, acoustic distortion, number of concurrent sources, and signal-to-noise ratio (SNR). Your model will be evaluated on a held-out test set of clips.

## Inputs
- `/app/data/train.csv`: 400 rows with columns: `clip_id`, `category`, `distortion`, `n_sources`, `snr_db`, `f1`...`f12`, `human_accuracy`. The `human_accuracy` column is the target (between 0 and 1).
- `/app/data/test.csv`: 200 rows with the same columns except `human_accuracy` is omitted.
- `/app/data/features.md`: Description of the 12 acoustic features (`f1`...`f12`).
- The data was generated deterministically. The generator script is not available at solve time.

## Task
1. Explore the training data and feature descriptions.
2. Build a regression model to predict `human_accuracy` from the provided features and metadata (`category`, `distortion`, `n_sources`, `snr_db`).
3. Use proper validation (e.g., cross-validation or a held-out set) to select and tune your model.
4. Generate predictions for all 200 test clips.
5. Write predictions to `/app/output/predictions.csv` with exactly two columns: `clip_id`, `predicted_accuracy`.
6. Write a summary to `/app/output/model_summary.json` with keys `validation_rmse` (float) and `model_type` (string).

## Constraints
1. `predictions.csv` must have exactly 200 rows, with the same `clip_id` values as `test.csv` in the same order.
2. `predicted_accuracy` must be a float in [0, 1] and must not contain NaN.
3. `model_summary.json` must exist and contain the keys `validation_rmse` and `model_type`.
4. Your predictions must achieve a Pearson correlation >= 0.60 with the true human accuracy on the test set.
5. Your predictions must achieve a Pearson correlation >= 0.75 with the true human accuracy on the test set.
6. Your predictions must achieve an RMSE <= 0.15 on the test set.
7. Your predictions must achieve an RMSE <= 0.12 on the test set.
8. Your predictions must achieve a mean absolute error <= 0.10 on the test set.
9. The standard deviation of your predictions must be > 0.02 (i.e., not constant).
10. You must not use the `clip_id` as a feature.

## Output
`/app/output/predictions.csv`:
```
clip_id,predicted_accuracy
clip_0400,0.82
clip_0401,0.75
...
```
`/app/output/model_summary.json`:
```json
{"validation_rmse": 0.062, "model_type": "RandomForestRegressor"}
```

## Scoring
The grader checks each of the 10 constraints independently. Score = (number of constraints passed) / 10. Partial credit is given for each threshold met. The primary metric is Pearson correlation.
