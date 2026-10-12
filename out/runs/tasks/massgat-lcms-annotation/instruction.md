# MassGAT-style LC-MS peak detection and annotation

You are implementing a graph-based collective learning approach for untargeted LC-HRMS data, inspired by MassGAT. Your goal is to detect which candidate features are valid signals from real compounds and annotate their ion species (adduct and isotope) by reasoning over the graph of m/z and retention time relationships.

## Inputs

- `/app/data/sample_features.csv`: a sample feature table with columns `feature_id,mz,rt,intensity`. `mz` is the mass-to-charge ratio, `rt` is retention time in minutes, `intensity` is the ion intensity. The sample was generated with seed 42, 30 compounds, and 100 noise features.
- `/app/data/sample_labels.csv`: ground-truth labels for the sample (columns `feature_id,is_valid,compound_id,ion_type`). You can use this for training/validation.
- `/app/data/gen_data.py`: a deterministic simulator you can run to generate more labeled data. Example: `python /app/data/gen_data.py --seed 123 --n_compounds 50 --n_noise 200 --out_features /tmp/features.csv --out_labels /tmp/labels.csv`. The hidden test will use a different seed and parameter set.

## Chemical transformations

The following adduct and isotope mass shifts are used (in Da):
- `M+H`: +1.007276
- `M+Na`: +22.989218
- `M+NH4`: +18.033823
- `M+K`: +38.963158
- Isotopes: `iso0`: +0.0, `iso1`: +1.003355, `iso2`: +2.006710

An ion type is denoted by its adduct and isotope, e.g., `M+H_iso0`, `M+Na_iso1`, `M+K_iso2`. The allowed ion types are all 12 combinations of the 4 adducts and 3 isotopes, plus `noise` for invalid features.

## Task

Write a Python script `/app/predict.py` that takes two command-line arguments: `<input_features.csv>` and `<output_predictions.csv>`. It must read the feature table, perform graph-based collective detection and annotation, and write predictions.

Your algorithm should:
1. Build a graph where nodes are features. Connect two features if their retention times are within 0.1 min and their m/z difference matches one of the expected adduct/isotope transformations within 5 ppm.
2. Find connected components.
3. For each component, infer a consistent neutral mass and assign each feature an ion type. A feature is valid (`is_valid=1`) if it can be explained by a neutral mass that is supported by at least two features in the component. Features that cannot be explained should be marked as `is_valid=0`.
4. Assign a `compound_id` to each valid feature. The IDs are arbitrary but must be consistent: all features from the same compound must share the same positive integer, and features from different compounds must have different IDs. Noise features must have `compound_id=-1` and `ion_type=noise`.

## Constraints

1. The output CSV must have the exact header `feature_id,is_valid,compound_id,ion_type`.
2. Every `feature_id` from the input must appear exactly once in the output.
3. `is_valid` must be 0 or 1. For `is_valid=0`, `compound_id` must be -1 and `ion_type` must be `noise`. For `is_valid=1`, `compound_id` must be a positive integer and `ion_type` must be one of the 12 allowed ion types.
4. Detection F1 on the hidden test must be >= 0.90.
5. Annotation accuracy (ion type correctness among gold-valid features) must be >= 0.85.
6. Compound grouping pairwise F1 must be >= 0.85.
7. Mass consistency: for each predicted compound group with >=2 features, the implied neutral masses (computed as `mz - adduct_shift - isotope_shift`) must have a maximum deviation <= 10 ppm. At least 90% of such groups must satisfy this.
8. Your script must complete on the hidden test within 300 seconds.

## Output

Write the predictions to the path given as the second argument. Example of a correct output row:

```
feature_id,is_valid,compound_id,ion_type
12,1,3,M+H_iso0
13,1,3,M+H_iso1
14,0,-1,noise
```

## Scoring

The grader checks the 8 constraints above independently. Your score is the fraction of constraints passed. Partial credit is awarded per constraint.
