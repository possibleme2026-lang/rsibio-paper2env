# Benchmarking deep learning predictions of mutation-induced fold switching

You are a computational biologist re-analyzing the GA/GB fold-switching benchmark. The dataset contains experimentally measured fold-state populations for 300 mutants of the GA/GB model system, along with predictions from six computational methods: `alphafold2`, `esmfold`, `rosetta`, `foldx`, `evfold`, and `sparks`. Each method predicts populations for four states: a 3-helix bundle (`p_3`), a 4-beta+alpha fold (`p_4b_plus`), a mixed state (`p_mixed`), and an unfolded state (`p_unfolded`).

## Inputs
- `/app/data/benchmark_data.csv`: 300 rows. Columns: `variant_id`, `position`, `wt_aa`, `mut_aa`, `measured_p_3`, `measured_p_4b_plus`, `measured_p_mixed`, `measured_p_unfolded`, and for each method `<method>_p_3`, `<method>_p_4b_plus`, `<method>_p_mixed`, `<method>_p_unfolded`. Some method predictions are missing (NaN) for ~5% of variants.
- The data is synthetic but realistic, generated deterministically from a fixed seed. The generator script is at `/app/data/gen_data.py` (you may inspect it, but it only generates the raw data, not the analysis results).

## Task
Compute the benchmark metrics comparing each method's predictions to the experimental measurements. Follow these rules exactly:

1. For a given method, only consider variants where **all four** predicted populations for that method are non-NaN. For per-position metrics, additionally filter to the given position.
2. For each method and each state (`p_3`, `p_4b_plus`, `p_mixed`, `p_unfolded`), compute:
   - Root-mean-square error (RMSE) between predicted and measured populations.
   - Spearman rank correlation coefficient between predicted and measured populations.
3. Compute these metrics **overall** (all positions combined) and **per position** for positions 26, 33, and 45.
4. For each of the three key positions (26, 33, 45), determine:
   - The method with the **lowest average RMSE** across the four states.
   - The method with the **highest average Spearman correlation** across the four states.
   - For ties, sort by value first (lowest RMSE, highest Spearman), then alphabetically by method name.
5. Write the results to `/app/output/benchmark_report.json` with the following structure:

```json
{
  "overall_metrics": {
    "alphafold2": {
      "rmse": {"p_3": 0.0, "p_4b_plus": 0.0, "p_mixed": 0.0, "p_unfolded": 0.0},
      "spearman": {"p_3": 0.0, "p_4b_plus": 0.0, "p_mixed": 0.0, "p_unfolded": 0.0}
    },
    ...
  },
  "per_position_metrics": {
    "26": {
      "alphafold2": {
        "rmse": {"p_3": 0.0, ...},
        "spearman": {"p_3": 0.0, ...}
      },
      ...
    },
    "33": { ... },
    "45": { ... }
  },
  "best_methods": {
    "26": {"best_rmse_method": "method_name", "best_spearman_method": "method_name"},
    "33": {...},
    "45": {...}
  }
}
```

## Constraints
1. Output file must exist at `/app/output/benchmark_report.json` and be valid JSON.
2. Top-level keys must be exactly `overall_metrics`, `per_position_metrics`, `best_methods`.
3. `overall_metrics` must contain exactly the six method names, each with `rmse` and `spearman` dicts containing exactly the four state keys.
4. `per_position_metrics` must contain exactly the three position keys "26", "33", "45", each with the same method/state structure.
5. `best_methods` must contain exactly the three position keys, each with `best_rmse_method` and `best_spearman_method` strings that are valid method names.
6. All RMSE values must be in [0, 1].
7. All Spearman values must be in [-1, 1].
8. The overall RMSE for `alphafold2` and state `p_3` must be within 0.01 of the true value computed from the data.
9. The overall Spearman correlation for `esmfold` and state `p_4b_plus` must be within 0.02 of the true value.
10. The best method by average RMSE at position "33" must be correct.
11. The best method by average Spearman at position "45" must be correct.
12. The mean absolute error between your overall RMSE matrix (6 methods × 4 states) and the true matrix must be less than 0.015.
13. The mean absolute error between your overall Spearman matrix (6 methods × 4 states) and the true matrix must be less than 0.02 (ignoring NaN).
14. The mean absolute error between your per-position RMSE matrix (3 positions × 6 methods × 4 states) and the true matrix must be less than 0.02.
15. The mean absolute error between your per-position Spearman matrix (3 positions × 6 methods × 4 states) and the true matrix must be less than 0.025 (ignoring NaN).
16. All best methods (both RMSE and Spearman for all three positions) must match the true best methods.

## Scoring
The grader checks constraints 1–16 independently. Each constraint contributes equally to the final score (partial credit). The score is the fraction of constraints passed. The true metrics are recomputed by the verifier from the raw data using the same rules; your answer is compared against those recomputed values.
