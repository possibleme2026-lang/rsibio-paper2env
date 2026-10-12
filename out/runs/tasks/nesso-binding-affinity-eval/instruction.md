# Benchmark evaluation of Nesso-1 vs Boltz-2 binding affinity predictions

## Context
You are a computational biologist tasked with reproducing the key evaluation from the Nesso-1 paper. Nesso-1 is a fast coarse-grained cofolding model for binding affinity prediction. You are given a set of 30 binding affinity assays, each containing measured pKd values and predictions from two models: Nesso-1 and Boltz-2. Your goal is to compute the evaluation metrics, classify assays as in-distribution (ID) or out-of-distribution (OOD) based on sequence identity to the training set, and analyze selectivity.

## Inputs
- `data/train_proteins.fasta`: 100 protein sequences (the training set) in FASTA format.
- `data/assays/`: Directory with 30 CSV files, named `assay_XX.csv`. Each file has columns: `compound_id`, `smiles`, `protein_id`, `protein_sequence`, `measured_pKd`, `nesso_pred`, `boltz_pred`. Some values may be missing (NaN).
- `data/selectivity.csv`: A CSV with columns: `assay_id`, `compound_id`, `smiles`, `on_protein_id`, `on_protein_sequence`, `on_measured_pKd`, `on_nesso_pred`, `on_boltz_pred`, `off_protein_id`, `off_protein_sequence`, `off_measured_pKd`, `off_nesso_pred`, `off_boltz_pred`. Contains 5 selectivity assays, each with 50 compounds measured against an on-target and an off-target protein.

## Task
Write a script that:
1. Parses all assay CSV files in `data/assays/`.
2. For each assay, computes the Spearman correlation and mean absolute error (MAE) between measured pKd and predicted pKd for both Nesso-1 and Boltz-2. Ignore compounds where either measured or predicted value is missing.
3. For each assay, compute the maximum sequence identity of the assay protein to any protein in `data/train_proteins.fasta`. Use Biopython's `PairwiseAligner` with the following parameters: match_score=1, mismatch_score=0, open_gap_score=-2, extend_gap_score=-1. Compute identity as the fraction of aligned positions that are identical (matches / total aligned positions). Classify the assay as OOD if max identity < 0.40, otherwise ID.
4. Aggregate the mean Spearman correlation across ID assays and across OOD assays for both models.
5. Identify assays where Nesso-1 outperforms Boltz-2 by at least 0.1 in Spearman correlation (nesso_spearman - boltz_spearman >= 0.1).
6. Identify assays where Nesso-1 has Spearman correlation < 0.3 (failure cases).
7. For the selectivity data, for each assay (`assay_id`), compute the true selectivity delta as `on_measured_pKd - off_measured_pKd`, and predicted deltas for both models. Compute the Spearman correlation between true delta and each predicted delta across compounds. Ignore compounds with any missing values.
8. Write the results to `/app/results/report.json` with the exact schema described below.

## Constraints
1. All metrics must be rounded to 4 decimal places.
2. OOD classification threshold is strictly `< 0.40` maximum sequence identity.
3. Spearman correlations must be computed using `scipy.stats.spearmanr`.
4. MAE must be computed using `sklearn.metrics.mean_absolute_error` or equivalent.
5. Sequence identity must be computed using Biopython's `PairwiseAligner` with the specified parameters.
6. All 30 assays must be included in `per_assay` with their `assay_id` as keys.
7. Missing values must be dropped pairwise (i.e., for Nesso-1 metrics, drop rows where measured or nesso_pred is NaN; for Boltz-2 metrics, drop rows where measured or boltz_pred is NaN).
8. The output JSON must contain the keys: `per_assay`, `aggregates`, `nesso_outperforms`, `nesso_failures`, `selectivity`.
9. `nesso_outperforms` and `nesso_failures` must be sorted lists of assay IDs.
10. `selectivity` must contain all 5 selectivity assay IDs from `data/selectivity.csv`.

## Output
Write a JSON file to `/app/results/report.json`. The schema is:

```json
{
  "per_assay": {
    "assay_00": {
      "nesso_spearman": 0.8123,
      "boltz_spearman": 0.7654,
      "nesso_mae": 0.4567,
      "boltz_mae": 0.5123,
      "max_identity": 0.9234,
      "is_ood": false
    },
    ...
  },
  "aggregates": {
    "mean_spearman_id_nesso": 0.8000,
    "mean_spearman_id_boltz": 0.7500,
    "mean_spearman_ood_nesso": 0.6500,
    "mean_spearman_ood_boltz": 0.5000
  },
  "nesso_outperforms": ["assay_10", "assay_11"],
  "nesso_failures": ["assay_05"],
  "selectivity": {
    "sel_00": {
      "nesso_spearman": 0.7500,
      "boltz_spearman": 0.6000
    },
    ...
  }
}
```

## Scoring
The grader recomputes all metrics from the input data and compares your report. Each check is worth an equal fraction of the total score. Checks include:
- Presence and correct values (within 0.01) for all per-assay metrics (`nesso_spearman`, `boltz_spearman`, `nesso_mae`, `boltz_mae`, `max_identity`).
- Correct `is_ood` boolean for all assays.
- Correct aggregate mean Spearman values.
- Exact match of `nesso_outperforms` and `nesso_failures` lists.
- Correct selectivity Spearman values for all selectivity assays.
Partial credit is given per check.
