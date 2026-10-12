# TM-Vec 2s: Accelerated Protein Remote Homology Detection

You are a computational biologist working on protein remote homology detection. The exponential growth of protein sequence data has created a bottleneck for structural homology detection. TM-Vec 2s is a distilled deep learning model that predicts TM-scores directly from sequence pairs, bypassing expensive structural alignment. In this task, you will implement a sequence-based TM-score predictor and use it to search a protein domain database for remote homologs.

## Inputs

All files are in `/app/data/`:
- `database.fasta`: 200 protein domain sequences (each 128 aa).
- `queries.fasta`: 30 query protein domain sequences (each 128 aa).
- `database_structures.npz`: Cα coordinates for the 200 database domains (shape 200 x 128 x 3). These are provided as a training resource; you may use them to compute structural features for the database.
- `train_pairs.tsv`: 2000 pairs of database domains with experimentally determined TM-scores (columns: `id1`, `id2`, `tm_score`). These labels are computed from the provided structures.

## Task

Build a sequence-based predictor that estimates the TM-score between any query sequence and any database domain. The TM-score measures structural similarity (range 0 to 1, where >0.5 indicates remote homology). You must predict TM-scores for all 30 queries against all 200 database domains.

Your workflow should include:
1. Parse the input files and understand the data.
2. Extract informative features from protein sequences (e.g., amino acid composition, k-mer profiles, physicochemical properties).
3. Train a regression model on the provided training pairs to map sequence-pair features to TM-score.
4. Predict TM-scores for all query-database pairs.
5. Calibrate predictions to the [0,1] range and identify remote homologs (TM-score >= 0.5).

## Constraints

1. Output file must be at `/app/output/predictions.tsv` with exactly three columns: `query_id`, `target_id`, `predicted_tm`. The file must contain exactly 6000 rows (30 queries x 200 targets) plus a header.
2. All `predicted_tm` values must be floating-point numbers in the interval [0, 1].
3. The Spearman correlation between your predicted TM-scores and the true TM-scores (computed by the grader from hidden query structures) must be >= 0.5.
4. The root mean squared error (RMSE) between predicted and true TM-scores must be <= 0.20.
5. For each query, the top-ranked database domain (highest predicted TM-score) must have a true TM-score >= 0.5 for at least 60% of queries.
6. Using a predicted TM-score threshold of 0.5, the average F1 score for remote homolog retrieval (true positives = true TM >= 0.5) across all queries must be >= 0.6.
7. The mean predicted TM-score for true remote homologs (true TM >= 0.5) must exceed the mean predicted TM-score for non-homologs (true TM < 0.5) by at least 0.25.

## Output

Write your predictions to `/app/output/predictions.tsv`. The file must be tab-separated with a header. Example row:
```
query_id	target_id	predicted_tm
Q01	DB001	0.8723
Q01	DB002	0.1234
...
```

## Scoring

The grader will verify each constraint independently. Your score is the fraction of constraints that pass (0 to 1). Partial credit is given for each constraint passed. The final reward is the score.
