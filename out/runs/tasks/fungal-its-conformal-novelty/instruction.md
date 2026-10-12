# Calibrated novelty flag for fungal ITS metabarcoding

You are working on a fungal ITS metabarcoding project. Environmental surveys recover sequences that cannot be assigned to known genera. The goal is to flag such sequences as novel with a controlled false novelty rate of 5% (alpha = 0.05), using a conformal novelty flag. The reference database contains 100 known genera, and each query sequence is scored under four methods (alignment identity, k-mer bootstrap consensus, and two neural classifiers) for two amplicon regions (ITS1, ITS2).

## Inputs
- `/app/data/reference_db.tsv`: reference sequences. Columns: ref_id, genus, region, sequence.
- `/app/data/calibration_scores.tsv`: 4000 rows (2 regions x 4 methods x 500 sequences). Columns: cal_id, region, method, score, genus. All calibration sequences are from known genera.
- `/app/data/validation_scores.tsv`: 4000 rows, same structure, held-out known-genus sequences.
- `/app/data/query_scores.tsv`: 16000 rows (2000 queries x 2 regions x 4 methods). Columns: query_id, region, method, score. No labels.
- `/app/data/soil_asvs.tsv`: 500 soil ASVs with alignment/ITS1 scores and abundances. Columns: asv_id, region, method, score, abundance.

## Task
1. Implement the conformal novelty flag. For a given region and method, let the calibration scores be an array of n values. For a query score s, define the nonconformity as `1 - s`. The p-value is `p = (1 + C(s)) / (n + 1)`, where `C(s)` is the number of calibration scores <= s. A query is flagged novel if `p <= 0.05`.
2. Compute the threshold `t` for each region and method. Sort the calibration scores ascending. Let `r = floor(0.05 * (n + 1))`. The threshold is the `r`-th smallest calibration score (1-indexed). Flag novel if `score < t`.
3. Apply the flag to `query_scores.tsv` and write `/app/output/novelty_flags.tsv` with columns: query_id, region, method, p_value, novel_flag. Sort by query_id, region, method. Round p_value to 6 decimal places. novel_flag is 1 if flagged, else 0.
4. Write `/app/output/thresholds.json`: a JSON object with keys `"{region}|{method}"` (e.g., `"ITS1|alignment"`) and values as floats of the thresholds.
5. For each region and method, compute the empirical false novelty rate on `validation_scores.tsv` (all known) as the proportion of validation rows flagged novel. Write `/app/output/coverage_report.json` with keys `"{region}|{method}"` and values as floats rounded to 4 decimals.
6. Apply the alignment/ITS1 threshold to `soil_asvs.tsv`. Write `/app/output/soil_flags.tsv` with columns: asv_id, region, method, p_value, novel_flag, abundance. Sort by asv_id. Then write `/app/output/soil_summary.json` with:
   - `fraction_novel_all`: proportion of all 500 soil ASVs flagged novel (novel_flag == 1).
   - `fraction_novel_abundant`: proportion of ASVs with abundance >= 10 that are flagged novel.
   Both rounded to 4 decimals.

## Constraints
1. `novelty_flags.tsv` must contain exactly 16000 rows with unique (query_id, region, method) keys. p_value in [0,1], novel_flag in {0,1}, sorted by query_id, region, method.
2. For every row in `novelty_flags.tsv`, the p_value must match the exact conformal formula using the provided calibration scores, within 1e-6.
3. `thresholds.json` must contain the correct threshold for each of the 8 region/method combinations, within 1e-6.
4. On `validation_scores.tsv`, the empirical false novelty rate for each region/method must be <= 0.07. Your `coverage_report.json` must match the true rates within 0.01.
5. On the hidden true labels for `query_scores.tsv`, the false novelty rate among known-genus queries must be <= 0.08 for each region/method.
6. The detection rate (proportion of truly novel queries flagged) must be >= 0.10 for at least 6 of the 8 region/method combinations, and the overall detection rate across all query rows must be >= 0.15.
7. The precision of novel flags (fraction of flagged queries that are truly novel) must be >= 0.50 overall.
8. `soil_summary.json` must report `fraction_novel_all` and `fraction_novel_abundant` within 0.005 of the true fractions computed from the gold alignment/ITS1 threshold. The true fractions are expected in [0.15, 0.25] and [0.10, 0.20] respectively.

## Output
- `/app/output/novelty_flags.tsv`
- `/app/output/thresholds.json`
- `/app/output/coverage_report.json`
- `/app/output/soil_flags.tsv`
- `/app/output/soil_summary.json`

## Scoring
The grader checks each constraint independently. Score = average of the per-constraint scores. Constraint 2 is graded as the fraction of rows with correct p-values. Constraint 3 is graded as the fraction of the 8 thresholds correct. Constraint 4 and 5 are graded as the fraction of region/method pairs meeting the bound and reporting correctly. Constraint 6 is graded as the fraction of 9 checks (8 region/method pairs with detection >= 0.10, plus overall detection >= 0.15). Constraints 1, 7, 8 are pass/fail. Partial credit is given as described. The final reward is the score.
