# SNP-propagated TF activity in ulcerative colitis

In ulcerative colitis (UC), non-coding genetic variants perturb transcription factor (TF) regulatory networks. We provide synthetic genotype, SNP-TF mapping, TF interaction network, and clinical severity data for 200 patients. Your task is to implement an iSNP-like network propagation pipeline to compute per-patient TF activity scores, correlate them with severity, and identify the top perturbed TFs.

## Inputs
- `/app/data/genotypes.tsv`: 200 samples x 50 SNPs. Tab-separated. First column `sample_id`, then `SNP_001`...`SNP_050`. Values are 0, 1, or 2.
- `/app/data/snp_tf_map.tsv`: 50 rows. Columns: `snp_id`, `tf_id`, `effect_weight`. Each SNP maps to one TF.
- `/app/data/tf_network.tsv`: 20x20 matrix. First column is TF name, header row has TF names. Values are interaction weights.
- `/app/data/clinical.tsv`: 200 rows. Columns: `sample_id`, `severity`.

## Task
1. Compute the initial TF perturbation matrix `P` (200 samples x 20 TFs). For each sample and TF, sum over all SNPs mapped to that TF: `P[s, t] = sum(genotype[s, snp] * effect_weight[snp])`.
2. Load the TF network `W`. Row-normalize it: for each row i, divide by the sum of that row (if sum > 0). Let `W_norm` be the row-normalized matrix.
3. Propagate each sample's perturbation using the closed-form solution: `F = (1 - alpha) * (I - alpha * W_norm)^-1 * P`, with `alpha = 0.7`. Use `numpy.linalg.solve` or `numpy.linalg.inv`.
4. Z-score each TF's propagated scores across the 200 samples: `Z[s, t] = (F[s, t] - mean_t) / std_t`, where `std_t` is the population standard deviation (ddof=0).
5. Compute Pearson correlation between each TF's z-scored activity and `severity` across the 200 samples. Also compute the two-sided p-value.
6. Write the three output files as described below.

## Constraints
1. `/app/output/tf_activity.tsv` must have exactly 200 rows and 21 columns: `sample_id` plus 20 TF columns named `TF_01`...`TF_20`. Values must be z-scored per TF: mean ~0, standard deviation ~1 (tolerance 0.01).
2. `/app/output/correlations.tsv` must have exactly 20 rows and columns `tf`, `pearson_r`, `p_value`. The values must match the grader's recomputation within 1e-4.
3. `/app/output/top_tfs.tsv` must have columns `rank`, `tf`, `abs_r`. It must list the top 3 TFs by absolute Pearson correlation, sorted descending by `abs_r`, with ranks 1, 2, 3.
4. All output files must be tab-separated, have the specified headers, and contain no NaN values.
5. The propagation must use `alpha = 0.7` and the row-normalized network exactly as described.

## Output
- `/app/output/tf_activity.tsv`
- `/app/output/correlations.tsv`
- `/app/output/top_tfs.tsv`

## Scoring
The grader recomputes the expected outputs from the input files. It checks each constraint independently. Score is the fraction of constraints passed (0 to 1). Partial credit is given.
