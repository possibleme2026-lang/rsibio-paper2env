# Music-responsive tear proteome signature and classifier

You are analyzing a quantitative tear proteomics dataset from a prospective multi-cohort study of musical stimulation. The dataset contains paired pre- and post-stimulation tear samples from two independent cohorts (A and B), including music-stimulated and unstimulated control participants. Your goal is to reproduce the paper's analysis: identify a reproducible protein signature, test functional enrichment, and build a classifier that discriminates pre- from post-stimulation samples.

## Inputs
- `/app/data/tear_proteomics_train.tsv`: discovery dataset. 336 rows (samples) x 200 protein columns plus metadata. Columns: `sample_id`, `participant_id`, `cohort` (A/B), `condition` (stimulated/control), `timepoint` (pre/post). Protein intensities are log2 LFQ. Missing values are `NA`. True sizes: 200 proteins, 336 samples, ~5% missing values.
- `/app/data/blind_test_proteomics.tsv`: blinded validation dataset. 112 rows x 200 protein columns + `sample_id`. No cohort/condition/timepoint labels. Same protein columns as training.
- `/app/data/pathways.tsv`: protein-pathway annotations. Columns: `pathway`, `protein`. 200 proteins mapped to 20 pathways (some proteins in multiple pathways).
- `/app/data/README.md`: additional notes.

## Task
1. **Differential expression.** For each cohort (A, B) and condition (stimulated, control), compute paired pre-vs-post log2 fold changes (post - pre) and paired two-sided t-test p-values. Use only participants with both pre and post values for a given protein. Filter out proteins with >20% missing values in any of the four cohort/condition groups. Apply Benjamini-Hochberg FDR correction within each cohort/condition across all tested proteins.
2. **Reproducible signature.** Define the reproducible signature as proteins with |log2FC| >= 1.0 and FDR < 0.05 in both stimulated cohorts, the same direction of change in both stimulated cohorts, and FDR >= 0.05 in both control groups. Write the signature to `/app/output/signature.tsv`.
3. **Functional enrichment.** Using `pathways.tsv`, perform a hypergeometric test (one-sided, over-representation) for the signature proteins in each pathway. Use all proteins that pass the missingness filter as the universe. Write the top 3 enriched pathways to `/app/output/enrichment.tsv`.
4. **Classifier.** Using only the signature proteins, train a logistic regression classifier to discriminate pre- from post-stimulation samples in the stimulated training participants. Use standardization and `LogisticRegression(solver='liblinear', penalty='l2', C=1.0, random_state=42)`. Predict the probability of post-stimulation for each sample in `blind_test_proteomics.tsv`. Write predictions to `/app/output/predictions.csv`.
5. **Validation.** The grader will evaluate the signature, enrichment, and classifier predictions against held-out labels for the blind test set. No labels are provided to you.

## Constraints
1. `signature.tsv` must contain exactly 14 unique proteins.
2. `signature.tsv` columns: `protein`, `log2FC_A_stim`, `FDR_A_stim`, `log2FC_B_stim`, `FDR_B_stim`, `direction`, `log2FC_A_ctrl`, `FDR_A_ctrl`, `log2FC_B_ctrl`, `FDR_B_ctrl`. Values must match reference within tolerance: log2FC absolute difference <= 0.25, FDR < 0.05 for stimulated, FDR >= 0.05 for control, direction `up` or `down` consistent.
3. `enrichment.tsv` must contain exactly 3 rows with columns `pathway`, `p_value`, `overlap_count`. Top 3 pathways must match reference set and be ordered by p_value ascending (ties by pathway name).
4. `predictions.csv` must contain 112 rows with columns `sample_id`, `post_probability`. Probabilities in [0,1].
5. Classifier AUROC on stimulated blind samples >= 0.85.
6. Classifier AUROC on control blind samples <= 0.65 (no effect in unstimulated controls).
7. All output files must be in `/app/output/` and use the specified delimiters (`.tsv` tab-separated, `.csv` comma-separated).

## Output
Example `signature.tsv` row (tab-separated):
`protein	log2FC_A_stim	FDR_A_stim	log2FC_B_stim	FDR_B_stim	direction	log2FC_A_ctrl	FDR_A_ctrl	log2FC_B_ctrl	FDR_B_ctrl`
`PRTN3	1.82	1.2e-12	1.75	3.4e-11	up	0.05	0.87	-0.03	0.92`

`enrichment.tsv` example row:
`pathway	p_value	overlap_count`
`Innate Immune Response	1.2e-08	5`

`predictions.csv` example row:
`sample_id,post_probability`
`BL_A_stimulated_000_pre,0.12`

## Scoring
Each constraint is checked independently; score = fraction of constraints passed.
