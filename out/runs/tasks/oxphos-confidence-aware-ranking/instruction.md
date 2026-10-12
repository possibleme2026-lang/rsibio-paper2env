# Confidence-aware prioritisation of OXPHOS genes in *Caenorhabditis elegans*

Oxidative phosphorylation (OXPHOS) is essential in every *C. elegans* tissue, yet only a small minority of genes carry experimental OXPHOS annotation: most functional assignments rest on indirect or computational evidence and are unreliable. Your job is to exploit the transcriptomic resources in `/app/data` to prioritise new OXPHOS candidates with a confidence-aware learning framework that (i) learns from the reliable annotations first, (ii) incorporates the unreliable ones only after an initial model exists, (iii) removes complex-specific bias by withholding one OXPHOS complex at a time, and (iv) integrates the time-resolved bulk RNA-seq landscape with the embryonic and adult single-cell landscapes.

## Inputs (all paths relative to `/app/data`)

- `expression/bulk_timecourse.tsv` — 800 genes x 16 developmental time points (embryo to adult) of log2-normalised bulk RNA-seq; first column `gene_id`, remaining columns are time points.
- `expression/sc_embryonic.npz`, `expression/sc_adult.npz` — single-cell atlases (log1p-normalised). Each archive holds `X` (cells x genes, dense float32), `genes` (gene ids, giving the column order of `X`), `cells` and `cell_types` (per-cell labels).
- `annotations/high_confidence_oxphos.tsv` — 72 genes with experimental OXPHOS annotation; columns `gene_id, gene_name, complex, evidence`, with `complex` in {Complex I, Complex II, Complex III, Complex IV, Complex V}.
- `annotations/low_confidence_oxphos.tsv` — 132 genes whose OXPHOS annotation rests on indirect/computational evidence (ISS/IEA); same columns. Many of these annotations are wrong.
- `annotations/other_process_annotations.tsv` — 120 genes with experimental annotation to non-OXPHOS processes; columns `gene_id, gene_name, go_term, evidence`. Use them as reliable negatives.
- `candidates/gene_universe.tsv` — the 608 genes to be scored and ranked; columns `gene_id, gene_name, has_low_confidence_oxphos_annotation`.

## Task

1. **Supervised score.** Build a per-gene temporal feature representation from the bulk time course (standardise each gene across time) and train a probabilistic classifier with the 72 high-confidence genes as positives and the 120 experimentally annotated non-OXPHOS genes as negatives.
2. **Two-round training.** After the round-1 model, add the low-confidence genes that round 1 supports above a confidence threshold, retrain, and score the whole candidate universe with the retrained model. Report how many genes were added.
3. **Informed bagging.** Hold out one OXPHOS complex at a time (Complex I..V) — together with the low-confidence genes carrying that complex label — train on the remaining complexes and keep that model's predictions. For every candidate record how many of the five models support it (predicted probability > 0.5).
4. **Co-expression score.** Derive a per-gene co-expression score against the high-confidence OXPHOS set in **both** single-cell atlases and combine the two atlases into one co-expression score. Each atlas carries atlas-specific structure; one atlas alone is not sufficient.
5. **Integration.** Combine the supervised and co-expression evidence into a single final score (both streams must contribute to the ranking) and rank all 608 candidates.

## Constraints

1. `/app/results/candidate_scores.tsv` must be tab-separated with exactly the header `gene_id, score_supervised, score_coexpression, n_modules_supporting, final_score, rank` and exactly 608 data rows, one for every gene in `candidates/gene_universe.tsv` (no missing, duplicated or extra genes); each score column must be a finite number.
2. Rows must be sorted by `final_score` descending and `rank` must run 1..608 in file order.
3. `n_modules_supporting` must be an integer in [0, 5] and must increase with the final ranking (Spearman correlation with `final_score` >= 0.2).
4. `/app/results/shortlist.tsv` must be tab-separated with header `gene_id, rank, final_score` and must contain exactly the 25 top-ranked candidates, in the same order as `candidate_scores.tsv`.
5. `/app/results/report.json` must be a JSON object with keys: `n_round1_positives` (number of genes used in round 1), `n_round2_added` (1 <= value <= 132), `n_module_models` (must equal 5), `per_module_auroc` (object mapping each of the five complex names to that model's held-out AUROC, each float in [0.4, 1.0]), `overall_cv_auroc` (float in [0.5, 1.0]), `n_coexp_reference_genes` (number of high-confidence genes used for the co-expression score).
6. Both evidence streams must genuinely contribute: the Spearman correlation of `final_score` with `score_supervised` and with `score_coexpression` must each be >= 0.15, and the AUROC of `final_score` against the hidden held-out evidence must be at least 0.02 above the better of the two single-stream AUROCs (unless the final AUROC already reaches 0.85).
7. Scientific quality: the AUROC of `final_score` against the hidden held-out evidence for the 608 candidates must be >= 0.62 (full credit at 0.86), and the AUROC restricted to the 132 low-confidence annotated genes must be >= 0.55 (full credit at 0.75).

## Output

Write exactly three files under `/app/results`:

- `candidate_scores.tsv` — example data row (tab separated):
  `WBGene00001234	0.870000	0.610000	4	0.830000	1`
- `shortlist.tsv` — example data row: `WBGene00001234	1	0.830000`
- `report.json` — example: `{"n_round1_positives": 72, "n_round2_added": 38, "n_module_models": 5, "per_module_auroc": {"Complex I": 0.91, "Complex II": 0.89, "Complex III": 0.93, "Complex IV": 0.90, "Complex V": 0.88}, "overall_cv_auroc": 0.90, "n_coexp_reference_genes": 72}`

## Scoring

The verifier regenerates the hidden OXPHOS labels from a protected seed and scores each constraint group independently with partial credit:
format/completeness (weight 0.12), consensus column (0.06), shortlist (0.07), report diagnostics (0.10), evidence integration (0.15), AUROC over all 608 candidates (0.35, linear from 0.62 to 0.86), AUROC over the low-confidence annotated subset (0.15, linear from 0.55 to 0.75). The weighted sum is written as `score`/`reward` to `/logs/verifier/reward.json`.
