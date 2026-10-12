# Circulating miRNA signature of mild SARS-CoV-2 infection: end-to-end qPCR analysis

## Context

A plasma circulating-microRNA profiling study compared individuals with asymptomatic/mild COVID-19 against uninfected controls and reported a set of up-regulated miRNAs (miR-126, miR-146b-5p, miR-223-5p, miR-144-3p, miR-22, miR-146a, miR-30c) with heterogeneous diagnostic performance (best AUC about 0.77), while miR-21-5p and miR-155 were poor discriminators. You are given the raw quantitative-PCR export of that study (synthetic but realistic, produced by a seeded simulator) plus the clinical metadata and a miRNA-target interaction table. Rebuild the complete analysis chain yourself: plate quality control, spike-in normalisation, differential expression with FDR control, ROC/Youden diagnostic metrics, and the target-gene network of the resulting signature.

Every number must be derived from the files below - nothing may be hard-coded, and no network access is available.

## Inputs (all under /app/data)

* `plate_cq.csv` - about 2300 rows, 96-well style Ct export. Columns: `sample_id`, `plate_id`, `well`, `sample_type` (`sample` or `NTC`), `mirna`, `replicate` (1 or 2), `cq`. `cq` is either a number (Ct, 2 decimals) or the string `Undetermined`. The two spike-in controls `cel-miR-39-3p` and `UniSp6` are measured on every study sample; each sample/miRNA pair has two technical replicates; rows are sorted by plate and well, NOT grouped by sample. Twelve miRNAs were assayed on the plate.
* `sample_metadata.csv` - 80 study samples (40 cases, 40 controls): `sample_id`, `group` (`COVID-19` or `Control`), `age`, `sex`, `vaccination_status`, `sars_cov_2_rtqpcr`, `days_since_test`. Only this file carries the group label.
* `mirna_target_pairs.tsv` - miRNA to target gene interactions (~890 rows): `mirna`, `gene`, `confidence` (0-1), `evidence_type`.
* `immune_gene_panel.txt` - 30 curated immune gene symbols, one per line.

## Task

1. **Plate QC / parsing.** Drop no-template controls (`sample_type == NTC`). Treat a measurement as a non-detect when `cq` is the string `Undetermined` or a numeric Ct greater than 38.0. Collapse technical replicates: the value of a (sample, miRNA) pair is the mean of its valid replicates; a pair with no valid replicate is missing and must simply not appear anywhere downstream. A sample's spike-in reference is the mean of ALL its valid spike-in measurements (both spike-in miRNAs, both replicates); drop any sample with fewer than two valid spike-in measurements. Finally drop any miRNA that does not have a valid value in at least 70 percent of the remaining samples (i.e. fewer than 56 of 80 samples).
2. **Normalisation.** For each retained pair: `dct = mean Ct of that miRNA in that sample - spike-in reference of that sample`.
3. **Differential expression.** For every retained miRNA compute `mean_dct_control`, `mean_dct_covid`, `log2fc = mean_dct_control - mean_dct_covid`, and a two-sided Welch t-test evaluated as `scipy.stats.ttest_ind(covid_values, control_values, equal_var=False)`. Correct the p-values with Benjamini-Hochberg across the retained miRNAs (`statsmodels.stats.multitest.multipletests(..., method='fdr_bh')`). A miRNA is significant when `q_value < 0.05` AND `log2fc > log2(1.5) = 0.5849625007`.
4. **Diagnostic performance.** For every retained miRNA compute the ROC AUC from the discriminant score `-dct` with COVID-19 as the positive class (`sklearn.metrics.roc_auc_score`). Then find the Youden operating point: maximise `J = sensitivity + specificity - 1` over the thresholds returned by `sklearn.metrics.roc_curve`; numpy.argmax semantics apply, so on ties take the first maximum, which is the highest threshold (roc_curve returns thresholds in decreasing order). Report sensitivity, specificity and the threshold on the `-dct` score scale.
5. **Signature.** Report the significant miRNAs ordered by descending AUC and the single best miRNA (highest AUC, ties broken alphabetically).
6. **Target network.** Restrict `mirna_target_pairs.tsv` to pairs whose miRNA belongs to the significant signature and whose `confidence >= 0.75`. The degree of a target gene is the number of DISTINCT signature miRNAs that target it. `top_hub_genes` are the 5 genes with the largest degree, ties broken by ascending gene name. `immune_overlap` is the sorted intersection of the network genes with `immune_gene_panel.txt`.

## Constraints

1. **QC and row selection.** `normalized_dct.tsv` must contain exactly one row per retained (sample, miRNA) pair that has a valid measurement: no NTC samples, no spike-in miRNAs, none of the miRNAs failing the 70 percent detection filter, and no rows for missing pairs (do not emit NaN rows).
2. **Normalisation accuracy.** Every `dct` value must match the reference recomputation to within 0.02 Ct.
3. **Group statistics.** Per miRNA, `mean_dct_control` and `mean_dct_covid` must be within 0.05 Ct of the reference and `log2fc` within 0.10 of the reference, using the sign convention defined above (positive log2fc means higher expression in COVID-19).
4. **Welch test.** The magnitude of `t_statistic` must be within 10 percent of the reference and `p_value` within 15 percent relative (or 0.01 absolute) for every retained miRNA.
5. **FDR control.** `q_value` must be the Benjamini-Hochberg adjusted p-value across exactly the retained miRNAs and must be within 0.01 absolute (or 20 percent relative) of the reference.
6. **Signature.** The reported signature set must equal the reference significant set; it must contain miR-30c and miR-126 (the strongest discriminators) and must not contain miR-155 (the negative-control miRNA of this panel).
7. **ROC AUC.** AUC must be within 0.02 of the reference for every retained miRNA, and `best_mirna` must be among the two miRNAs with the highest reference AUC.
8. **Youden operating points.** Sensitivity and specificity at the Youden point must be within 0.05 (absolute) of the reference for every retained miRNA.
9. **Target network.** `n_edges`, `n_target_genes`, the complete `gene_degrees` map, the `top_hub_genes` set and `immune_overlap` must all match the recomputed reference exactly.

## Output

Write everything to `/app/output/` (create it if needed). Numbers need at least 6 significant digits; TSV files use tab separators and the exact headers below.

* `/app/output/normalized_dct.tsv` - columns `sample_id`, `mirna`, `dct`. Example row: `COVID-001<TAB>miR-126<TAB>6.412300`
* `/app/output/de_results.tsv` - columns `mirna`, `mean_dct_control`, `mean_dct_covid`, `log2fc`, `t_statistic`, `p_value`, `q_value`, `significant` (the last one is `yes` or `no`). One row per retained miRNA. Example row: `miR-DEMO<TAB>7.412000<TAB>6.264000<TAB>1.148000<TAB>-5.841200<TAB>1.2e-07<TAB>1.1e-06<TAB>yes`
* `/app/output/roc_results.tsv` - columns `mirna`, `auc`, `youden_cutoff_score`, `sensitivity`, `specificity`. One row per retained miRNA. Example row: `miR-DEMO<TAB>0.771000<TAB>-6.550000<TAB>1.000000<TAB>0.775000`
* `/app/output/signature.json` - `{"signature": ["miR-DEMO-A", "miR-DEMO-B"], "n_signature": 2, "best_mirna": "miR-DEMO-A"}`
* `/app/output/network.json` - `{"n_edges": 123, "n_target_genes": 87, "gene_degrees": {"GENE1": 4, "GENE2": 2}, "top_hub_genes": ["GENE1", "GENE2"], "immune_overlap": ["GENE2"]}`

The example values above are illustrative placeholders only; they are not the answer.

## Scoring

The grader recomputes the entire reference analysis from the raw files and checks constraints 1-9 independently (each worth 1/9 of the score). `score = number of passed constraints / 9`, written together with `reward` into `/logs/verifier/reward.json`. Partial credit is therefore granted for getting some stages of the pipeline right even if others fail. A constraint passes only if all of its sub-checks pass.
