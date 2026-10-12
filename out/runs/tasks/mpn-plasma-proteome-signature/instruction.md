# Plasma-proteome diagnostic signature for Philadelphia-negative MPN

## Context
Patients referred to a haematology clinic because of elevated blood counts are a diagnostic challenge: some carry a Philadelphia-negative myeloproliferative neoplasm (MPN; ET, PV or PMF), others do not. You have plasma proximity-extension assay (PEA / Olink-style) NPX data (log2 scale, arbitrary units) for 96 such subjects together with routine blood-count covariates. 64 subjects have a confirmed clinical diagnosis (training cohort); the diagnosis of the remaining 32 is withheld and must be predicted from the proteome.

Reproduce, on this cohort, the analysis of a plasma-proteome MPN screen: (1) QC and imputation of the NPX matrix, (2) differential abundance between MPN and non-MPN, (3) an L1-penalised (Lasso) logistic-regression diagnostic panel with cross-validated model selection and prospective prediction of the withheld cohort, and (4) the PV subtype-specific signal, in particular TFRC.

## Inputs
* `/app/data/proteome_npx.csv` - 96 rows, columns `sample_id` + 44 protein columns. Values are NPX (log2 scale); roughly 3% of the cells are empty (assay failure / below LOD). Three different assay plates were used.
* `/app/data/sample_metadata.csv` - `sample_id,cohort,plate,age,sex,platelet_count,hemoglobin,wbc_count,ldh,jak2_v617f`. The `cohort` column is `train` (64 rows) or `test` (32 rows).
* `/app/data/clinical_train.csv` - `sample_id,diagnosis,mpn_subtype` for the 64 training subjects. `diagnosis` is `MPN` or `non-MPN`; `mpn_subtype` is `ET`, `PV`, `PMF` or `non-MPN`.

The diagnosis of the 32 `test` subjects is not present anywhere in the container.

## Task
1. Load the NPX matrix and the metadata. Drop proteins with more than 20% missing values (keep track of how many proteins remain) and impute the remaining missing cells (for example per-protein median). Consider correcting the per-plate offsets before testing.
2. In the training cohort only, test every retained protein for differential abundance between MPN and non-MPN. Report `log2_fc = mean(NPX | MPN) - mean(NPX | non-MPN)` and Benjamini-Hochberg adjusted q-values.
3. Fit an L1-penalised logistic regression (Lasso) to the retained proteins of the training cohort; select the penalty by cross-validation with at least 5 folds using ROC-AUC. Report the non-zero-coefficient panel (3-12 proteins) and predict the probability of MPN for every withheld test subject.
4. Within the MPN training subjects, contrast PV against non-PV (ET + PMF): log2_fc and BH q-values for every retained protein, plus the diagnostic AUC of TFRC alone for PV vs non-PV in the training cohort.
5. Write the five output files described below.

## Constraints
1. `predictions.csv` must contain exactly the 32 `test` subjects, in the order in which they appear in `sample_metadata.csv`, each with a probability in [0,1].
2. Predictions must be non-degenerate: at least 5 subjects above 0.5, at least 5 below 0.5, and at least 8 distinct probability values.
3. The test-cohort ROC-AUC of your probabilities against the withheld diagnoses must be at least 0.75.
4. Balanced accuracy of the 0.5 hard call must be at least 0.70.
5. `differential_expression.csv` must report at least 40 proteins with p-values and BH q-values in [0,1] (q >= p in every row) and at least 12 proteins at q < 0.05.
6. At least 55% of the proteins that are truly differential in this cohort must be recovered at q < 0.05, and you may not declare more than 8 non-differential proteins significant.
7. For the truly differential proteins you recover, at least 80% must carry the correct sign of `log2_fc` (up in MPN means a positive value).
8. Negative control: IL6, a pro-inflammatory cytokine reported to be unchanged between MPN and non-MPN, must come out with q >= 0.05 and |log2_fc| <= 0.30.
9. `pv_markers.csv` must list TFRC with q < 0.05 and a positive log2_fc, and the `tfrc_pv_auc` in `report.json` must agree with the TFRC PV-vs-non-PV AUC computed from the training NPX values (tolerance 0.10).
10. `panel.json` must list 3-12 unique measured proteins, and at least 4 of them must be genuinely informative for the MPN-vs-non-MPN contrast rather than fit artefacts.
11. `report.json` must be internally consistent with your own tables: `n_de_significant` equals the number of rows of `differential_expression.csv` with q < 0.05, `il6_q_mpn_vs_nonmpn` equals the IL6 q-value in that table, and `n_train_samples` / `n_test_samples` are 64 and 32.

## Output
All files are written into `/app/output/`.

* `predictions.csv` - header `sample_id,mpn_probability`; example row `P004,0.873`
* `differential_expression.csv` - header `protein,log2_fc,p_value,q_value`; example row `HP,0.04,4.2e-01,5.1e-01`
* `pv_markers.csv` - header `protein,log2_fc,q_value`; one row per analysed protein
* `panel.json` - e.g. `{"selected_proteins": ["<protein A>", "<protein B>", "<protein C>"], "cv_auc": 0.91, "n_cv_folds": 5, "model": "lasso_logistic"}`
* `report.json` - e.g. `{"n_de_significant": 18, "tfrc_pv_auc": 0.91, "il6_q_mpn_vs_nonmpn": 0.42, "il6_log2_fc": 0.05, "n_train_samples": 64, "n_test_samples": 32, "n_proteins_analyzed": 44}`

## Scoring
The grader verifies the 11 constraints above independently and writes `score = (number of constraints passed) / 11` to `/logs/verifier/reward.json`. There is no partial credit inside a constraint; every constraint counts the same. The per-constraint breakdown is printed to stdout and stored in `/logs/verifier/report.json`.
