# Facility-level heterogeneity confounds arsenic effects on glucose homeostasis

You are a computational biologist analyzing a multi-omics mouse study of arsenic exposure across two animal facilities. The study measured gut microbiome species abundances, metabolites, and gene family frequencies, along with metabolic phenotypes (glucose AUC). The central scientific question is whether the housing facility or the arsenic treatment explains more variance in the microbiome and metabolome, and whether a facility-specific microbial signature can be learned.

## Inputs
The public data are in `/app/data/`:
- `metadata.csv` — 80 rows: sample_id, facility (A/B), treatment (control/iAs), batch, glucose_auc, weight_g.
- `microbiome_species.csv` — 80 rows × 200 species count columns (`sp_000` … `sp_199`). Rows are samples.
- `metabolites.csv` — 80 rows × 300 metabolite intensity columns (`met_000` … `met_299`).
- `gene_families.csv` — 80 rows × 2000 gene family count columns (`gene_0000` … `gene_1999`).
- `README.txt`.

A held-out dataset with the same schema will be provided at test time. Your code must not hardcode sample IDs, species names, or expected numeric results.

## Task
Write a reusable analysis script at `/app/analysis.py` that accepts `--data-dir <dir>` and `--out-dir <dir>`. When run, it must:
1. Load and align the metadata, microbiome, metabolite, and gene-family tables.
2. For the microbiome table, filter low-prevalence species, normalize to relative abundances, and compute Bray–Curtis distances.
3. Use PERMANOVA (or an equivalent distance-based variance partitioning) to quantify the fraction of variance explained (R²) by facility and by treatment on the microbiome.
4. Train random-forest classifiers (5-fold cross-validation) to predict facility and treatment from microbiome species. Report the mean cross-validated accuracy for each.
5. Identify a minimal facility-discriminative species panel by ranking species from the facility classifier. The panel should contain ~22 species (aim for 20–25).
6. Perform differential abundance analysis for species, gene families, and metabolites separately for facility and treatment. Use a non-parametric test (e.g., Mann–Whitney U) with Benjamini–Hochberg FDR correction. Report the number of features with q < 0.05 for each contrast.

## Constraints
1. `results.json` must contain all required numeric keys (see Output) with JSON number values.
2. Facility cross-validated accuracy must be ≥ 0.85.
3. Treatment cross-validated accuracy must be ≤ 0.70.
4. Facility PERMANOVA R² on the microbiome must be ≥ 0.15.
5. Treatment PERMANOVA R² on the microbiome must be < 0.05.
6. The facility signature must contain between 15 and 30 species, all present in the input species table, and at least 50% of them must overlap with the true facility-associated species (hidden ground truth).
7. `differential_species.csv` must contain one row per species with columns `species_id,log2FC_facility,pvalue_facility,qvalue_facility,log2FC_treatment,pvalue_treatment,qvalue_treatment`. The reported `n_differential_species_facility` must equal the number of rows with `qvalue_facility < 0.05` and be ≥ 20. The reported `n_differential_species_treatment` must equal the number of rows with `qvalue_treatment < 0.05` and be ≤ 20.
8. `n_differential_genes_facility` must be ≥ 800 and `n_differential_genes_treatment` ≤ 150.
9. `n_differential_metabolites_facility` must be ≥ 50 and `n_differential_metabolites_treatment` ≤ 30.
10. `variance_partition.csv` must contain rows for `facility` and `treatment` with columns `term,R2,pvalue` and R² values in [0,1].

## Output
Write all outputs to the directory given by `--out-dir` (default `/app/output` if run manually). Required files:
- `results.json`:
  ```json
  {
    "facility_cv_accuracy": 0.96,
    "treatment_cv_accuracy": 0.62,
    "facility_variance_explained": 0.24,
    "treatment_variance_explained": 0.02,
    "n_signature_species": 22,
    "n_differential_species_facility": 31,
    "n_differential_species_treatment": 8,
    "n_differential_genes_facility": 1200,
    "n_differential_genes_treatment": 50,
    "n_differential_metabolites_facility": 110,
    "n_differential_metabolites_treatment": 15
  }
  ```
- `signature_species.txt`: one species ID per line, exactly `n_signature_species` lines.
- `variance_partition.csv`: `term,R2,pvalue` with rows for facility and treatment.
- `differential_species.csv`: as described in Constraint 7.

## Scoring
The grader runs your `/app/analysis.py` on a held-out dataset with a different random seed. It then checks each constraint independently and assigns partial credit: `score = (number of constraints passed) / (total constraints)`. The reward is this score. The grader also verifies internal consistency (e.g., reported counts match the CSV files). Hardcoding public-data answers will fail because the held-out dataset has different sample IDs and effect realizations.
