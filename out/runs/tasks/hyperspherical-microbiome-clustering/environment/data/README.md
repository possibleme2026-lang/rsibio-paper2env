Data dictionary (all rows are subjects S001..S240, identical order).

microbiome_abundance.csv  - 240 x 55 relative abundances in (0,1), rows sum to 1 (taxon_01..taxon_55)
host_metabolomics.csv     - 240 x 45 log-scale metabolite intensities (metab_01..metab_45)
host_transcriptomics.csv  - 240 x 35 host gene expression values (tx_01..tx_35)
sample_metadata.csv       - sample_id, cohort, sex, age, bmi (nuisance covariates)

The cohort was simulated deterministically with a fixed seed. Ground truth (community labels, transitional-sample flags, discriminative features and taxon-host association pairs) is held by the evaluator only.
