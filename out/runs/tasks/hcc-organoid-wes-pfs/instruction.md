# HCC organoid cohort: genomic fidelity and growth-phenotype outcome analysis

You are the computational biologist on a translational hepatocellular carcinoma (HCC) study. Resected tumours from 27 patients were used to derive patient-derived organoids (PDOs); organoids were established for part of the cohort and classified morphologically as **cystic** or **solid**. For six patients, whole-exome sequencing (WES) was performed on matched tumour and organoid material.

Your job is to rebuild the study's analysis from the raw tables: describe the cohort, quantify tumour-to-organoid genomic fidelity, and test whether the organoid growth phenotype is associated with progression-free survival (PFS).

All data are synthetic but structurally faithful to the study. **The numbers reported in the published paper are not reproduced by this cohort** - you must compute everything from the files.

## Inputs

- `/app/data/clinical.tsv` - 27 tab-separated rows, header:
  `patient_id, age, sex, etiology, tumor_size_cm, differentiation, bcat_positive, afp_ng_ml, organoid_established, organoid_morphology, pfs_months, pfs_event`.
  `etiology` is one of `HBV`, `HCV`, `alcohol`, `NBNC`; `organoid_morphology` is `cystic`, `solid` or `NA` (NA means the organoid was not established); `pfs_event` is 1 for progression/death and 0 for censoring; `bcat_positive` is 1/0 for beta-catenin immunohistochemistry.
- `/app/data/common_snps.tsv` - 600 common germline SNPs with columns `chrom, pos, ref, alt, rsid`. Any somatic call at one of these loci must be discarded.
- `/app/data/variants/<PATIENT>_tumor.tsv` and `/app/data/variants/<PATIENT>_organoid.tsv` - WES somatic call tables for six matched pairs (`HCC-01`, `HCC-04`, `HCC-08`, `HCC-12`, `HCC-17`, `HCC-23`). Each file holds 280-480 rows and the columns `chrom, pos, ref, alt, gene, consequence, depth, alt_depth, vaf, qual`.
- `/app/data/gen_inputs.py` - the deterministic generator (seed 20260802) that produced all of the above.

Do not modify, move or regenerate anything under `/app/data`. The evaluation rebuilds the dataset from the same seed and any difference invalidates the submission.

## Task

1. **Cohort description.** Count the patients and the established organoids, compute the establishment rate in percent, and count the cystic and solid organoids.
2. **Establishment associations.** For each of three candidate factors - age > 70 years, HBV etiology, beta-catenin positive - build the 2x2 table of establishment versus the factor and run a **two-sided Fisher exact test**.
3. **Somatic variant filtering.** For each of the 12 sample tables keep only rows with `depth >= 20` **and** `vaf >= 0.10` **and** whose `(chrom, pos, ref, alt)` key does not occur anywhere in `common_snps.tsv`.
4. **Tumour-organoid concordance.** Variants are matched between the tumour and the organoid of a pair on the key `(chrom, pos, ref, alt)`. For each pair report the number of shared variants, the number private to the tumour, the number private to the organoid, the Jaccard index `shared / (tumour + organoid - shared)` computed on the filtered sets, the Spearman rank correlation of the tumour and organoid `vaf` values over the shared variants, and the median absolute difference of those VAFs.
5. **Genomic fidelity call.** A pair passes if `jaccard >= 0.60` **and** `vaf_spearman >= 0.70`.
6. **Growth phenotype and outcome.** Restrict the survival analysis to the patients with an established organoid. Fit a **univariate Cox proportional-hazards model** for PFS with `solid` versus `cystic` as the only covariate (cystic is the reference) and report the hazard ratio; run the **log-rank test** between the two morphology groups; report the Kaplan-Meier median PFS of each group.
7. Write the three output files described below.

## Constraints

1. Filtering thresholds are exactly `depth >= 20` and `vaf >= 0.10` (inclusive); use the rounded `vaf` printed in the file.
2. Germline removal is by exact match of `(chrom, pos, ref, alt)` against `common_snps.tsv` - do not filter by rsid or by VAF.
3. Variant identity between tumour and organoid is `(chrom, pos, ref, alt)`; `gene` and `consequence` are not part of the key.
4. The Jaccard index is computed on the **filtered** sets, and the VAF statistics only on the shared variants.
5. `fidelity_pass` is true only when **both** thresholds are met, i.e. `jaccard >= 0.60` and `vaf_spearman >= 0.70`.
6. The survival analysis uses only `organoid_established == 1` patients, `pfs_months` as the time and `pfs_event` as the event indicator; the Cox covariate is `solid` (1) versus `cystic` (0).
7. `establishment_rate_pct` is reported to one decimal place as `100 * established / total`.
8. Numeric columns are rounded to 4 decimal places in the TSV outputs.
9. All output files must be created under `/app/output/`.

## Output

**`/app/output/summary.json`** - a single JSON object with these keys:

- `n_patients` (integer), `n_organoid_established` (integer), `establishment_rate_pct` (number)
- `morphology_counts` (object with `cystic` and `solid` integers, established patients only)
- `filtered_variant_counts` (object mapping `<PATIENT>_tumor` / `<PATIENT>_organoid` to the integer number of retained variants, 12 keys)
- `n_fidelity_pass` (integer)
- `cox_hr_solid_vs_cystic` (number), `logrank_p` (number)
- `median_pfs_months` (object with `cystic` and `solid` numbers)
- `establishment_p_values` (object with `age_gt70`, `hbv` and `bcat_positive` numbers)

**`/app/output/pair_concordance.tsv`** - tab separated, one row per matched pair, sorted by `patient_id` ascending, header `patient_id\tshared\ttumor_only\torganoid_only\tjaccard\tvaf_spearman\tmedian_abs_vaf_diff\tfidelity_pass`, lowercase `true`/`false` in the last column. Example row (illustrative only, patient does not exist):

`HCC-99\t250\t18\t15\t0.7560\t0.9410\t0.0231\ttrue`

**`/app/output/shared_variants.tsv`** - tab separated, one row per shared variant of every pair, sorted by `patient_id`, then by chromosome in natural order (1..22, X), then by `pos` ascending, header `patient_id\tchrom\tpos\tref\talt\tgene\ttumor_vaf\torganoid_vaf`. Example row (illustrative only):

`HCC-99\t7\t148463512\tC\tT\tTP53\t0.4120\t0.3987`

## Scoring

The grader rebuilds the dataset from the protected seed, recomputes the reference analysis, and checks each constraint independently. The final score is the weighted fraction of constraints that pass, so partial credit is available.

- cohort summary (`n_patients`, `n_organoid_established`, `establishment_rate_pct` within 0.15) - weight 1
- morphology counts - weight 1
- filtered variant counts, all 12 samples - weight 2
- pair intersections (shared / tumor_only / organoid_only exact, jaccard within 0.005) - weight 2
- VAF concordance (spearman within 0.02, median absolute difference within 0.005) - weight 1.5
- fidelity classification and `n_fidelity_pass` - weight 1
- Cox hazard ratio > 1 and within 20% of the reference value - weight 1.5
- log-rank p < 0.05 and within 25% of the reference value - weight 1.5
- Kaplan-Meier medians within 0.3 months - weight 1
- Fisher exact p-values within 15% relative - weight 1
- shared variant catalogue (header, row count, loci set, sort order) - weight 1.5
