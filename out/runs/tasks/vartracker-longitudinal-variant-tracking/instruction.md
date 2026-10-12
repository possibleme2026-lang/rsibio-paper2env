# Longitudinal variant tracking from VCF and coverage inputs

You are analyzing a longitudinal SARS-CoV-2-like infection series from one patient. The dataset contains a sample manifest, a reference genome, gene annotations, per-sample VCF files, and per-sample per-base coverage tables. Implement the VCF-mode analysis of vartracker: integrate these inputs, filter low-quality and low-coverage variant calls, annotate amino-acid changes, track each variant across timepoints, and classify variants as original/emerging and transient/persistent.

## Inputs

- `/app/data/manifest.tsv`: 8 samples ordered by `timepoint` (0..7) and `days_since_onset` (0,3,6,10,14,21,28,35). Columns: `sample_id`, `patient_id`, `timepoint`, `days_since_onset`, `vcf_path`, `coverage_path`.
- `/app/data/reference.fa`: one 29,903 bp record named `synthetic_ref`.
- `/app/data/genes.gff3`: CDS features for 11 genes (ORF1ab, S, ORF3a, E, M, ORF6, ORF7a, ORF7b, ORF8, N, ORF10). All features are on the + strand; use the `Name` attribute as the gene name.
- `/app/data/vcf/<sample_id>.vcf`: VCF v4.2 files with single-nucleotide variants (SNVs). FILTER is `PASS` or `LowQual`. Each file contains tens of records.
- `/app/data/coverage/<sample_id>.coverage.tsv`: per-base coverage for all 29,903 positions, columns `chrom`, `pos`, `coverage`. This table is authoritative for coverage; ignore DP fields in the VCF if they disagree.

## Task

Write a program that produces three output files in `/app/output/`:

1. `variants.tsv`: one row per unique retained variant across the whole cohort.
2. `presence_matrix.tsv`: a variant x sample 0/1 matrix.
3. `summary.json`: cohort-level counts.

A variant is retained if it has at least one sample where `FILTER == PASS` and coverage at its POS is >= 50. Use the coverage table, not VCF DP, for the coverage value. Apply all constraints below.

## Constraints

1. Filtering: only VCF records with `FILTER == PASS` and coverage >= 50 contribute to a variant's presence. Records with `FILTER != PASS` are counted as low quality; records with `FILTER == PASS` but coverage < 50 are counted as low coverage.
2. Variant identity: `variant_id` is `CHROM:POS:REF:ALT` using 1-based POS. All records are SNVs. Do not emit duplicate `variant_id` rows.
3. Amino-acid annotation: for each variant, find the CDS in `genes.gff3` containing its POS. If none, `gene` and `aa_change` are `NA`. If coding, `gene` is the gene `Name` and `aa_change` is `Gene:RefAA<codon_pos><AltAA>` (e.g., `S:D614G`). Use the standard genetic code and the codon containing POS. Synonymous changes keep the same amino acid (e.g., `N:Y203Y`). All genes are + strand.
4. Temporal tracking: for each retained variant, determine the set of samples where it passes filters. `first_sample`/`first_timepoint` and `last_sample`/`last_timepoint` follow `days_since_onset` order. `n_present` is the number of passing samples. `variant_type` is `original` if the variant is present in the first sample (`timepoint == 0`), otherwise `emerging`. `temporal_class` is `persistent` if `n_present >= 2`, otherwise `transient`.
5. Output schema for `variants.tsv`: exact columns and order: `variant_id`, `pos`, `ref`, `alt`, `gene`, `aa_change`, `first_sample`, `first_timepoint`, `last_sample`, `last_timepoint`, `n_present`, `variant_type`, `temporal_class`, `mean_coverage`, `present_samples`. `present_samples` is comma-separated sample IDs in timepoint order. `mean_coverage` is the mean coverage from the coverage table over present samples, rounded to 1 decimal place.
6. `presence_matrix.tsv`: first column `variant_id`, then one column per sample in manifest timepoint order. Rows must be in the same order as `variants.tsv`. Values are `0` or `1`.
7. `summary.json`: exact integer keys: `total_variants`, `original_variants`, `emerging_variants`, `persistent_variants`, `transient_variants`, `coding_variants`, `noncoding_variants`, `n_vcf_records`, `n_pass_records`, `n_low_quality_records`, `n_low_coverage_records`. `n_vcf_records` is the total number of VCF records across all files. `n_pass_records` is the number of records that pass both filters. `n_low_quality_records` is the number of records with `FILTER != PASS`. `n_low_coverage_records` is the number of records with `FILTER == PASS` but coverage < 50.
8. Ordering: `variants.tsv` sorted by `pos`, then `ref`, then `alt`. `presence_matrix.tsv` rows match `variants.tsv` order.

## Output

`/app/output/variants.tsv` (tab-separated), for example:
```
variant_id	pos	ref	alt	gene	aa_change	first_sample	first_timepoint	last_sample	last_timepoint	n_present	variant_type	temporal_class	mean_coverage	present_samples
synthetic_ref:1234:A:G	1234	A	G	NA	NA	P1_D0	0	P1_D3	3	2	original	persistent	123.4	P1_D0,P1_D3
synthetic_ref:21563:C:T	21563	C	T	S	S:D614D	P1_D3	1	P1_D35	7	7	emerging	persistent	456.7	P1_D3,P1_D6,P1_D10,P1_D14,P1_D21,P1_D28,P1_D35
```

`/app/output/presence_matrix.tsv` (tab-separated):
```
variant_id	P1_D0	P1_D3	P1_D6	P1_D10	P1_D14	P1_D21	P1_D28	P1_D35
synthetic_ref:1234:A:G	1	1	0	0	0	0	0	0
```

`/app/output/summary.json`:
```json
{"total_variants": 2, "original_variants": 1, "emerging_variants": 1, "persistent_variants": 2, "transient_variants": 0, "coding_variants": 1, "noncoding_variants": 1, "n_vcf_records": 100, "n_pass_records": 80, "n_low_quality_records": 10, "n_low_coverage_records": 10}
```

## Scoring

The grader recomputes the expected result from the raw inputs and checks eight constraint groups: file schema, variant set (Jaccard), amino-acid annotation, temporal classification, mean coverage, presence-matrix bits, summary JSON keys, and ordering. Each group contributes equally; score = average of the eight group scores.
