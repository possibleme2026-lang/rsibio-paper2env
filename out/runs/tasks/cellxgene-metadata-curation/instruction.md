# Curate a single-cell submission into a schema-validated community release

## Background
Community single-cell resources such as CZ CELLxGENE Discover grow by having data contributors hand over *raw, heterogeneous* submissions while dedicated curators convert them into standardized, ontology-annotated, schema-validated releases. A contributor has just deposited a downsampled slice of a study (free-text metadata, versioned Ensembl gene list, sparse count matrix). You are the curation engineer: your job is to turn that deposit into the release-ready metadata tables that the resource would publish, applying the resource's curation rules exactly as written below.

This is a real curation job, not a format conversion: free-text values must be resolved against controlled vocabularies, cells that cannot be curated or are too low quality must be dropped, donor-level attributes must be reconciled across the cells of a donor, and gene-level quality flags must be recomputed from the count matrix.

## Inputs (all under `/app/data`)
- `submission/cell_metadata.tsv` - ~1,300 rows (one row per cell, duplicates included) with the columns
  `cell_id`, `donor_id`, `tissue_free_text`, `cell_type_free_text`, `assay_free_text`, `organism_free_text`,
  `donor_sex_free_text`, `development_stage_free_text`, `donor_health_status_free_text`, `donor_disease_free_text`.
  Values are free text: mixed case, padded with whitespace, sometimes an ontology ID, sometimes a label, sometimes an ontology synonym, sometimes `NA` / `not collected`.
- `submission/genes.tsv` - 532 rows, columns `gene_id`, `gene_name`. `gene_id` is an Ensembl ID, roughly a third of the rows carry a `.<version>` suffix, and several genes appear twice.
- `submission/counts.mtx` - MatrixMarket coordinate integer matrix. Row *i* is cell *i* of `cell_metadata.tsv`; column *j* is row *j* of `genes.tsv` (both 1-based). ~1,300 x 532 with a few hundred thousand non-zero entries.
- `ontology/` - controlled-vocabulary tables `cl.tsv`, `uberon.tsv`, `efo.tsv`, `ncbitaxon.tsv`, `mondo.tsv`, `pato.tsv`, `hsapdv.tsv`, `mmusdv.tsv`. Format: header `id\tlabel\tsynonyms`, synonyms separated by `|`.
- `gen_inputs.py` - the deterministic generator that produced the slice (readable for transparency; it contains no answer key).

## Task
Write a curation program and run it. It must:
1. parse the submission tables and the sparse count matrix;
2. resolve every free-text value against the correct vocabulary table;
3. apply the retention rules of `R3` in the stated priority order;
4. reconcile donor-level attributes by majority voting over the donor's retained cells;
5. apply the disease (`R5`) and developmental-stage (`R6`) rules;
6. recompute gene-level flags and totals from the matrix restricted to retained cells;
7. write `/app/output/obs.tsv`, `/app/output/var.tsv` and `/app/output/report.json` with exactly the columns/keys described in `R7`-`R9`.

## Rules
**R1 - Normalization (matching only).** To compare a free-text value with the vocabulary, strip leading/trailing whitespace, collapse internal runs of whitespace to a single space, and lowercase. Never change the output labels: the label written to `obs.tsv` is always the canonical `label` field of the matched vocabulary term.

**R2 - Vocabulary resolution.** Each field has exactly one vocabulary: `cell_type_free_text` -> `cl.tsv`; `tissue_free_text` -> `uberon.tsv`; `assay_free_text` -> `efo.tsv`; `organism_free_text` -> `ncbitaxon.tsv`; `donor_disease_free_text` -> `mondo.tsv`; `development_stage_free_text` -> `hsapdv.tsv` for human donors and `mmusdv.tsv` for mouse donors; healthy donors use `pato.tsv`. A value matches a term if its normalized form equals the normalized `id`, the normalized `label`, or one of the normalized synonyms; a value that matches nothing is *unresolvable*.

**R3 - Row retention (apply in this exact order; a row is counted under the first rule that removes it).**
1. `cell_id` already seen earlier in file order -> **duplicate**, drop (keep the first occurrence).
2. `cell_type_free_text`, `tissue_free_text`, `assay_free_text` or `organism_free_text` unresolvable -> **unresolved ontology**, drop.
3. The cell's row of `counts.mtx` has fewer than **100** non-zero entries -> **low complexity**, drop.
All other rows are retained.

**R4 - Donor-level reconciliation.** `organism`, `sex`, `development_stage`, `health status` and `disease` are properties of the *donor*, not of the cell. For every retained cell compute its resolved value, then for each donor take the **most frequent** value across that donor's retained cells (ties broken by the lexicographically smallest value) and apply it to every cell of the donor.
- `sex`: normalized text in {`female`, `f`, `woman`} -> `female`; {`male`, `m`, `man`} -> `male`; {`mixed`} -> `mixed`; anything else -> `unknown`.
- `health status`: {`healthy`, `healthy donor`, `healthy control`, `normal`, `control`, `none`, `unaffected`} -> `healthy`; {`diseased`, `disease`, `sick`, `patient`, `case`, `affected`} -> `diseased`; anything else -> `unknown`.
- `organism`: resolved `NCBITaxon` term ID (an unresolvable value counts as the literal `unknown`).
- `development_stage`: resolve the cell's text with the table selected by the donor's **prevailing organism** (human -> `hsapdv`, mouse -> `mmusdv`, otherwise the value is `unknown`). Order matters: determine the donor's organism first, then resolve developmental stages, then vote.
- `disease`: resolved `MONDO` term ID of `donor_disease_free_text`, or `unknown` if unresolvable. Voting on disease is only meaningful for diseased donors (see R5).

**R5 - Disease rule.** If the donor's prevailing health status is `healthy` (or `unknown`), the disease annotation is `PATO:0000461` with label `normal`, regardless of what `donor_disease_free_text` says. If the donor is `diseased`, the disease annotation is the donor's prevailing `MONDO` ID and its label; if that prevailing value is `unknown`, write `unknown` / `unknown`.

**R6 - Developmental-stage namespace.** Human donors (prevailing `NCBITaxon:9606`) must carry a `HsapDv:*` ID, mouse donors (`NCBITaxon:10090`) an `MmusDv:*` ID; any other organism, or a donor whose prevailing stage could not be resolved, gets `unknown` / `unknown`.

**R7 - `/app/output/obs.tsv`.** TSV with a header line, exactly these columns in exactly this order:
`cell_id`, `donor_id`, `tissue_ontology_term_id`, `tissue`, `cell_type_ontology_term_id`, `cell_type`, `assay_ontology_term_id`, `assay`, `disease_ontology_term_id`, `disease`, `organism_ontology_term_id`, `organism`, `sex`, `development_stage_ontology_term_id`, `development_stage`, `n_genes_detected`.
One row per retained cell, sorted ascending by `cell_id` (plain ASCII sort), `cell_id` values unique. `n_genes_detected` is the number of non-zero entries in that cell's row of `counts.mtx` (raw, before any gene deduplication).

**R8 - `/app/output/var.tsv`.** TSV with header `gene_id`, `gene_name`, `feature_is_filtered`. `gene_id` has the version suffix removed (everything from the first `.` onwards). Keep the **first occurrence** in file order of each deduplicated `gene_id` and its `gene_name`; sort ascending by `gene_id`. `feature_is_filtered` is `TRUE` when the gene's total counts over the retained cells is 0 (summing **all** matrix columns that map to that gene, i.e. duplicate rows included) and `FALSE` otherwise, uppercase.

**R9 - `/app/output/report.json`.** A JSON object with exactly these integer keys:
`n_input_rows` (data rows of `cell_metadata.tsv`, duplicates included), `n_duplicate_rows_removed`, `n_dropped_unresolved_ontology`, `n_dropped_low_complexity`, `n_retained_cells`, `n_retained_donors`, `n_genes_input_rows` (rows of `genes.tsv`), `n_genes_output_rows` (unique version-stripped gene IDs), `n_genes_filtered` (genes with `feature_is_filtered == TRUE`).

## Output
- `/app/output/obs.tsv`, `/app/output/var.tsv`, `/app/output/report.json`.
- Worked example rows (values are illustrative):

```
cell_id	donor_id	tissue_ontology_term_id	tissue	cell_type_ontology_term_id	cell_type	assay_ontology_term_id	assay	disease_ontology_term_id	disease	organism_ontology_term_id	organism	sex	development_stage_ontology_term_id	development_stage	n_genes_detected
C000042	DONOR-003	UBERON:0002048	lung	CL:0000235	macrophage	EFO:0009922	10x 3' v3	PATO:0000461	normal	NCBITaxon:9606	Homo sapiens	female	HsapDv:0000087	human adult stage	213
```

```
gene_id	gene_name	feature_is_filtered
ENSG00000123456	MALAT1	FALSE
ENSG00000987654	LOC100000137	TRUE
```

## Scoring
Ten independently checked constraints, partial credit each, weighted: output schema and sort/type discipline 0.08; exact retained-cell set (Jaccard) 0.14; ontology-ID validity and namespace correctness 0.12; label/ID agreement 0.12; sex + organism correctness 0.10; disease rule correctness 0.10; developmental-stage correctness 0.10; donor set + within-donor consistency 0.08; `var.tsv` structure and flags 0.08; `report.json` counts 0.08. The score is the weighted sum (0.0-1.0) and is written to `/logs/verifier/reward.json`.
