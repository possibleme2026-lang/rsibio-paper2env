# Reproduce the PG-LLM sequence-only baseline for protein variant ranking

## Context

PG-LLM (2026) benchmarks general-purpose language models against 95 published protein predictors on 276 deep-mutational-scanning (DMS) variant-prioritisation tasks. Every task has the same shape: given a wild-type protein sequence and a list of variants, rank the variants by predicted fitness effect, then score the ranking with the Spearman correlation between predicted and measured DMS fitness. Tool-free baselines must be run on exactly the same variants and scored with exactly the same metric.

You are given a miniature replica of that benchmark: six DMS assays, each with a wild-type sequence, the multiple sequence alignment (MSA) of its homologues, and the assay's variant table together with the measured DMS fitness scores. Your job is to implement the benchmark's sequence-only baseline predictor, run it on all six assays, and reproduce the benchmark report.

## Inputs

All paths are relative to `/app/data`.

- `assay_manifest.csv` — one row per assay: `assay_id,uniprot_id,protein_name,sequence_length,n_variants,split`. `split` is either `proteinGym` (4 assays) or `heldout` (2 assays, the temporal hold-out set of the paper). `n_variants` is the number of data rows in that assay's `variants.csv`.
- `assays/<assay_id>/wt.fasta` — a single FASTA record: the wild-type sequence (129-330 aa).
- `assays/<assay_id>/msa.fasta` — 64 aligned homologue sequences, every one exactly the same length as the wild-type. The **first record is the wild-type sequence itself**; the remaining 63 records are homologues (`>HOMOLOG_001` ... `>HOMOLOG_063`). Alignment characters are the 20 standard uppercase amino acids plus `-` (gap) and `X` (unknown).
- `assays/<assay_id>/variants.csv` — one row per assayed variant: `variant_id,mutations,dms_score`.
  - `variant_id` is unique within an assay and is currently identical to the mutation string.
  - `mutations` is either the literal string `WT` (the wild-type control row) or one or more point substitutions separated by `:` — for example `A123T` or `A123T:G45D`.
  - A substitution is written `<wild-type residue><1-based position><mutant residue>`, so position numbering is the wild-type sequence numbering.
  - `dms_score` is the experimentally measured fitness of the variant (z-scored within the assay; the wild-type control is `0.0`). **These are measurements, not predictions — they must never be copied into the prediction column.**

## The baseline you must implement (name it `MSA-CONS-BLOSUM`)

For each assay, with wild-type sequence `W` of length `L` and its MSA:

1. **Column frequencies.** For alignment column `p` (0-based; column `p` corresponds to wild-type position `p+1`), let `c_p(a)` be the number of MSA rows whose character at that column is the standard amino acid `a`, and let `C_p = sum_a c_p(a)`. Ignore `-` and `X` completely: they are counted neither in `c_p` nor in `C_p`. The pseudocount-smoothed frequency is

   `q_p(a) = (c_p(a) + 1) / (C_p + 20)`

2. **Per-substitution score.** For a substitution of the wild-type residue `w = W[p]` by residue `a` at position `p+1`:

   `S(p, a) = log2( q_p(a) / q_p(w) ) + 0.10 * ( B(w, a) - B(w, w) )`

   where `B` is the standard NCBI **BLOSUM62** matrix (`Bio.Align.substitution_matrices.load("BLOSUM62")`) and `log2` is the base-2 logarithm.

3. **Variant score.** A variant's predicted score is the **sum** of `S(p, a)` over all of its substitutions. The wild-type control (`mutations == "WT"`) has score exactly `0.0`.

Interpretation: a positive score means the variant is predicted to be fitter than wild type.

## Task

1. Implement the baseline above and compute a predicted score for **every** row of **every** `variants.csv`, including the `WT` control rows.
2. Compute, for each assay, the Spearman correlation between the predicted scores and the measured `dms_score` column over **all rows of that assay's `variants.csv` (including the WT control)**. Use average ranks for ties (`scipy.stats.spearmanr`); if the correlation is undefined because one of the vectors is constant, report `0.0`.
3. Rank each assay's variants by **descending predicted score**, breaking ties by **ascending `variant_id`** (lexicographic), and take the first five as the assay's prioritised variants.
4. Aggregate the per-assay correlations into split means: the arithmetic mean of the per-assay rho over the `proteinGym` assays and over the `heldout` assays.
5. Write the two output files described below.

## Constraints

1. `predictions.csv` must have the exact header shown below, must contain every `(assay_id, variant_id)` pair present in the inputs exactly once, must contain no other rows, and every `predicted_score` must be a finite number.
2. Every predicted score must match the reference implementation of the rule in this brief to within `1e-6` (absolute), compared over all rows of all six assays.
3. The wild-type control row of every assay (`mutations == "WT"`) must have `predicted_score` exactly `0.0`.
4. The `top5` list of every assay must match the reference ranking (descending score, ties broken by ascending `variant_id`) exactly, including order.
5. Every `per_assay[<assay_id>]["spearman_rho"]` must match the Spearman correlation recomputed from **your own** `predictions.csv` and the provided `dms_score` column to within `1e-4`.
6. `split_mean_rho` and `n_assays` must be consistent with the per-assay values and the assay splits (arithmetic means within `1e-4`; correct assay counts).

## Output

**`/app/output/predictions.csv`** — exactly this header, one row per `(assay_id, variant_id)`, no duplicates:

```
assay_id,variant_id,predicted_score
TP53_HUMAN_Giacomelli_2018,A5G,0.183204
TP53_HUMAN_Giacomelli_2018,WT,0.000000
```

Write scores with at least 6 significant digits (plain decimal or scientific notation).

**`/app/output/report.json`** — exactly this structure:

```json
{
  "predictor": "MSA-CONS-BLOSUM",
  "per_assay": {
    "TP53_HUMAN_Giacomelli_2018": {
      "n_variants": 171,
      "spearman_rho": 0.4123,
      "top5": ["A5G", "R17C", "L35V", "T41A", "E52D"]
    }
  },
  "split_mean_rho": {"proteinGym": 0.4011, "heldout": 0.3887},
  "n_assays": {"proteinGym": 4, "heldout": 2}
}
```

`per_assay` must contain an entry for all six assays; `n_variants` is the number of rows in that assay's `variants.csv`.

## Scoring

The grader regenerates a pristine copy of the inputs from the protected seed, recomputes the reference scores with an independent implementation of the rule, and evaluates six independent constraints. Partial credit is granted per constraint:

| # | constraint | partial credit |
|---|---|---|
| 1 | output format, coverage, uniqueness | fraction of expected rows present, scaled by 0.5 if the header is wrong and by 0.5 if duplicates exist |
| 2 | score fidelity within `1e-6` | fraction of rows matching |
| 3 | wild-type controls exactly `0.0` | fraction of control rows correct |
| 4 | top-5 prioritisation | fraction of assays whose top-5 list matches exactly |
| 5 | per-assay Spearman | fraction of assays whose reported rho matches the recomputed one |
| 6 | split means, assay counts, per-assay `n_variants` | mean of the four sub-checks |

`score = reward = mean of the six constraint scores`, written to `/logs/verifier/reward.json`.
