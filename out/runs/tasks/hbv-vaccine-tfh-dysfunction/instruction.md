# Reconstruct Tfh dysfunction analysis from HBV vaccine lymph node scRNA-seq

You are given single-cell RNA-seq and immune repertoire data from 10 lymph node biopsies (6 High, 4 Low HBV vaccine responders). Reproduce the key analysis to identify the GNG4+ Tfh subset, its association with antibody titers, and BCR repertoire features.

## Inputs
- `/app/data/scrna/counts.mtx`: raw UMI counts (genes x cells), Matrix Market format.
- `/app/data/scrna/genes.tsv`: gene names in row order.
- `/app/data/scrna/barcodes.tsv`: cell barcodes in column order.
- `/app/data/scrna/cell_metadata.tsv`: columns: cell_id, sample_id, donor_id, cell_type (Tfh, GC_B, T_cell, B_cell, other).
- `/app/data/clinical/participants.tsv`: columns: donor_id, vaccination_type, end_of_study_titer_mIU_mL.
- `/app/data/genesets/*.txt`: one gene symbol per line for: tfh_markers, gng4_tfh_signature, inflammatory, bcl6_regulon, dz_markers, lz_markers, myc_targets, mtorc1_targets.
- `/app/data/bcr/airr_repertoire.tsv`: BCR contigs with columns: cell_id, contig_id, v_call, d_call, j_call, cdr3_nt, cdr3_aa, somatic_hypermutation_rate, isotype, clonotype_id.
- `/app/data/tcr/airr_repertoire.tsv`: TCR contigs (similar format; not required for scoring).

## Task
Write a reproducible analysis script (e.g., `/app/solution.py`) that performs:
1. **Load and normalize** the scRNA-seq count matrix. Normalize each cell to 10,000 total counts, then apply log1p. Use all genes for DE and the provided gene sets for signature scores.
2. **Define responder groups**: High if `end_of_study_titer_mIU_mL >= 100`, else Low.
3. **Annotate GNG4+ Tfh cells**: Among cells with `cell_type == "Tfh"`, for each donor, compute the 60th percentile of GNG4 log1p expression; cells above this percentile are GNG4+ Tfh.
4. **Per-donor scores**: For each donor, compute:
   - `gng4_tfh_signature_score`: mean z-scored (across all Tfh cells) expression of the GNG4+ Tfh signature genes, averaged over GNG4+ Tfh cells.
   - `inflammatory_score`: same for inflammatory genes.
   - `bcl6_regulon_activity`: same for BCL6 regulon genes.
   - `bcl6_regulon_specificity`: Pearson correlation across GNG4+ Tfh cells between BCL6 expression and the mean z-scored expression of BCL6 regulon genes (excluding BCL6 itself).
   - `dz_score`, `lz_score`, `myc_score`, `mtorc1_score`: for GC_B cells, mean z-scored expression of respective gene sets.
5. **BCR repertoire metrics**:
   - `mean_shm`: mean `somatic_hypermutation_rate` across all BCR contigs for the donor.
   - `clonotype_diversity`: Shannon entropy of clonotype abundances (count of contigs per `clonotype_id`) for the donor.
   - `public_clonotypes`: number of distinct `clonotype_id` observed in >=2 donors that are also present in this donor.
6. **Differential expression**: Within GNG4+ Tfh cells, compare High vs Low responders. For each gene, compute log2 fold change of mean normalized expression (add pseudocount 1 to means), and a Wilcoxon rank-sum test p-value. Apply Benjamini-Hochberg correction. Output all genes tested (at least 250 genes), sorted by p-value ascending. Mark direction as `up_in_high` if log2FC > 0 else `up_in_low`.
7. **Correlations**:
   - Pearson correlation between `gng4_tfh_signature_score` and `end_of_study_titer_mIU_mL` across donors.
   - Pearson correlation between `mean_shm` and `end_of_study_titer_mIU_mL`.
   - Pearson correlation between `inflammatory_score` and `end_of_study_titer_mIU_mL`.

## Output
Write two files:
- `/app/results/analysis.json` with exactly:
```json
{
  "donors": {
    "<donor_id>": {
      "responder_group": "High" | "Low",
      "end_of_study_titer_mIU_mL": <number>,
      "n_tfh": <int>,
      "n_gng4_tfh": <int>,
      "gng4_tfh_signature_score": <number>,
      "inflammatory_score": <number>,
      "bcl6_regulon_activity": <number>,
      "bcl6_regulon_specificity": <number>,
      "dz_score": <number>,
      "lz_score": <number>,
      "myc_score": <number>,
      "mtorc1_score": <number>,
      "mean_shm": <number>,
      "clonotype_diversity": <number>,
      "public_clonotypes": <int>
    },
    ...
  },
  "correlations": {
    "gng4_score_vs_titer": <number>,
    "shm_vs_titer": <number>,
    "inflammatory_vs_titer": <number>
  }
}
```
- `/app/results/de_genes.csv` with columns: gene, mean_high, mean_low, log2_fold_change, p_value, padj, direction. Include all genes tested (at least 250 genes). Sorted by p_value ascending.

## Constraints
1. `analysis.json` must be valid JSON with all 10 donors and all required fields.
2. `gng4_tfh_signature_score` must be computed exactly as specified (z-score normalization across all Tfh cells, then mean over GNG4+ Tfh cells). Grader compares per-donor values to reference with tolerance.
3. `bcl6_regulon_specificity` must be a Pearson correlation between BCL6 and mean regulon targets across GNG4+ Tfh cells.
4. `de_genes.csv` must contain all genes tested (>= 250) and top 20 up in High must include the true up genes with F1 >= 0.7 against hidden gold.
5. `correlations.gng4_score_vs_titer` must be > 0.5 and match reference within ±0.1.
6. `mean_shm` must be higher in High responders than Low responders (p < 0.05 by Mann-Whitney).
7. `public_clonotypes` count per donor must be within ±1 of reference.
8. `inflammatory_score` must be negatively correlated with titer (r < -0.4).
9. `dz_score` and `lz_score` must be higher in High responders (p < 0.05).
10. `myc_score` and `mtorc1_score` must be higher in High responders (p < 0.05).

## Scoring
Score is the fraction of constraints passed (10 constraints, partial credit). The grader recomputes the reference from a protected seed and compares your output. Each constraint is verified independently.
