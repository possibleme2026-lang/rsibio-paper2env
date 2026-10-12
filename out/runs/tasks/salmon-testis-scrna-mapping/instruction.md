# Map cultured Atlantic salmon testicular cells to the reference atlas

You are given single-cell RNA-seq count matrices from an Atlantic salmon testis reference atlas and from testicular cells cultured under three conditions for 14 days. Your goal is to computationally map each cultured cell to a reference cell type, quantify condition-specific composition shifts, and characterize Sertoli cell state changes in vitro.

## Inputs
All files are in `/app/data/`:
- `reference_counts.mtx`: sparse genes x cells count matrix (MatrixMarket format) for 2000 reference cells.
- `genes.txt`: 6000 gene names, one per line, corresponding to rows of both count matrices.
- `reference_cells.csv`: 2000 rows with columns `cell_id,cell_type`. Cell types are: Endothelial, Leydig, Macrophage, Peritubular, Sertoli, Spermatids, Spermatocytes, Spermatogonia.
- `cultured_counts.mtx`: sparse genes x cells count matrix for 1200 cultured cells. Rows correspond to the same genes in `genes.txt`.
- `cultured_cells.csv`: 1200 rows with columns `cell_id,condition`. Conditions are: Basal, Differentiation, EGF_IGF.

## Task
Implement the following deterministic pipeline and write the specified outputs to `/app/output/`:

1. Gene filtering: using only the reference counts, keep genes with total counts >= 10 across all reference cells. Record the number of kept genes.
2. Normalization: for each cell (reference and cultured), normalize counts to 10,000 counts per cell (CPM-like), then apply log2(1 + normalized_count).
3. Reference centroids: for each reference cell type, compute the mean log2-normalized expression across all reference cells of that type, using the filtered genes.
4. Cell mapping: for each cultured cell, compute the Pearson correlation between its log2-normalized expression vector and each reference centroid. Assign the cell to the cell type with the highest correlation. Ties are broken alphabetically by cell type.
5. Composition: for each condition, count the number of cultured cells assigned to each reference cell type and compute the proportion of cells in that condition.
6. Enrichment: for each condition, compute the log2 fold change of each cell type's proportion relative to the mean proportion of that cell type in the other two conditions, using a pseudocount of 1e-6. Identify the enriched cell type as the one with the maximum log2 fold change. Ties are broken alphabetically.
7. Sertoli DE: among cultured cells assigned to Sertoli, compare EGF_IGF versus Basal. For each filtered gene, compute the log2 fold change as the difference of mean log2-normalized expression (EGF_IGF minus Basal). Compute Welch's t-test p-values and Benjamini-Hochberg adjusted p-values. Select the top 10 upregulated genes (log2 fold change > 0) sorted by adjusted p-value ascending, then log2 fold change descending, then gene name ascending.
8. QC summary: record the number of filtered genes, number of reference cells, number of cultured cells, and the number of cells assigned to Sertoli in EGF_IGF and Basal.

## Constraints
1. All 1200 cultured cells must appear exactly once in `assignments.csv` with a cell type from the reference atlas.
2. `qc_summary.json` must contain the correct filtered gene count, cell counts, and Sertoli counts.
3. `composition.csv` must report n_cells and proportion for every condition and every one of the 8 reference cell types; proportions must sum to 1 per condition within 1e-6.
4. `enrichment.csv` must identify the enriched cell type for each condition and report the corresponding log2 fold change.
5. `sertoli_de.csv` must list exactly 10 genes (or all if fewer pass log2FC > 0) with columns gene, log2FC, p_value, adj_p_value, sorted as specified.
6. The mapping must use the specified gene filtering, normalization, and Pearson-correlation-to-centroid rule; outputs must match the deterministic expected result within the stated tolerances.

## Output
Write these files to `/app/output/`:
- `assignments.csv`: columns `cell_id,condition,assigned_type`, one row per cultured cell, sorted by cell_id.
- `composition.csv`: columns `condition,cell_type,n_cells,proportion`, sorted by condition (Basal, Differentiation, EGF_IGF) and cell_type alphabetical.
- `enrichment.csv`: columns `condition,enriched_type,log2FC`, sorted by condition (Basal, Differentiation, EGF_IGF).
- `sertoli_de.csv`: columns `gene,log2FC,p_value,adj_p_value`, sorted by adjusted p-value ascending, then log2FC descending, then gene name ascending.
- `qc_summary.json`: JSON object with keys `n_genes_filtered`, `n_reference_cells`, `n_cultured_cells`, `n_sertoli_EGF_IGF`, `n_sertoli_Basal`.

Example row for `assignments.csv`:
`cell_c0001,Basal,Sertoli`

## Scoring
The grader recomputes the expected outputs from the input data using the exact pipeline above. Partial credit is given across six independently checked constraints: (1) QC summary, (2) assignment file completeness/validity, (3) assignment accuracy versus expected mapping, (4) composition table correctness, (5) enrichment table correctness, (6) Sertoli DE top-10 correctness. The final score is the mean of the constraint scores, and `reward` equals `score`.
