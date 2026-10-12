# Identify SMC foam cell markers and pathways in human atherosclerosis scRNA-seq

You are analyzing a synthetic-but-realistic scRNA-seq dataset from human coronary atherosclerotic plaques. The dataset contains 8 pre-computed cell clusters spanning SMC/fibromyocyte, macrophage, T cell, endothelial, fibroblast, and pericyte populations, including foam cell states. You also have bulk RNA-seq of cultured human SMCs loaded with aggregated LDL (agLDL) or cholesterol bound to methyl-β-cyclodextrin (Chol-MβCD), and a set of Hallmark gene sets. Your goal is to recapitulate the paper's key findings: identify the SMC foam cell cluster, derive specific markers relative to macrophage foam cells, and identify enriched pathways.

## Inputs
- `/app/data/sc_counts.csv`: raw UMI counts, genes x cells (about 800 genes x 640 cells).
- `/app/data/sc_genes.txt`: gene symbols in row order.
- `/app/data/sc_cells.txt`: cell barcodes in column order.
- `/app/data/cell_metadata.csv`: cell_id, cluster (integers 0-7).
- `/app/data/bulk_counts.csv`: raw counts, genes x 9 samples.
- `/app/data/bulk_samples.csv`: sample_id, condition (ctrl, agLDL, chol).
- `/app/data/hallmark_sets.json`: dict of Hallmark pathway names to gene lists.

## Task
1. QC and normalize the scRNA-seq data. Keep cells with >=200 detected genes and mitochondrial fraction <0.20 (mitochondrial genes start with `MT-`). Keep genes detected in >=3 retained cells. Normalize each retained cell to 10,000 total counts and apply log1p.
2. Derive an in-vitro agLDL signature from bulk RNA-seq: compute CPM per sample, then `log2(CPM+1)`. For each gene, Welch t-test between agLDL and control replicates, with `log2FC = mean(log2(CPM+1) in agLDL) - mean(log2(CPM+1) in control)`. BH-adjust p-values. Signature = genes with log2FC > 0.5 and padj < 0.05.
3. Score each cluster by the mean log1p(CP10k) of the signature genes. Identify the SMC foam cell cluster as the cluster with the highest score among clusters with mean MYH11 > 0.5 and mean CD68 < 0.5. Identify the macrophage foam cell cluster as the cluster with the highest mean log1p(CP10k) of APOE among clusters with mean CD68 > 1.0.
4. Differential expression between SMC foam cells and macrophage foam cells: for each gene, compute `log2FC = log2((mean CP10k in SMC foam + 1)/(mean CP10k in macrophage foam + 1))`; Welch t-test on log1p(CP10k) values; BH-adjust. Report the top 20 genes upregulated in SMC foam cells (log2FC > 0, sorted by log2FC descending, then padj ascending, then gene name ascending).
5. Run GSEA with `gseapy.prerank` using all genes ranked by the DE log2FC (descending), the provided Hallmark sets, `min_size=5`, `max_size=500`, `permutation_num=200`, `seed=42`. Report the top 5 pathways with positive NES (enriched in SMC foam cells) sorted by NES descending.
6. Validate that CFH and SERPINE1 are significantly upregulated in SMC foam vs macrophage foam (log2FC > 1.0 and padj < 0.01). Report booleans.

## Constraints
1. `results.json` must contain all required fields with correct types.
2. QC numbers must exactly match the reference QC (cells with >=200 genes and mito fraction <0.20; genes detected in >=3 retained cells).
3. The in-vitro signature must contain CFH and SERPINE1 and have Jaccard index >= 0.8 with the reference signature.
4. `smc_foam_cluster` must equal the reference SMC foam cluster ID.
5. `macrophage_foam_cluster` must equal the reference macrophage foam cluster ID.
6. `top_upregulated` must contain at least 15 of the reference top 20 genes, and CFH and SERPINE1 must be present with log2fc > 1.0 and padj < 0.01.
7. `gsea_top` must contain at least 3 of the reference top 5 pathways, and must include HALLMARK_COMPLEMENT and HALLMARK_COAGULATION in the top 10 with positive NES.
8. `marker_validation` must have both CFH and SERPINE1 as true.
9. Your reusable pipeline `/app/pipeline.py` must produce results on a held-out dataset (different random seed) that satisfy at least 6 of constraints 2-8 on that dataset.

## Output
Write `/app/results.json` with the following schema (worked example):
```json
{
  "qc": {"n_cells_retained": 580, "n_genes_retained": 780},
  "invitro_signature": ["CFH", "SERPINE1", "COL1A1"],
  "smc_foam_cluster": 3,
  "macrophage_foam_cluster": 5,
  "top_upregulated": [
    {"gene": "CFH", "log2fc": 3.2, "padj": 1e-15},
    {"gene": "SERPINE1", "log2fc": 2.8, "padj": 1e-12}
  ],
  "gsea_top": [
    {"pathway": "HALLMARK_COMPLEMENT", "nes": 2.3, "fdr": 0.001},
    {"pathway": "HALLMARK_COAGULATION", "nes": 2.1, "fdr": 0.005}
  ],
  "marker_validation": {"CFH": true, "SERPINE1": true}
}
```

Also write a reusable pipeline `/app/pipeline.py` that accepts `--data_dir` and `--out`, performs all steps, and writes the same JSON schema. The grader will run your pipeline on a held-out dataset with the same schema but a different random seed; your pipeline must not hardcode results from the provided data.

## Scoring
The grader checks each of the 9 constraints independently. Score = (number of constraints passed) / 9. Partial credit is given only through the individual constraint pass/fail conditions (e.g., Jaccard >= 0.8 for signature, >=15/20 overlap for top genes).
