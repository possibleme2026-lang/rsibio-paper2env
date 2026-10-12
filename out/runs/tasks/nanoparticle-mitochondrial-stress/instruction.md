# Comparative Analysis of Nanoparticle-Induced Mitochondrial Stress and Immunotoxicity

You are a computational biologist analyzing gene expression data from human peripheral blood mononuclear cells (PBMCs) exposed to three environmental nanoparticles: ultrafine particulate matter (UFPM), black carbon (BC), and polystyrene nanoplastics (PS-NPs). The goal is to reproduce the systems-level analysis from a recent study by identifying conserved and particle-specific mitochondrial stress responses, inflammation, and candidate biomarkers.

## Inputs
- `/app/data/expression_matrix.csv`: 108 samples x 30 genes. Rows are samples (sample_id, then gene columns). Values are log2-normalized expression.
- `/app/data/sample_metadata.csv`: sample_id, donor (D01-D12), particle (UFPM, BC, PS_NP), time_h (6, 24, 72).
- `/app/data/gen_inputs.py`: deterministic generator script (seed 20260729) used to create the above files. You may inspect it to understand data structure, but the analysis must be performed on the provided CSV files.

## Task
Perform the following multi-step analysis and write the results to `/app/output/results.json`.

1. **Preprocessing**: Load expression matrix and metadata. Align samples. Standardize each gene (column) across all 108 samples to mean 0 and population standard deviation 1 (ddof=0).
2. **PCA**: Run PCA on the standardized matrix with `n_components=5`, `svd_solver='full'`, `random_state=0`. Extract explained variance ratios for PC1-PC3. Identify the top 5 genes by absolute loading for PC1 and PC2. Ties are broken alphabetically.
3. **Hierarchical clustering**: Compute pairwise correlation distance between samples as `1 - Pearson_r` on the standardized matrix. Use average linkage (`scipy.cluster.hierarchy.linkage`, method='average'). Cut the dendrogram into 3 clusters using `fcluster` with `criterion='maxclust'`, `t=3`. Re-label clusters 0, 1, 2 in ascending order of mean PC1 score of the samples in each cluster. Output a mapping from sample_id to cluster label.
4. **Correlation network**: Compute Pearson correlation between all pairs of genes across all 108 samples. Define an edge if `|r| >= 0.7`. Compute the degree of each gene (number of edges, excluding self). Identify the top 5 hub genes by degree (descending), breaking ties alphabetically.
5. **Conserved network**: For each particle subset separately (UFPM, BC, PS_NP), compute gene-gene Pearson correlations. An edge is **conserved** if `|r| >= 0.7` in all three particle subsets. Count the number of conserved edges. Identify the top 5 conserved edges by mean absolute correlation across the three subsets (descending), breaking ties alphabetically by gene pair. Represent an edge as `"GENE1-GENE2"` with genes sorted alphabetically.
6. **OMA1-DELE1 correlation**: Compute the Pearson correlation between OMA1 and DELE1 across all 108 samples.
7. **Particle response classification**: Using the standardized matrix, compute module scores for each sample as the mean of the standardized expression of genes in each module:
   - Oxidative stress: NFE2L2, HMOX1, NQO1, GCLM, TXN, SOD1, CAT, GPX1
   - Mitochondrial adaptation: PPARGC1A, TFAM, NRF1, POLG, TWNK
   - Mitochondrial ISR: OMA1, DELE1, ATF4, DDIT3, CHAC1
   - Inflammation: NFKB1, TNF, IL6, NLRP3, IL1B, CXCL8
   - Complex integrity: MT-ND1, NDUFA1, SDHA, UQCRC1, COX5A, ATP5F1A
   For each particle, average module scores across donors for each timepoint. Classify each particle using these rules:
   - **kinetics**: `"acute"` if peak oxidative score is at 6h and the 72h score is < 0.5 * the 6h score; `"chronic"` if peak is at 72h and the 72h score >= 0.8 * max(6h, 24h); otherwise `"mixed"`.
   - **mito_state**: `"adaptation"` if (adaptation_72h - adaptation_6h) >= 0.3 and ISR_72h < 0.0 and complex_72h > 0.0; `"dysfunction"` if ISR_72h > 0.5 or complex_72h < -0.5; `"biogenesis"` if (adaptation_72h - adaptation_6h) >= 0.5 and ISR_72h > 0.0; otherwise `"other"`.
   - **isr_axis**: `"active"` if mean ISR score across all timepoints > 0.0, else `"weak"`.
   - **inflammation**: `"high"` if inflammation score at 72h > 0.0, else `"low"`.
8. **Biomarkers**: Identify genes that are both in the top 5 hub genes and appear in at least one conserved edge. Sort alphabetically.

## Constraints
1. Output file must exist at `/app/output/results.json` and be valid JSON with all required keys.
2. PCA explained variance ratios for PC1-PC3 must match gold within ±0.002.
3. PC1 top-5 gene set must match gold exactly (order-insensitive).
4. PC2 top-5 gene set must match gold exactly (order-insensitive).
5. Hierarchical clustering assignments must match gold exactly for all 108 samples.
6. Top 5 hub genes must match gold exactly (order-sensitive, sorted by degree descending then alphabetically).
7. OMA1-DELE1 correlation must match gold within ±0.01.
8. Conserved network edge count must match gold exactly.
9. Top 5 conserved edges must match gold set exactly (order-insensitive).
10. Particle response labels must match gold exactly for all 12 labels.
11. Biomarker list must match gold set exactly (order-insensitive).

## Output
Write a single JSON object to `/app/output/results.json` with the following schema:
```json
{
  "pca_explained_variance": [0.1234, 0.0987, 0.0765],
  "pc1_top_genes": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
  "pc2_top_genes": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
  "hierarchical_clusters": {"D01_UFPM_6h": 0, "D01_UFPM_24h": 1, ...},
  "hub_genes": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
  "oma1_dele1_correlation": 0.8123,
  "conserved_network_edge_count": 42,
  "conserved_network_top_edges": ["OMA1-DELE1", "GENE1-GENE2", ...],
  "particle_response": {
    "UFPM": {"kinetics": "acute", "mito_state": "adaptation", "isr_axis": "weak", "inflammation": "low"},
    "BC": {"kinetics": "chronic", "mito_state": "dysfunction", "isr_axis": "active", "inflammation": "high"},
    "PS_NP": {"kinetics": "chronic", "mito_state": "biogenesis", "isr_axis": "active", "inflammation": "low"}
  },
  "biomarkers": ["OMA1", "DELE1"]
}
```
(Floats must be rounded to 4 decimal places. Cluster labels must be integers 0, 1, or 2.)

## Scoring
The grader recomputes all gold values from a protected deterministic seed. Each of the 11 constraints above is worth 1 point. The final score is the fraction of passed constraints. Partial credit is awarded per constraint. The grader writes `{"score": <fraction>, "reward": <fraction>}` to `/logs/verifier/reward.json`.
