# Integrative analysis of DOX-induced renal tubular injury and sirtuin rescue

You are a computational biologist analyzing multi-omics and functional data from a study on chronic doxorubicin (DOX) nephrotoxicity. The goal is to reproduce the key integrative analysis: identify mitochondrial and inflammatory signatures, quantify mitochondrial respiratory function from Seahorse data, assess mitochondrial morphology from SIM imaging, and compute a composite rescue score for resveratrol (RSV) and NMN treatments.

## Inputs
Files are in `/app/data/`:
- `rna_counts.csv`: RNA-seq count matrix (genes x 24 samples). Samples: 6 per group: Control, DOX, DOX+RSV, DOX+NMN. 500 genes.
- `proteomics.csv`: Protein abundance matrix (300 proteins x 24 samples).
- `seahorse_ocr.csv`: Seahorse XF OCR measurements. Columns: well, group, time (min), OCR (pmol/min). 24 wells (6 per group), time points every 3 min from 0 to 90.
- `sim_morphology.csv`: Single-cell mitochondrial morphology from SIM. Columns: cell_id, group, aspect_ratio, form_factor, area, perimeter, mitochondrial_count. 200 cells (50 per group).
- `sod2_acetyl.csv`: SOD2 acetylation measurements. Columns: sample, group, total_SOD2, acetylated_SOD2.
- `gene_sets.json`: Gene sets for pathways: OXPHOS (50 genes), FAO (40 genes), Inflammation (50 genes).

## Task
Perform the following analysis steps and write the results to `/app/output/results.json`.

1. **RNA-seq QC and normalization**: Compute CPM (counts per million) for each sample: CPM = (count / total counts per sample) * 1e6. Filter genes: keep genes with CPM >= 1 in at least 3 samples. Record the number of genes after filtering. Work with log2(CPM + 1) for downstream analysis.
2. **Proteomics QC**: Filter proteins with non-zero intensity in at least 3 samples. Record the number of proteins after filtering.
3. **Differential expression (DOX vs Control)**: For each gene, compute log2 fold change (log2FC) as the difference of mean log2(CPM+1) between DOX and Control groups. Perform Welch's t-test (unequal variance) for each gene between DOX and Control groups. Adjust p-values using Benjamini-Hochberg FDR. Consider a gene significantly changed if FDR < 0.05 and |log2FC| > 1.0. Count significant genes in the OXPHOS set with log2FC < 0 (downregulated) and in the Inflammation set with log2FC > 0 (upregulated). List the top 10 downregulated genes (most negative log2FC) and top 10 upregulated genes (most positive log2FC) among all significant genes. If fewer than 10, list all.
4. **Pathway scores**: For each gene, compute a z-score across all samples (mean 0, std 1) using log2(CPM+1) values. For each pathway (OXPHOS, FAO), compute a pathway score for each sample as the mean z-score of genes in that pathway. Compute the mean pathway score per group (Control, DOX, DOX+RSV, DOX+NMN).
5. **Seahorse analysis**: The Seahorse protocol: oligomycin injected at 20 min, FCCP at 40 min, rotenone/antimycin A at 60 min. Use the following time windows: basal = 0-18 min, oligomycin = 21-38 min, FCCP = 42-58 min, non-mitochondrial = 63-90 min. For each well, compute: `non_mito` = mean OCR in non-mitochondrial window. `basal` = mean OCR in basal window - non_mito. `maximal` = mean OCR in FCCP window - non_mito. `spare` = maximal - basal. `ATP` = basal - (mean OCR in oligomycin window - non_mito). Average each metric per group.
6. **SIM morphology**: Classify a cell as fragmented if aspect_ratio < 2.0 AND form_factor < 0.5. Compute the percentage of fragmented cells per group.
7. **SOD2 acetylation**: Compute the ratio acetylated_SOD2 / total_SOD2 for each sample. Compute the mean ratio per group.
8. **Rescue scores**: For each treatment (DOX+RSV, DOX+NMN), compute a composite rescue score as the sum of normalized improvements relative to DOX, normalized by the Control-to-DOX difference. Metrics (higher is better unless noted): OXPHOS score, FAO score, Basal respiration, Maximal respiration, Spare respiratory capacity, ATP production, Percent non-fragmented cells (100 - percent_fragmented), SOD2 acetylation ratio (lower is better). For higher-is-better metrics: improvement = (treatment - DOX) / (Control - DOX). For lower-is-better metrics: improvement = (DOX - treatment) / (DOX - Control). Sum the eight improvements to get the rescue score.

## Constraints
1. RNA-seq and proteomics QC counts must exactly match the true values.
2. Differential expression counts (num_down_OXPHOS, num_up_inflammation) must exactly match.
3. Top 10 down/up gene lists must exactly match (order and content).
4. Pathway scores for OXPHOS and FAO must be within 0.05 absolute error for every group.
5. Seahorse metrics (basal, maximal, spare, ATP, non_mito) must be within 5.0 pmol/min for every group.
6. SIM percent fragmented must be within 3.0 percentage points for every group.
7. SOD2 acetylation ratio must be within 0.05 absolute error for every group.
8. Rescue scores must be within 0.1 absolute error for each treatment.

## Output
Write `/app/output/results.json` with exactly this structure:
```json
{
  "qc": {
    "rna_genes_after_filter": <int>,
    "proteins_after_filter": <int>
  },
  "differential_expression": {
    "num_down_OXPHOS": <int>,
    "num_up_inflammation": <int>,
    "top10_down": [<gene names>],
    "top10_up": [<gene names>]
  },
  "pathway_scores": {
    "OXPHOS": {"Control": <float>, "DOX": <float>, "DOX+RSV": <float>, "DOX+NMN": <float>},
    "FAO": {"Control": <float>, "DOX": <float>, "DOX+RSV": <float>, "DOX+NMN": <float>}
  },
  "seahorse": {
    "basal": {"Control": <float>, "DOX": <float>, "DOX+RSV": <float>, "DOX+NMN": <float>},
    "maximal": {"Control": <float>, "DOX": <float>, "DOX+RSV": <float>, "DOX+NMN": <float>},
    "spare": {"Control": <float>, "DOX": <float>, "DOX+RSV": <float>, "DOX+NMN": <float>},
    "ATP": {"Control": <float>, "DOX": <float>, "DOX+RSV": <float>, "DOX+NMN": <float>},
    "non_mito": {"Control": <float>, "DOX": <float>, "DOX+RSV": <float>, "DOX+NMN": <float>}
  },
  "sim_morphology": {
    "percent_fragmented": {"Control": <float>, "DOX": <float>, "DOX+RSV": <float>, "DOX+NMN": <float>}
  },
  "sod2_acetylation": {
    "ratio": {"Control": <float>, "DOX": <float>, "DOX+RSV": <float>, "DOX+NMN": <float>}
  },
  "rescue_scores": {
    "DOX+RSV": <float>,
    "DOX+NMN": <float>
  }
}
```
Round all floats to 4 decimal places.

## Scoring
The grader will check the 8 constraints above. Each constraint contributes equally to the final score (1/8 each). The score is written to `/logs/verifier/reward.json` as `{"score": <float>, "reward": <float>}`.
