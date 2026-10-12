# Lower respiratory tract microbiome: does sampling method drive inferred community structure?

You are handed 16S rRNA gene amplicon (ASV) data from a pilot study of the human lower respiratory tract. Thirty-three participants (8 lung cancer, 25 non-cancer controls) were each sampled three ways - oral rinse, bronchoalveolar lavage (BAL) and bronchial brushing - and six negative controls (extraction and library blanks) were sequenced alongside them. Your job is to run the comparative analysis that decides how much of the inferred community structure is explained by the sampling method itself, which ASVs are indicative of each method, and whether cancer status is detectable at this sample size. Low-biomass respiratory samples are notoriously contaminated, so the negative controls must be used to decontaminate the table before any ecological statistic is computed.

## Inputs

All inputs are tab-separated files in `/app/data/`:

- `asv_table.tsv` - raw ASV counts, 400 ASVs (rows, index column `asv_id`) x 105 samples (columns): 99 participant samples named `P01_oral`, `P01_bal`, `P01_brush`, ..., `P33_brush`, plus 6 negative controls `NC01`-`NC06`.
- `taxonomy.tsv` - 7-level taxonomy (`Kingdom`...`Genus`) for the same 400 ASVs, index column `asv_id`.
- `metadata.tsv` - columns `sample_id`, `participant_id`, `sample_type` (`oral`, `bal`, `brush`, `control`), `cancer_status` (`cancer`, `non_cancer`, `NA` for controls), `batch`.

## Task

1. Decontaminate and filter the ASV table using the negative controls.
2. Characterise beta diversity of the retained participant samples (relative abundance, Bray-Curtis, PCoA, PERMANOVA on `sample_type`) and quantify how dominant sampling method is as an axis of variation.
3. Test the three sample types pairwise (adjusted p-values) to establish which methods differ.
4. Run an indicator-species analysis (IndVal) per sample type and determine which ASVs are shared between indicator sets and which are method-specific, including the identity of the brush-specific indicator belonging to family Sphingomonadaceae.
5. Test whether cancer status is detectable, both as a community-level effect (PERMANOVA) and as per-ASV differential abundance with FDR control.

## Constraints

1. **Decontamination and abundance filtering.** Prevalence is computed on the raw counts, before any filtering. Presence means count >= 1. An ASV is a *contaminant* if it is present in at least 50% of the 6 negative-control samples AND present in fewer than 50% of the 99 participant samples. Retain only ASVs that are not contaminants and whose total count across the 99 participant samples is >= 10.
2. **Filtered table format.** `/app/output/filtered_asv_table.tsv` must be tab-separated with a header row containing exactly the 99 participant sample IDs, in the same order in which they appear as columns of `asv_table.tsv`. Rows are the retained ASVs sorted by ASV id ascending; values are integer counts identical to the input table (no re-normalisation, no rounding).
3. **Beta diversity protocol.** Convert retained counts to relative abundances (each sample divided by its retained-ASV total). Compute Bray-Curtis dissimilarity with `skbio.diversity.beta_diversity('braycurtis', ...)` and ordinate with `skbio.stats.ordination.pcoa`; report the proportion of variance explained by axis 1. Run PERMANOVA (`skbio.stats.distance.permanova`) on `sample_type` with 999 permutations, with `numpy.random.seed(42)` set immediately before the call. Effect size is R2 = F(a-1) / (F(a-1) + (N-a)) where F is the pseudo-F, a the number of groups and N the number of samples.
4. **Pairwise PERMANOVA.** Repeat the PERMANOVA on the three two-type subsets of the same Bray-Curtis matrix (oral vs bal, oral vs brush, bal vs brush) and adjust the three p-values with Benjamini-Hochberg. Bronchial brushing must be significantly different from both other methods after adjustment (adjusted p < 0.05) while oral rinse vs BAL must not be (adjusted p >= 0.05).
5. **Indicator-species analysis (IndVal).** For each retained ASV and each sample type compute IndVal = 100 x A x B, where A is the mean relative abundance of the ASV in samples of that type divided by the sum of the three per-type mean relative abundances, and B is the fraction of samples of that type in which the ASV is present (count >= 1). The p-value comes from 999 permutations of the sample-type labels: p = fraction of permutations in which the maximum IndVal of that ASV over the three types is >= its observed IndVal for the type. An ASV is an indicator of a type when IndVal >= 40 and p < 0.05. Report the three indicator lists sorted by ASV id, the pairwise overlap counts, and the ASV id of the brush indicator whose taxonomic family is Sphingomonadaceae (exactly one ASV in the table belongs to that family and it must be a brush indicator).
6. **Cancer-status testing.** Restrict to the 33 BAL samples and compare cancer (n=8) versus non-cancer (n=25) for every retained ASV with a two-sided Mann-Whitney U test on relative abundance using `scipy.stats.mannwhitneyu(..., method='asymptotic')`, then adjust across all retained ASVs with Benjamini-Hochberg. Also run the same PERMANOVA (999 permutations) of the 99 participant samples on `cancer_status`. Report how many ASVs have adjusted p < 0.05 and the cancer-status PERMANOVA p-value. No ASV should be significant after adjustment.
7. **Report schema and rounding.** `/app/output/report.json` must be valid JSON containing exactly the keys shown below, with all floating-point values rounded to 4 decimal places and every indicator list sorted by ASV id ascending.

## Output

`/app/output/filtered_asv_table.tsv` (tab-separated, index column `asv_id`):

```
asv_id	P01_oral	P01_bal	P01_brush	P02_oral	...
ASV_0001	271	318	2	204	...
ASV_0002	0	12	0	9	...
```

`/app/output/report.json`:

```json
{
  "n_asvs_retained": 312,
  "n_contaminants_removed": 19,
  "beta_diversity": {
    "pcoa_pc1_variance_explained": 0.4123,
    "permanova_sample_type": {"pseudo_f": 18.4211, "r2": 0.4512, "p_value": 0.001},
    "pairwise_permanova": {
      "oral_vs_bal": {"r2": 0.0512, "p_value": 0.214, "p_adjusted": 0.214},
      "oral_vs_brush": {"r2": 0.3011, "p_value": 0.001, "p_adjusted": 0.003},
      "bal_vs_brush": {"r2": 0.2874, "p_value": 0.001, "p_adjusted": 0.003}
    }
  },
  "indicator_analysis": {
    "oral": ["ASV_0003", "ASV_0004"],
    "bal": ["ASV_0003", "ASV_0061"],
    "brush": ["ASV_0138", "ASV_0140"],
    "overlaps": {"oral_bal": 47, "oral_brush": 0, "bal_brush": 0},
    "brush_specific_sphingomonadaceae_asv": "ASV_0138"
  },
  "cancer_status": {"permanova_p_value": 0.312, "n_asvs_fdr_significant": 0}
}
```

## Scoring

`tests/grade.py` re-derives the reference answer from `/app/data` and checks seven independent constraints: (1) the filtered table (contaminant rule, abundance rule, row order, identical counts), (2) the PCoA axis-1 variance fraction, (3) the `sample_type` PERMANOVA R2 and p-value, (4) the three adjusted pairwise p-values, (5) the indicator structure (Sphingomonadaceae ASV id, brush indicator set size, oral/BAL overlap non-empty, brush overlaps empty, reported overlaps consistent with the reported lists), (6) the cancer-status result, and (7) the JSON schema and rounding. Each constraint is worth 1/7 of the score and the final score is the fraction of constraints passed (partial credit). The score and the same value as `reward` are written to `/logs/verifier/reward.json`.
