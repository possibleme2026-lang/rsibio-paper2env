# Hippo-pathway inhibition in mouse kidney: transcriptional concordance, dose dependence and aldosterone-signalling loss

You are a computational toxicologist on the preclinical team that characterised a covalent pan-TEAD inhibitor (TEADi, GNE-8025) and a genetically inducible YAP/TAZ knockdown. Bulk RNA-seq of whole mouse kidney was generated for five conditions (6 biological replicates each): vehicle, low-dose TEADi, high-dose TEADi, high-dose TEADi followed by a 14-day washout, and tamoxifen-induced YAP/TAZ knockdown. Reproduce the paper's transcriptomic analysis: quantify the response shared between pharmacological and genetic Hippo-pathway inhibition, test whether the kidney response is dose dependent and partially irreversible after washout, and determine whether the aldosterone/mineralocorticoid programme is specifically suppressed by pan-TEAD inhibition.

## Inputs
- `/app/data/kidney_counts.csv` - raw integer RNA-seq counts; rows are 2,000 mouse genes (first column `gene`), columns are the 30 sample IDs (matrix orientation: genes x samples).
- `/app/data/sample_metadata.csv` - columns `sample_id,condition,perturbation,dose_mg_kg,washout_days,replicate`. Conditions: `vehicle`, `TEADi_low` (3 mg/kg), `TEADi_high` (30 mg/kg), `TEADi_recovery` (30 mg/kg + 14 d washout), `YAPTAZ_KD` (genetic, tamoxifen-induced).
- `/app/data/gene_sets.gmt` - six tab-delimited gene sets (`name`, `description`, then gene symbols): three curated sets (`ALDOSTERONE_SIGNALING`, `YAP_TEAD_TARGETS`, `KIDNEY_TUBULAR_INJURY`) and three decoy sets.
- Installed: `pydeseq2` (0.4.x), `numpy`, `pandas`, `scipy`, `scikit-learn`. No network access. Treat `/app/data` as read-only.

## Task
1. **QC filter.** Keep genes with at least 10 counts in at least 6 samples; report how many genes and samples survive.
2. **Differential expression with PyDESeq2.** Fit one DESeq2 model on the filtered matrix with design `~condition` (counts passed as samples x genes, `metadata` indexed by `sample_id`). For each of the four contrasts `TEADi_low`, `TEADi_high`, `TEADi_recovery`, `YAPTAZ_KD` (each against `vehicle`) call `DeseqStats` with `contrast=['condition', <level>, 'vehicle']` and `independent_filter=False`; do not apply LFC shrinkage. Call a gene DE when `padj <= 0.05` and `abs(log2FoldChange) >= 1.0` (rows with NaN padj are never DE).
3. **Concordance between pharmacological and genetic perturbation.** A gene is *concordant* when it is DE in both the high-dose TEADi contrast and the YAPTAZ_KD contrast with the same sign of log2FoldChange. Report the count and the full alphabetically sorted gene list.
4. **Aldosterone signature.** For every `ALDOSTERONE_SIGNALING` gene present in the filtered matrix report its log2FoldChange in the high-dose contrast; the signature score is their arithmetic mean. Test the score with a gene-set permutation: create `rng = numpy.random.default_rng(42)`, then perform 1,000 draws of `rng.choice(universe, size=k, replace=False)` where `universe` is the vector of log2FoldChange values of ALL retained genes and `k` is the number of retained panel genes; the empirical p-value is `(1 + #{|mean_null| >= |mean_obs|}) / 1001`.
5. **Dose dependence.** Over the retained `ALDOSTERONE_SIGNALING` genes, report the fraction for which the high-dose log2FoldChange is more negative than the low-dose log2FoldChange AND is negative.
6. **Recovery / irreversibility.** Report the fraction of high-dose DE genes that are still DE with the same sign of log2FoldChange in the washout (`TEADi_recovery`) contrast.
7. **Gene-set ranking.** For each of the six sets in the GMT file compute the mean high-dose log2FoldChange over its retained genes and rank the sets from most positive to most negative.
8. **Outputs.** Write the two files described in the Output section.

## Constraints
1. The QC filter is exactly `sum(counts >= 10) >= 6` per gene, applied before model fitting; `n_genes_retained` must match that rule exactly.
2. `n_samples` must equal 30, and the whole matrix must be used: no extra filtering, no gene collapsing, no per-gene normalisation other than what PyDESeq2 performs internally.
3. All statistics must come from PyDESeq2 with `design='~condition'`, `independent_filter=False`, no LFC shrinkage, and the four contrasts listed above; DE requires `padj <= 0.05` AND `abs(log2FoldChange) >= 1.0`.
4. `concordant_high_vs_KD['genes']` must be alphabetically sorted, contain no duplicates, and `n` must equal the length of that list.
5. `aldosterone_panel_log2fc` must contain an entry for every retained `ALDOSTERONE_SIGNALING` gene; values are high-dose log2FoldChange floats.
6. The permutation test must use `numpy.random.default_rng(42)`, exactly 1,000 samples, each of size k (number of retained panel genes) drawn without replacement from the log2FoldChange values of all retained genes.
7. `set_mean_log2fc_ranking` must contain all six GMT set names, ordered from highest to lowest mean log2FoldChange over retained members.
8. `/app/output/de_TEADi_high_vs_vehicle.tsv` must contain one row per retained gene with the exact tab-separated header `gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj`, sorted by padj ascending with NaN-padj rows last and ties broken by gene name; `top20_high_vs_vehicle` must be the first 20 gene names of that ordering restricted to genes with non-NaN padj.
9. `/app/data` must not be modified, and both output files must be produced deterministically (running the same code twice yields identical files).

## Output
Two artefacts, both under `/app/output`:

1. `results.json` - a JSON object with exactly these keys:
```
{
  'n_samples': <int>,
  'n_genes_retained': <int>,
  'de_counts': {'TEADi_low': <int>, 'TEADi_high': <int>, 'TEADi_recovery': <int>, 'YAPTAZ_KD': <int>},
  'concordant_high_vs_KD': {'n': <int>, 'genes': [<alphabetically sorted gene symbols>]},
  'aldosterone_panel_log2fc': {'<gene>': <float>, ...},
  'aldosterone_signature_score': <float>,
  'aldosterone_permutation_p': <float>,
  'dose_monotonic_fraction': <float>,
  'recovery_persistent_fraction': <float>,
  'set_mean_log2fc_ranking': [<six set names, descending>],
  'top20_high_vs_vehicle': [<twenty gene symbols, most significant first>]
}
```

2. `de_TEADi_high_vs_vehicle.tsv` - tab-separated, exact header order, one row per retained gene. Worked example row (dummy values, not from this dataset):
```
gene	baseMean	log2FoldChange	lfcSE	stat	pvalue	padj
Gm1234	812.4	-1.87	0.21	-8.90	1.4e-19	2.7e-17
```

## Scoring
The grader regenerates the input matrix from a protected seed, independently re-runs the same PyDESeq2 pipeline, and evaluates ten independent constraints: (1) sample/gene accounting, (2) the four DE counts (each contrast scored separately, tolerance max(2 genes, 2%)), (3) concordant-set F1 against the recomputed set, (4) aldosterone panel log2FoldChange coverage plus per-gene accuracy (tolerance 0.10), (5) signature score and permutation p-value, (6) dose-monotonic fraction, (7) recovery-persistent fraction, (8) gene-set ranking agreement, (9) top-20 ordering agreement, (10) output format (keys, header, sort order, row count). Each constraint contributes a score in [0, 1]; the final score and reward are the arithmetic mean of the ten constraint scores, written to `/logs/verifier/reward.json`. Partial credit is therefore available for a partially correct pipeline.
