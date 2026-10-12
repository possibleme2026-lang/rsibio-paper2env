# Tumour epigenetic age shifts and phyloepigenetic reconstruction across 8 solid cancers

You are re-analysing a multi-cancer DNA methylation cohort. `/app/data` holds Illumina-style beta values for 960 samples (480 patients, each with a matched tumour and adjacent-normal sample) spanning 8 solid tumour types, together with the coefficients of a published multi-tissue epigenetic clock ("MT-clock", 353 CpG probes). Your job is to reproduce the two analyses of the study and write machine-readable results.

## Inputs

All inputs live in `/app/data`:

| file | contents |
|---|---|
| `beta_matrix.tsv` | 960 rows (samples) x 551 tab-separated columns: `sample_id` followed by 550 `cg...` probe columns. Beta values in [0,1] with 3 decimals; `NA` marks a missing value. |
| `probe_annotation.csv` | `probe_id,chromosome,position,gene_symbol` for all 550 probes (`chr1`-`chr22`, `chrX`, `chrY`). |
| `sample_metadata.csv` | `sample_id,patient_id,cancer_type,tissue,chronological_age_years,sex,array_batch`. `tissue` is `tumor` or `normal`; every `patient_id` has exactly one of each. |
| `clock_coefficients.csv` | `probe_id,coefficient,transform` for the 353 clock probes; `transform` is `beta` or `one_minus_beta`. |
| `clock_model.json` | the clock intercept and the exact age transformation. |

## Task

### Part 1 - epigenetic age

For every sample `s`, compute the linear predictor

    x_s = I + SUM over the 353 clock probes p of  c_p * f_p(beta_{s,p})

where `I` is the intercept in `clock_model.json`, `c_p` is the probe coefficient, and
`f_p(b)` equals `b` when `transform == "beta"` and `1 - b` when `transform == "one_minus_beta"`.
Convert the predictor to a DNA methylation age with the piecewise transform

    DNAmAge = 21*exp(x) - 1      if x >= 0
    DNAmAge = 20 + 20*x          if x <  0

Then:

1. Report the mean absolute error between DNAmAge and chronological age over the **normal** samples only.
2. For every cancer type, form the per-patient paired difference `DNAmAge(tumour) - DNAmAge(normal)` and report the number of pairs, the median difference in years, the two-sided Wilcoxon signed-rank p-value, the Benjamini-Hochberg adjusted q-value across the 8 cancer types, and a call: `accelerated` if q < 0.05 and the median difference is positive, `decelerated` if q < 0.05 and the median difference is negative, otherwise `ns`.

### Part 2 - phyloepigenetic analysis

Work separately inside each cancer type using only that type's tumour samples.

1. **Probe filter.** Drop every probe whose fraction of `NA` entries across the 960 samples exceeds 0.02. Impute every remaining missing beta with the mean of that probe over its non-missing entries (this imputed matrix is used for *all* downstream Part-2 computations).
2. **Events.** For a tumour sample of patient `i` and a retained probe `p`, an event occurs when `|beta(tumour) - beta(normal_i)| >= 0.30`.
3. **Tree.** Build the UPGMA tree of the cancer type's tumour samples using the `1 - Pearson correlation` distance between beta profiles over the retained probes (average linkage). Report the cophenetic correlation coefficient of the resulting dendrogram against the original distances, and write the dendrogram as Newick to `/app/output/trees/<CANCER_TYPE>.nwk` with branch lengths (each edge length is the difference between the child's and the parent's linkage height; the root carries no incoming edge).
4. **Event architecture.** A retained probe is an *event probe* of that cancer type if it is an event in at least one of its tumour samples. Among event probes, `trunk` means the event is present in **all** tumour samples of that cancer type, `private` means present in **exactly one**, everything else is `shared`. Report the counts and `private_fraction = n_private / n_event_probes`.
5. **Chromosome distribution.** Pooling all (tumour sample, retained probe) pairs over all 8 cancer types, report the fraction of event occurrences that fall on `chrX`, the fraction of all pairs that fall on `chrX`, and a one-sided (greater) hypergeometric p-value for over-representation of `chrX` among event occurrences.

## Constraints

1. The DNAm age must use the exact piecewise transform above; samples with a negative predictor must not be pushed through the exponential branch.
2. `normal_mae_years` is computed over normal samples only (480 of them) and reported rounded to 3 decimals.
3. Wilcoxon tests are two-sided, paired by `patient_id`, and BH adjustment is applied across all 8 cancer types; `call` is decided by the adjusted q-value at 0.05.
4. The probe filter threshold is strictly `> 0.02` missing fraction; the event threshold is `>= 0.30` on absolute beta difference against the matched normal of the same patient.
5. UPGMA means average linkage on the `1 - Pearson r` distance over the retained probes; the Newick file must carry branch lengths and tip labels equal to the sample IDs.
6. `trunk` requires the event in exactly all tumour samples of that cancer type (not "most"), `private` in exactly one.
7. `chrX` enrichment uses a one-sided hypergeometric test on event occurrences pooled across all cancer types.
8. All numbers must be emitted into a single JSON file `/app/output/results.json`; no extra top-level keys are required but the required keys must be present and correctly typed.

## Output

Write `/app/output/results.json`:

```json
{
  "clock": {
    "normal_mae_years": 14.871,
    "n_normal_samples": 480,
    "per_cancer_type": {
      "UCEC": {"n_pairs": 60, "median_delta_years": 20.114, "wilcoxon_p": 1.2e-11, "bh_q": 9.6e-11, "call": "accelerated"},
      "THCA": {"n_pairs": 60, "median_delta_years": -18.402, "wilcoxon_p": 3.4e-10, "bh_q": 1.4e-09, "call": "decelerated"},
      "BRCA": {"n_pairs": 60, "median_delta_years": 1.104, "wilcoxon_p": 0.51, "bh_q": 0.51, "call": "ns"}
    }
  },
  "phylo": {
    "per_cancer_type": {
      "UCEC": {"n_tumor_samples": 60, "n_event_probes": 437, "n_trunk_probes": 6, "n_private_probes": 170, "n_shared_probes": 261, "private_fraction": 0.389, "cophenetic_correlation": 0.8311}
    },
    "x_enrichment": {"n_event_occurrences": 7764, "n_chrx_event_occurrences": 1680, "event_chrx_fraction": 0.2164, "background_chrx_fraction": 0.0652, "hypergeom_p": 0.0}
  }
}
```

plus `/app/output/trees/UCEC.nwk`, `/app/output/trees/LUSC.nwk`, ... one file per cancer type.

## Scoring

The grader recomputes the whole analysis from `/app/data` with a protected reference implementation and compares your output field by field, awarding partial credit over 12 independent constraint groups: (1) normal-sample MAE, (2) number of pairs per cancer type, (3) median tumour-normal age differences, (4) acceleration/deceleration/ns calls, (5) Wilcoxon p-values, (6) event-probe counts, (7) trunk/private counts, (8) private fraction, (9) cophenetic correlation, (10) Newick clade topology, (11) Newick branch lengths, (12) chrX enrichment. `score = passed / 12`.
