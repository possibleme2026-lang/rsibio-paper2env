# Confirmatory re-analysis of a multicenter preclinical RCT of rhNRGβ1 in a murine NF2-schwannomatosis model

You are the lead biostatistician of a pre-registered, multicenter preclinical randomized controlled trial. 216 Nf2-flox;P0-Cre;Nefh-Cre mice were randomized at three independent research sites to systemic rhNRGβ1 (10 µg/kg) or vehicle for 13 weeks after a standardized sciatic nerve crush. The primary outcome is week-13 sciatic nerve thickness (µm) obtained with an automated Fiji morphometry macro; secondary outcomes are quantitative Western blotting and DIA proteomics of a pre-defined 60-protein functional marker panel.

Your job is to execute the pre-specified analysis plan exactly as written below, on the raw data supplied in `/app/data`, and to emit machine-readable results. The analysis must be a faithful implementation of the plan — the numbers will be recomputed independently and compared to your output within tolerance, so any deviation in model specification, missing-data handling, normalization or multiple-testing correction is detected.

## Inputs

All raw data live in `/app/data` (comma- or tab-separated, first line is the header):

- `nerve_thickness.csv` — one row per randomized mouse (n = 216), columns `mouse_id, site, sex, arm, baseline_um, week13_um`. `site` ∈ {S1, S2, S3}; `sex` ∈ {male, female}; `arm` ∈ {Vehicle, rhNRGb1}. `baseline_um` is the pre-treatment (post-crush) nerve thickness in µm and is never missing. `week13_um` is the primary outcome in µm and is the empty string for mice lost to follow-up (procedure-related deaths, exclusion at the site level).
- `western_blot.csv` — one row per mouse in the pre-specified densitometry subset (n = 144), columns `mouse_id, site, sex, arm, actin, MBP, ERBB2, pERBB2, ERK1_2, pERK1_2, YAP1`. All protein columns are raw band intensities in arbitrary densitometry units; `actin` is the lane loading control.
- `proteomics_panel.tsv` — DIA quantification of the 60-protein marker panel in long format: one row per (`sample_id`, `protein`) pair for all 216 samples (12,960 rows). Columns `sample_id, site, sex, arm, protein, intensity`; `intensity` is a linear (not log) value and is the empty string when the protein was not quantified in that sample.
- `protein_panel_pathways.tsv` — tab-separated `protein` / `pathway` membership for the 60 panel proteins.

## Task

Implement the pre-specified analysis plan below, then write the two output files. Work from the raw data; do not hardcode any statistic.

### A. Primary outcome (complete case)

A1. Fit an ordinary least squares model of `week13_um` on `arm` (rhNRGb1 = 1, Vehicle = 0), `site` (categorical), `sex` (categorical) and `baseline_um` (continuous), with an intercept, on the complete cases. Report the `arm` coefficient and its standard error, t statistic, two-sided p-value (t distribution, n − k residual degrees of freedom) and 95% confidence interval. `n_complete` is the number of mice with a non-missing primary outcome.

A2. Report how many randomised mice are missing the primary outcome in each arm.

A3. Sensitivity analysis with extreme-case imputation. **Best case** (most favourable to the drug): every missing week-13 value in the rhNRGb1 arm is replaced by the minimum observed week-13 value of the rhNRGb1 arm, and every missing week-13 value in the Vehicle arm is replaced by the maximum observed week-13 value of the Vehicle arm. **Worst case**: the mirrored rule (rhNRGb1 → maximum observed week-13 of the rhNRGb1 arm; Vehicle → minimum observed week-13 of the Vehicle arm). Refit the model of A1 on each completed dataset and report the two-sided p-value of the `arm` coefficient.

### B. Western blot

For each of the six targets (`MBP`, `ERBB2`, `pERBB2`, `ERK1_2`, `pERK1_2`, `YAP1`) form the actin-normalised ratio (target / actin), log2-transform it, and fit an OLS model of the log2 ratio on `arm` + `site` + `sex` (intercept, no baseline covariate). Report the `arm` effect (log2 fold change), its two-sided p-value, and the Benjamini–Hochberg adjusted p-value computed across exactly these six p-values.

### C. Proteomics panel

C1. Take log2 of the linear intensities. For every protein compute the missing fraction over all 216 samples (empty intensity = missing).

C2. Retain proteins whose missing fraction is ≤ 0.20; drop the rest entirely.

C3. For retained proteins only, replace a missing log2 value by the median of the observed log2 values of that same protein in the same site.

C4. Median-centre each sample: for every sample subtract the median of that sample's imputed retained-protein log2 values from each of its retained-protein values.

C5. Fit the OLS model `value ~ arm + site + sex` (intercept) separately for each retained protein on the centred values, and apply Benjamini–Hochberg to the resulting p-values across all retained proteins.

C6. Report the number of retained proteins, the number of retained proteins with adjusted p < 0.05, and the number of retained proteins with |effect| ≥ 0.585 (i.e. at least a 1.5-fold change in either direction).

C7. Using `protein_panel_pathways.tsv`, summarise the three signalling pathways `Hippo_YAP`, `mTORC1` and `RTK_Ras_MAPK`: over the retained members of each pathway report the number of retained members, the mean `arm` effect (mean log2 coefficient) and the number of retained members with adjusted p < 0.05.

### D. Decision rule

Set `primary_conclusion` to `"efficacy"` if the complete-case primary p-value is < 0.05 **and** the primary `arm` effect is negative (thinner nerves in the rhNRGβ1 arm); otherwise set it to `"no_efficacy"`.

### E. Outputs

E1. `/app/out/results.json` — a single JSON object with exactly the following structure (all numbers are plain JSON numbers, not strings):

```json
{
  "primary": {
    "n_complete": 0,
    "n_missing_treated": 0,
    "n_missing_vehicle": 0,
    "effect_um": 0.0,
    "se_um": 0.0,
    "t_stat": 0.0,
    "p_value": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.0,
    "p_best_case": 0.0,
    "p_worst_case": 0.0
  },
  "western": {
    "MBP":    {"effect_log2": 0.0, "p_value": 0.0, "p_adj": 0.0},
    "ERBB2":  {"effect_log2": 0.0, "p_value": 0.0, "p_adj": 0.0},
    "pERBB2": {"effect_log2": 0.0, "p_value": 0.0, "p_adj": 0.0},
    "ERK1_2": {"effect_log2": 0.0, "p_value": 0.0, "p_adj": 0.0},
    "pERK1_2":{"effect_log2": 0.0, "p_value": 0.0, "p_adj": 0.0},
    "YAP1":   {"effect_log2": 0.0, "p_value": 0.0, "p_adj": 0.0}
  },
  "proteomics": {
    "n_proteins_input": 0,
    "n_proteins_retained": 0,
    "n_significant_padj": 0,
    "n_abs_effect_ge_0.585": 0,
    "pathways": {
      "Hippo_YAP":   {"n_members_retained": 0, "mean_effect": 0.0, "n_significant": 0},
      "mTORC1":      {"n_members_retained": 0, "mean_effect": 0.0, "n_significant": 0},
      "RTK_Ras_MAPK":{"n_members_retained": 0, "mean_effect": 0.0, "n_significant": 0}
    }
  },
  "primary_conclusion": "no_efficacy"
}
```

E2. `/app/out/proteomics_results.tsv` — tab-separated, header exactly `protein<TAB>effect<TAB>se<TAB>t<TAB>p<TAB>padj`, one row per retained protein, sorted by `padj` ascending and, for ties, by `protein` ascending (lexicographic). Do not round values to 2 decimals; keep at least 6 significant digits. Illustrative row (values are placeholders, not the expected answer):

```
YAP1	0.104300	0.051200	2.036	0.043100	0.118200
```

## Constraints

1. **Primary model specification.** The primary analysis must be an OLS of `week13_um` on arm indicator, categorical site, categorical sex and continuous `baseline_um` on complete cases only; the reported effect must be the rhNRGb1 − Vehicle adjusted difference in µm, with the t-distribution two-sided p-value and the 95% CI reported on the same scale.
2. **Missing-data accounting and extreme-case sensitivity.** Missing primary outcomes must be counted per arm and the best-case/worst-case imputation must use exactly the min/max rules defined in A3, refitting the identical model.
3. **Western blot normalisation.** Target signals must be divided by the actin loading control before log2 transformation, the six-target model must not include a baseline covariate, and the adjusted p-values must be Benjamini–Hochberg over exactly the six targets.
4. **Proteomics quality control.** Proteins with a missing fraction > 0.20 (over all 216 samples) must be dropped; missing values of retained proteins must be filled with the within-protein, within-site median; each sample must be median-centred over its retained proteins after imputation.
5. **Proteomics statistics and reporting.** Each retained protein must be modelled as `value ~ arm + site + sex`, Benjamini–Hochberg applied across all retained proteins, and the hit counts and per-pathway summaries reported as defined in C6/C7.
6. **Output format.** `results.json` must follow the schema exactly (identical key names, all numeric fields as JSON numbers); `proteomics_results.tsv` must carry the exact header, one row per retained protein, and be sorted by adjusted p then protein name.
7. **Decision rule.** `primary_conclusion` must be derived from the complete-case primary p-value and effect sign, not from any other model.
8. **No hardcoding.** Every number must be computed from the supplied files; the primary data files must not be modified.

## Scoring

The grader regenerates the dataset deterministically from a protected seed, recomputes the whole analysis plan independently, and checks the following items, each worth an equal share of the score: submission files present; primary effect + SE; primary t + p; primary 95% CI; per-arm missing counts; p_best_case; p_worst_case; Western core markers (MBP, ERBB2: effect, p, adjusted p); remaining four Western targets (effect and adjusted p); number of retained proteomics proteins; proteomics hit counts (adjusted p < 0.05 and |effect| ≥ 0.585); per-pathway summaries; TSV header/row count; TSV sorting; TSV numeric values; primary conclusion; data integrity of `/app/data`. Numeric agreement is judged with a small absolute plus 2 % relative tolerance. `score = passed / total`, and the same value is written as `reward`.
