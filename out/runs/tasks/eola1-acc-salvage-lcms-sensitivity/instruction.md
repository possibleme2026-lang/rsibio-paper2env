# EOLA1-dependent acC salvage: quantify free nucleoside pools and dipyridamole sensitivity

## Background
N4-acetylcytidine (acC) is a naturally occurring RNA modification that blocks the first step of cytidine salvage (cytidine deamination). The human ASCH-domain protein EOLA1 deacetylates free acC so that the resulting cytidine re-enters nucleotide salvage. Two orthogonal quantitative phenotypes define this pathway: loss of EOLA1 raises the free intracellular acC pool without changing steady-state RNA acC, and EOLA1-deficient cells become hypersensitive to dipyridamole, an inhibitor of nucleoside export, a phenotype that is rescued by supplying cytidine. You are given the raw data from the two assays that establish these phenotypes and must reproduce the manuscript's quantitative result tables.

## Inputs
All inputs are comma-separated CSV with a header row, in `/app/data`.

| file | rows | contents |
|---|---|---|
| `lcms_channels.csv` | 8 | targeted LC-MS/MS channels. The instrument export kept only m/z and retention time; the analyte identities were lost. Columns: `channel_mz`, `rt_min`. |
| `compound_library.csv` | 12 | nucleoside standards available in the lab with monoisotopic-relevant molecular formula and the retention time measured on the same LC method. Columns: `compound`, `formula`, `expected_rt_min`. |
| `lcms_calibration.csv` | 30 | external calibration for 5 of the 8 channels, 6 concentration levels each (0, 2, 5, 20, 50, 100 nM). Columns: `channel_mz`, `level`, `conc_nM`, `area`. |
| `lcms_samples.csv` | 96 | integrated peak areas: 12 cell extracts (4 cell lines x 3 biological replicates) measured on all 8 channels. Columns: `sample_id`, `cell_line`, `replicate`, `channel_mz`, `area`. |
| `lcms_sample_metadata.csv` | 12 | per-extract protein amount and extraction volume. Columns: `sample_id`, `cell_line`, `replicate`, `protein_mg`, `extract_volume_mL`. |
| `viability_dose_response.csv` | 96 | dipyridamole dose-response plate P1: 4 cell lines x 8 doses x 3 replicates. `viability_pct` is the percentage of the vehicle-treated control of the same cell line. Columns: `well`, `cell_line`, `replicate`, `dipyridamole_uM`, `viability_pct`. |
| `viability_cytidine_rescue.csv` | 48 | plate P2: EOLA1-KO1 treated with dipyridamole alone or with 200 uM cytidine, 8 doses x 3 replicates. Columns: `well`, `cell_line`, `treatment`, `replicate`, `dipyridamole_uM`, `cytidine_uM`, `viability_pct`. |

Cell line labels are exactly `WT`, `EOLA1-KO1`, `EOLA1-KO2`, `KO1+EOLA1` (EOLA1-KO1 re-expressing EOLA1). Dipyridamole doses in both plates are 0.03, 0.1, 0.3, 1, 3, 10, 30 and 100 uM.

## Task
1. **Annotate the LC-MS channels.** For every channel decide which compound of the library (if any) and which adduct (`[M+H]+`, `[M+Na]+` or `[M+NH4]+`) it corresponds to. Compute monoisotopic masses from the molecular formulae (pyteomics, RDKit or your own accurate atomic-mass table), require m/z agreement within 10 ppm *and* retention-time agreement within 0.3 min against `expected_rt_min`. A channel that matches nothing must be reported as unassigned.
2. **Build the calibration curves.** For each calibrated channel fit peak area against concentration by ordinary least squares (`area = slope * conc + intercept`) and use the fit to convert each sample peak area into an extract concentration in nM.
3. **Normalise to protein.** Convert extract concentrations into pmol per mg protein using the extract volume and protein amount of that extract: `pmol/mg = conc_nM * extract_volume_mL / protein_mg`.
4. **Quantify per cell line.** For each of the 5 calibrated analytes and each cell line report the mean and the sample standard deviation (n = 3 biological replicates), the fold change versus WT, a two-sided Welch t-test p-value against WT, and a significance call. Call a comparison significant when the Benjamini-Hochberg adjusted p-value < 0.05 **and** |log2(fold change)| >= 1; apply BH within each analyte across the three comparisons against WT. WT rows have fold change 1.0, p = 1.0 and are not significant.
5. **Fit the dose-response plates.** Fit every curve of both plates with the 4-parameter logistic model
   `viability(x) = bottom + (top - bottom) / (1 + 10^((log10(IC50) - x) * hill))`, where `x = log10([dipyridamole] in uM)`,
   by non-linear least squares **on the replicate-level wells** (24 wells per curve on plate P1, 24 per treatment on plate P2). Report IC50 in uM, the Hill slope and the coefficient of determination R^2.
6. **Derive the phenotype calls** described under Output into `summary.json`.

## Constraints
1. Every one of the 8 channels must be annotated within 10 ppm and 0.3 min of the correct compound/adduct; the unmatched channel must be reported as `unassigned` with adduct `none`.
2. Quantified values must be normalised to protein: mean pmol/mg must agree with the hidden reference within 15 % for all 20 cell-line/analyte combinations.
3. Statistics must follow the stated rule: two-sided Welch t-test against WT, Benjamini-Hochberg within analyte, significance requires adjusted p < 0.05 AND |log2 fold change| >= 1.
4. Dose-response curves must be fitted with the stated 4PL model on replicate-level data; IC50 must agree with the hidden reference within 25 % relative error, Hill slope within 0.35 of 1.0, and R^2 >= 0.97 for all six curves.
5. Phenotype thresholds: both EOLA1-KO lines must have an IC50 at least 2-fold lower than WT (EOLA1 loss sensitises), the re-expressing line must be within 1.5-fold of WT, and free acC in the re-expressing line must be within 1.5-fold of WT.
6. Cytidine rescue: in plate P2 the IC50 of dipyridamole + cytidine must be at least 2-fold higher than dipyridamole alone, and the rescue index reported in `summary.json` must match the fitted ratio within 20 %.
7. All four output files must exist with exactly the specified column headers and row counts, values must be plain numbers (no units, no thousands separators), and all standard deviations must be >= 0.
8. `summary.json` must be internally consistent with the CSVs: fold changes, IC50 values, channel counts and rescue index must agree within 10 % between the two representations.

## Output
Write exactly four files into `/app/output`.

**`/app/output/channel_annotation.csv`** - header `channel_mz,rt_min,compound,adduct,ppm_error`, one row per input channel, ascending m/z. Worked example row (values are illustrative):
`244.0928,2.10,cytidine,[M+H]+,0.4`
For an unmatched channel use `312.1444,8.60,unassigned,none,` (empty ppm_error).

**`/app/output/nucleoside_quantification.csv`** - header `cell_line,analyte,mean_pmol_per_mg,sd_pmol_per_mg,fold_change_vs_WT,ttest_p,significant`, 20 rows (4 cell lines x the 5 calibrated analytes). Analyte names must be exactly as in `compound_library.csv`; `significant` is `true`/`false`. Worked example row (illustrative):
`EOLA1-KO1,N4-acetylcytidine,5.58,0.28,4.65,0.00012,true`

**`/app/output/ic50_summary.csv`** - header `dataset,cell_line,treatment,ic50_uM,hill_slope,r_squared`, 6 rows: four rows with `dataset=dose_response` and `treatment=dipyridamole` (WT, EOLA1-KO1, EOLA1-KO2, KO1+EOLA1), then two rows with `dataset=cytidine_rescue` (EOLA1-KO1 with `treatment=dipyridamole` and `treatment=dipyridamole+cytidine`). Worked example row (illustrative):
`dose_response,WT,dipyridamole,12.0,1.02,0.998`

**`/app/output/summary.json`** - exactly these keys:
`n_channels_annotated` (int), `n_channels_unassigned` (int), `free_acC_fold_change_KO1_vs_WT`, `free_acC_fold_change_KO2_vs_WT`, `free_acC_restored_by_reexpression` (bool, true if re-expressed free acC within 1.5-fold of WT), `other_nucleosides_unchanged` (bool, true if none of the other four analytes is significant in either KO line), `ic50_uM_WT`, `ic50_uM_EOLA1_KO1`, `ic50_uM_EOLA1_KO2`, `ic50_uM_KO1_reexpressed`, `ic50_uM_KO1_dipyridamole`, `ic50_uM_KO1_dipyridamole_cytidine`, `cytidine_rescue_index` (ratio of the two previous values), `cytidine_rescues_viability` (bool, true if index >= 2), `EOLA1_loss_sensitises_to_dipyridamole` (bool, true if both KO lines have IC50 <= 0.5x WT).

## Scoring
The grader evaluates eight independent constraints: (1) channel annotation, (2) absolute quantification accuracy, (3) statistically correct significance calls, (4) 4PL parameters, (5) sensitivity/restoration phenotype, (6) cytidine rescue, (7) file/format compliance, (8) internal consistency of `summary.json` with the CSVs. Each constraint is pass/fail and the final score is the fraction of constraints passed; partial credit is therefore possible for any subset of correctly completed analyses.
