# Synergistic cytotoxicity of prexasertib with lurbinectedin in SCLC

You are a computational biologist analyzing preclinical drug combination data. The paper hypothesizes that lurbinectedin activates G2/M checkpoint kinases Chk1/Chk2, and that adding the Chk1/Chk2 inhibitor prexasertib potentiates cell killing. You have viability matrices for four human SCLC cell lines treated with lurbinectedin, prexasertib, and their combination, plus Western blot densitometry for checkpoint and DNA-damage proteins.

## Inputs
- `/app/data/viability_matrix.csv`: 768 rows. Columns: `cell_line`, `lurbinectedin_nM`, `prexasertib_nM`, `replicate`, `viability`. Viability is fraction of vehicle (0-1.2). Doses are 8 x 8 full factorial including zero for each drug. Three biological replicates.
- `/app/data/western_blot.csv`: 720 rows. Columns: `cell_line`, `condition` (vehicle, L_1nM, L_10nM, P_30nM, P_300nM, L_10nM_P_300nM), `replicate`, `protein`, `intensity`. Proteins: beta-actin, pChk1_S296, pChk1_S317, pChk1_S345, pChk2_Y68, Chk1, Chk2, Wee1, gammaH2AX, cleaved_PARP.
- Cell lines: H69, H146, H526, DMS79.

## Task
1. Compute the mean viability for every combination dose of lurbinectedin and prexasertib in each cell line (mean across replicates).
2. Compute the Highest Single Agent (HSA) expected viability for each combination as `min(mean_viability_L(dL), mean_viability_P(dP))`, where `mean_viability_L(dL)` is the mean viability at lurbinectedin dose dL with prexasertib = 0, and `mean_viability_P(dP)` is the mean viability at prexasertib dose dP with lurbinectedin = 0. The HSA synergy score is `expected_HSA - mean_combination_viability`. Positive scores indicate synergy.
3. Classify each combination: `synergistic` if score > 0.05, `antagonistic` if score < -0.05, else `additive`.
4. Fit a 4-parameter logistic (4PL) dose-response curve to the single-agent data for each cell line and drug: `viability = bottom + (top - bottom) / (1 + (dose / IC50)^hill)`. Use only non-zero mean viabilities (mean across replicates). Fit in log10 dose space using `scipy.optimize.curve_fit` with bounds: bottom in [0, 0.5], top in [0.5, 1.2], IC50 in [0.01, 1000] nM, hill in [0.1, 5]. Initial guess: bottom=min(viability), top=max(viability), IC50=median(nonzero doses), hill=1.0. Report IC50_nM, hill, bottom, top, and R2. A fit is valid if R2 >= 0.90.
5. From the Western blot data, normalize each protein's intensity to beta-actin in the same well (same cell line, condition, replicate). For each cell line, protein, and condition, compute the median normalized value across replicates. Compute fold change relative to the vehicle condition for that cell line and protein. Report median fold changes.
6. Based on the fold changes, determine for each cell line:
   - `pchk_upregulated_by_lurbinectedin`: true if either (a) at least two of pChk1_S296, pChk1_S317, pChk1_S345 have fold change > 1.5 in L_10nM vs vehicle, or (b) pChk2_Y68 fold change > 1.5 in L_10nM vs vehicle.
   - `combination_suppresses_pchk`: true if the median fold change of at least two of the four phospho proteins (pChk1_S296, pChk1_S317, pChk1_S345, pChk2_Y68) in L_10nM_P_300nM is < 0.7 times the corresponding value in L_10nM.
   - `dna_damage_apoptosis`: true if gammaH2AX fold change > 2.0 and cleaved_PARP fold change > 1.5 in L_10nM_P_300nM vs vehicle.
7. Write the outputs. Also write a summary report JSON.

## Constraints
1. Output files must exist at exact paths: `/app/output/hsa_scores.csv`, `/app/output/ic50_summary.csv`, `/app/output/mechanism_summary.csv`, `/app/output/report.json`.
2. `hsa_scores.csv` must have columns `cell_line,lurbinectedin_nM,prexasertib_nM,mean_viability,expected_hsa,hsa_score,call`. It must contain exactly 4 cell lines x 64 dose combinations = 256 rows. HSA score must match the recomputed value within 0.01. Calls must follow the +/-0.05 rule.
3. `ic50_summary.csv` must have columns `cell_line,drug,ic50_nM,hill,bottom,top,r2`. It must contain exactly 8 rows (4 cell lines x 2 drugs). IC50 must be in nM. Fits with R2 < 0.90 must still be reported. IC50 values must be within 15% relative error of the reference 4PL fit.
4. `mechanism_summary.csv` must have columns `cell_line,protein,condition,median_fold_change`. It must contain exactly 4 x 9 x 6 = 216 rows (fold change for vehicle is 1.0 by definition). Each median fold change must match the reference within 5% relative error.
5. `report.json` must be valid JSON with keys: `hsa_synergy_cell_lines` (list of cell lines with >=3 synergistic combinations), `max_hsa_score` (object mapping cell line to max score), `ic50_nM` (object mapping cell line to object mapping drug to IC50), `mechanism` (object mapping cell line to the three boolean flags above). Numeric values must be JSON numbers.
6. Synergy must be cell-line specific: at least one cell line must have >=3 synergistic combinations, and the list must not include all four cell lines.
7. All numeric outputs must be derived from the provided data; do not hard-code cell-line names or expected results. The grader independently recomputes from the raw CSVs.

## Output
- `/app/output/hsa_scores.csv` example row:
`H69,3.0,30.0,0.42,0.71,0.29,synergistic`
- `/app/output/ic50_summary.csv` example row:
`H69,lurbinectedin,1.52,1.18,0.05,0.99,0.987`
- `/app/output/mechanism_summary.csv` example row:
`H69,pChk1_S296,vehicle,1.000`
- `/app/output/report.json` example:
```json
{
  "hsa_synergy_cell_lines": ["H69", "H146"],
  "max_hsa_score": {"H69": 0.29, "H146": 0.21, "H526": 0.04, "DMS79": -0.06},
  "ic50_nM": {"H69": {"lurbinectedin": 1.52, "prexasertib": 20.1}, ...},
  "mechanism": {"H69": {"pchk_upregulated_by_lurbinectedin": true, "combination_suppresses_pchk": true, "dna_damage_apoptosis": true}, ...}
}
```

## Scoring
The grader checks each constraint independently and assigns partial credit (score = fraction of constraints passed). Constraints 2-5 are checked with per-row/per-key tolerance; constraint 1 is existence; constraint 6 is the cell-line specificity check; constraint 7 is an internal-consistency check. Score is reported as a float in `[0,1]` in `/logs/verifier/reward.json` with key `score` and `reward`.
