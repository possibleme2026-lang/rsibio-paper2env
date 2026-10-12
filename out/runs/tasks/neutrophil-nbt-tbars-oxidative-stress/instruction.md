# Neutrophil NBT-reactivity and TBARS/MDA analysis in women with infertility

## Context
A reproductive-immunology laboratory has used its in-house nitroblue tetrazolium (NBT) assay to measure superoxide-driven formazan deposition in neutrophils from 100 Libyan women (21 healthy fertile controls, 79 women with infertility), together with a thiobarbituric-acid-reactive-substances (TBARS) assay for malondialdehyde (MDA), the classical lipid-peroxidation marker. High-content microscopy images have already been segmented (the per-object measurements were exported to CSV) and the raw plate-reader output is on disk. Your job is to rebuild the study's quantitative results from these raw tables: QC'd single-cell NBT metrics, per-subject MDA concentrations, group comparisons, correlations and diagnostic performance.

## Inputs (all under `/app/data`)
- `subject_metadata.csv` — 100 rows: `subject_id`, `group` (fertile_control|infertile), `age_years`, `specimen_type` (blood|serum).
- `nbt_cell_measurements.csv` — ~18,000 segmented objects: `subject_id`, `field_id`, `cell_id`, `object_class` (neutrophil|lymphocyte|debris|platelet_aggregate), `cell_area_px`, `formazan_area_px`, `mean_deposit_od`.
- `field_qc.csv` — one row per microscopy field: `subject_id`, `field_id`, `focus_score` (0–1), `background_od`.
- `tbars_plate_readings.csv` — raw plate-reader output for three 96-well plates: `plate_id`, `well`, `absorbance_532`, `absorbance_600`.
- `tbars_well_map.csv` — well identities: `plate_id`, `well`, `well_type` (standard|sample), `sample_id`, `mda_nmol_per_ml` (standard concentration; empty for samples), `replicate` (1 or 2 for samples; empty for standards).
- `assay_parameters.json` — `pixel_size_um` (image calibration, micrometres per pixel) and `tbars_dilution_factor`.
- `gen_inputs.py` — the deterministic generator that produced the tables above (documentation only).

## Task
Implement the complete analysis pipeline and write the results to `/app/output/results.json` as a single JSON object.

**Step 1 — single-cell QC.** Apply the rules in this exact order; a cell that fails several rules is counted under the FIRST rule it fails:
 (a) discard every cell belonging to a microscopy field whose `focus_score` < 0.60;
 (b) discard every remaining object whose `object_class` is not `neutrophil`;
 (c) discard every remaining cell whose `cell_area_px` < 200.

**Step 2 — per-subject NBT metrics.** Convert `formazan_area_px` to square micrometres using the pixel size from `assay_parameters.json`. A retained neutrophil is NBT-positive when its formazan deposit area is >= 25.0 um2. For each subject compute `n_cells_retained`, `nbt_positive_pct` = 100 * (NBT-positive cells / retained cells) and `nbt_reactivity` = mean formazan deposit area (um2) over retained neutrophils.

**Step 3 — per-subject MDA.** Corrected absorbance = `absorbance_532` - `absorbance_600`. For each plate fit an unweighted ordinary-least-squares line of corrected absorbance on MDA concentration using ALL standard wells of that plate (including the zero standard); report slope, intercept and R-squared for each of the three plates. Discard any sample well whose `absorbance_532` > 2.000 (saturated). For every remaining sample well, concentration = (corrected absorbance - intercept) / slope * `tbars_dilution_factor`. The subject-level MDA is the mean over that subject's non-saturated replicate wells.

**Step 4 — group comparison.** For each of the three biomarkers (`nbt_positive_pct`, `nbt_reactivity`, `mda_nmol_per_ml`) run a two-sided Mann-Whitney U test between `fertile_control` and `infertile` and report the group medians and the p-value.

**Step 5 — correlations.** Report Spearman rho and its p-value for the three pairs: (nbt_positive_pct, nbt_reactivity), (nbt_reactivity, mda_nmol_per_ml), (nbt_positive_pct, mda_nmol_per_ml).

**Step 6 — diagnostic performance.** Treating `infertile` as the positive class, report the ROC AUC for `nbt_positive_pct` and for `nbt_reactivity`, and the Youden-optimal cutoff. The cutoff is the SMALLEST observed per-subject value c such that classifying a subject as infertile when its value >= c maximises (sensitivity + specificity - 1); also report the sensitivity and specificity obtained at that cutoff.

## Constraints
1. **QC funnel.** Rules (a)–(c) must be applied in the stated order and the reported counts must be exact integers: `n_cells_measured`, `n_fields_measured`, `n_cells_retained`, `n_fields_retained`, `n_excluded_focus`, `n_excluded_object_class`, `n_excluded_cell_area`, `n_wells_saturated_excluded`.
2. **Output shape.** `per_subject` holds exactly one object per subject, ordered by ascending `subject_id`, with `subject_id`/`group`/`specimen_type` copied from `subject_metadata.csv` and `n_cells_retained` equal to the true retained-cell count.
3. **NBT metrics.** Every `nbt_positive_pct` within 0.01 percentage points and every `nbt_reactivity` within 0.05 um2 of the reference values. The 25.0 threshold is in micrometres squared — the raw column is in pixels, so convert it.
4. **MDA / calibration.** Every subject MDA within 0.10 nmol/mL of the reference; all three plate standard curves reported with R-squared >= 0.99 and slope/intercept within 5% of the reference.
5. **Group results.** For all three biomarkers the infertile median must exceed the control median and the Mann-Whitney p-value must be < 0.001; medians must match the reference within 0.2 (%), 0.5 (um2) and 0.05 (nmol/mL) respectively.
6. **Correlations.** The three Spearman rho values must each be within 0.03 of the reference.
7. **AUC.** Both AUCs within 0.015 of the reference and both > 0.70.
8. **Youden cutoff.** Both cutoffs within 1.0 unit of the reference cutoff, and the reported sensitivity/specificity must be reproducible from your own `per_subject` table at your own cutoff (within 0.035).

## Output
Write `/app/output/results.json`:

```
{
  "qc": {"n_cells_measured": int, "n_fields_measured": int, "n_cells_retained": int,
         "n_fields_retained": int, "n_excluded_focus": int,
         "n_excluded_object_class": int, "n_excluded_cell_area": int,
         "n_wells_saturated_excluded": int},
  "standard_curves": {"plate_1": {"slope": float, "intercept": float, "r_squared": float},
                      "plate_2": {...}, "plate_3": {...}},
  "per_subject": [{"subject_id": "S001", "group": "infertile", "specimen_type": "serum",
                   "n_cells_retained": 168, "nbt_positive_pct": 61.9,
                   "nbt_reactivity": 39.417, "mda_nmol_per_ml": 5.204}, ...],
  "group_comparison": {"nbt_positive_pct": {"control_median": float,
                       "infertile_median": float, "p_value": float},
                       "nbt_reactivity": {...}, "mda_nmol_per_ml": {...}},
  "correlations": {"nbt_positive_pct_vs_nbt_reactivity": {"rho": float, "p_value": float},
                   "nbt_reactivity_vs_mda_nmol_per_ml": {...},
                   "nbt_positive_pct_vs_mda_nmol_per_ml": {...}},
  "diagnostic_performance": {"nbt_positive_pct": {"auc": float, "youden_cutoff": float,
                              "sensitivity": float, "specificity": float},
                              "nbt_reactivity": {...}}
}
```

Notes: the illustrative values in the example row above are NOT the answers. All numbers must be JSON numbers (not strings). `n_cells_measured` is the total number of rows in `nbt_cell_measurements.csv` and `n_fields_measured` the number of rows in `field_qc.csv`; `n_fields_retained` counts distinct (subject_id, field_id) pairs that contributed at least one retained cell.

## Scoring
Eight independently checked constraints (C1–C8 above). `score` = number of constraints passed / 8, written to `/logs/verifier/reward.json` together with `reward` = same value. A constraint passes only if every sub-check inside it passes; partial credit is therefore graded across constraints.
