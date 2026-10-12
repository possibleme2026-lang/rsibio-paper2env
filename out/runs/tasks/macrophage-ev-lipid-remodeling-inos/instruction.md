# Bacterial Stimulation Remodels Macrophage EV Lipids and Reveals iNOS Cargo

## Context
You are a computational biologist analyzing extracellular vesicles (EVs) from RAW264.7 macrophages stimulated with bacterial lysate (BL) or lipopolysaccharide (LPS). Lipidomic profiling and EV cargo activity assays were performed. Your goal is to identify stimulus-specific EV lipid signatures and determine whether EV-associated iNOS is enzymatically active and whether LPS-EVs induce endothelial inflammatory responses.

## Inputs
- `/app/data/lipidomics.csv`: feature table with 1,200 lipid features. Columns: `feature_id`, `mz`, `rt`, `adduct`, and raw intensity columns for 5 Control, 5 BL, 5 LPS EV samples, 3 QC samples, and 3 blank samples.
- `/app/data/lipid_reference_db.csv`: reference database with lipid classes, exact m/z, adduct, and RT. Contains true entries and decoys.
- `/app/data/sample_metadata.csv`: sample IDs, conditions (Control, BL, LPS, QC, Blank), sample type, particle counts.
- `/app/data/ev_activity.csv`: per-sample EV cargo measurements: `iNOS_band_intensity`, `nitrite_uM`, `TNF_pg_ml`, particle counts.
- `/app/data/endothelial_response.csv`: per-sample endothelial cell responses after EV treatment: `IL6_pg_ml`, `CCL5_pg_ml`, `ICAM1_od`.

## Task
1. Annotate lipid features by matching to the reference DB using m/z within 10 ppm, same adduct, and RT within 0.3 min. Choose the closest ppm match.
2. Filter features: keep those with at least 4 non-missing values in at least one condition (Control, BL, or LPS); QC CV <= 30% (computed on raw QC intensities); and mean raw intensity in at least one condition > 3 * mean blank raw intensity.
3. Normalize: add 1 to raw intensities, log2 transform, then median-center each sample (subtract sample median across filtered features).
4. Differential analysis: for each feature, compute log2FC as median(log2 intensity in condition) - median(log2 intensity in Control). Use Welch's t-test (BL vs Control, LPS vs Control) and Benjamini-Hochberg FDR.
5. Define signature groups: `BL_enriched` if log2FC_BL >= 1, FDR_BL <= 0.05, and lipid class is `SaturatedFattyAcid` or `Ceramide`; `LPS_enriched` if log2FC_LPS >= 1, FDR_LPS <= 0.05 (any class); `shared` if both; otherwise `none`.
6. Analyze EV activity: compute Control mean+2SD for nitrite and iNOS band. Classify a condition as `iNOS_positive` if its median nitrite > Control mean+2SD AND median iNOS band > Control mean+2SD.
7. Analyze endothelial response: for each cytokine (IL6, CCL5, ICAM1), test BL and LPS vs Control using Welch's t-test and BH FDR across all 6 tests. A condition is inducing if FDR <= 0.05 and log2FC >= 1, where log2FC = log2(median(condition)+1) - log2(median(Control)+1).

## Constraints
1. Output `/app/results/ev_lipid_signature.csv` with exact columns: `feature_id`, `lipid_class`, `log2FC_BL_vs_Control`, `fdr_BL`, `log2FC_LPS_vs_Control`, `fdr_LPS`, `signature_group`. Include all filtered features.
2. Output `/app/results/ev_summary.json` with keys: `iNOS_positive_conditions` (list), `nitrite_median_by_condition` (dict), `endothelial_induction` (dict mapping IL6, CCL5, ICAM1 to lists), `signature_counts` (dict).
3. Annotation must use 10 ppm m/z tolerance, same adduct, and RT tolerance 0.3 min. No unannotated feature may appear in the signature CSV.
4. Filtering must follow exactly the missingness, QC CV, and blank criteria above.
5. Differential statistics must use Welch's t-test and BH FDR; log2FC must be median-based.
6. Signature classification must follow the class restriction for BL_enriched.
7. iNOS positivity must use the Control mean+2SD rule for both nitrite and iNOS band.
8. Endothelial induction must use BH FDR across all 6 tests and log2FC >= 1.

## Output
- `/app/results/ev_lipid_signature.csv`: one row per filtered feature. Example row:
  `LIP0001,SaturatedFattyAcid,2.1,0.001,0.2,0.8,BL_enriched`
- `/app/results/ev_summary.json`: example:
  `{"iNOS_positive_conditions": ["BL", "LPS"], "nitrite_median_by_condition": {"Control": 2.1, "BL": 18.5, "LPS": 22.3}, "endothelial_induction": {"IL6": ["LPS"], "CCL5": ["LPS"], "ICAM1": ["LPS"]}, "signature_counts": {"BL_enriched": 150, "LPS_enriched": 100, "shared": 30, "none": 800}}`

## Scoring
The grader checks 8 constraints: (1) file schema, (2) annotation accuracy >= 95%, (3) filtering F1, (4) differential statistics within tolerance, (5) signature macro F1, (6) iNOS positivity accuracy, (7) endothelial induction accuracy, (8) biological validity (BL_enriched enriched for saturated FA/ceramide). Partial credit is awarded per constraint; final score is the average.
