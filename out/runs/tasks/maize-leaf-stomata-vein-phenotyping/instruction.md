# Maize Leaf Stomatal and Vein Phenotyping + QTL Mapping

You are a computational biologist analyzing maize leaf epidermal microscopy images from a MAGIC population. The goal is to implement an image analysis pipeline to quantify stomatal and vein traits, aggregate them per recombinant inbred line (RIL), and perform QTL mapping to identify genetic loci controlling these traits.

## Inputs
- `/app/data/images/`: 100 grayscale PNG images (512x512 pixels) of maize leaf epidermis. Each image represents a 1 mm x 1 mm area. Stomata are dark elliptical structures; veins are thick dark bands.
- `/app/data/image_metadata.csv`: columns `image_id`, `ril_id`, `pixels_per_mm` (512).
- `/app/data/genotypes.csv`: 50 RILs x 200 markers. First column `ril_id`, then marker IDs. Genotypes are 0 or 1 (homozygous founder alleles).
- `/app/data/marker_info.csv`: columns `marker_id`, `chromosome`, `position` (cM).

## Task
Write a Python script (or run commands) to:
1. For each image, detect stomata and veins, and compute:
   - `stomatal_density`: number of stomata per mm^2.
   - `stomatal_size`: mean area of stomata in mm^2.
   - `vein_density`: total vein length per mm^2 (mm/mm^2).
   - `vein_thickness`: mean vein thickness in mm.
   - `nnd_mean`: mean nearest-neighbor distance among stomata in mm.
   - `nnd_cv`: coefficient of variation of nearest-neighbor distances.
2. Aggregate per RIL by taking the mean of each trait across the RIL's images.
3. Perform QTL mapping for each of the six traits using simple linear regression at each marker. Compute LOD score as `-log10(p-value)` from the F-test. Identify significant QTLs with LOD > 3.0.

## Output
- `/app/output/traits.csv`: per-image traits. Columns: `image_id`, `ril_id`, `stomatal_density`, `stomatal_size`, `vein_density`, `vein_thickness`, `nnd_mean`, `nnd_cv`.
- `/app/output/ril_traits.csv`: per-RIL mean traits. Columns: `ril_id`, `stomatal_density`, `stomatal_size`, `vein_density`, `vein_thickness`, `nnd_mean`, `nnd_cv`.
- `/app/output/qtl_results.json`: list of significant QTLs. Each entry: `{"trait": "<trait>", "marker_id": "<marker>", "chromosome": <int>, "position": <float>, "lod": <float>}`. Only include QTLs with LOD > 3.0.

## Constraints
1. All spatial units must be calibrated using `pixels_per_mm` from metadata.
2. Stomatal density per image must be within 15% mean absolute percentage error (MAPE) of the true density.
3. Vein density per image must be within 10% MAPE.
4. Stomatal nearest-neighbor CV (`nnd_cv`) per image must be within 20% MAPE.
5. Per-RIL aggregated traits must have Pearson correlation > 0.8 with true per-RIL traits for each of the six traits.
6. QTL mapping must use the provided genotype and marker data. At least one significant QTL (LOD > 3.0) must be detected for each of the three traits: `stomatal_density`, `vein_density`, and `nnd_cv`. The detected QTL must be within 20 cM of a true QTL.
7. Output files must exist at the specified paths with the specified columns and be valid CSV/JSON.

## Scoring
The grader checks each constraint independently. Score = fraction of constraints passed. Partial credit is given.
