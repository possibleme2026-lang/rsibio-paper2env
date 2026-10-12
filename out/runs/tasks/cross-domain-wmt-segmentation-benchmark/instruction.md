# Cross-domain benchmark of white-matter tract segmentation (EWMTS)

You are the computational neuroimaging engineer for a reproducibility study of
white-matter tract (WMT) segmentation. Four deep-learning models (two CNN-based
networks, one transformer, one 3D U-Net) were trained on adult diffusion MRI.
They must now be evaluated on three independently acquired cohorts and their
predictions fused into the Ensemble White Matter Tract Segmentation (EWMTS)
consensus.

## Inputs

Everything is read-only under `/app/data`:

* `atlas/tract_labels.csv` - 24-tract atlas with columns
  `label_value,tract_name,hemisphere,category,benchmark_core`. Exactly 20 tracts
  are flagged `benchmark_core=yes` and form the scored panel. The four remaining
  tracts (`TPT_left`, `TPT_right`, `STFO_left`, `STFO_right`) are context only
  and must never appear in any deliverable.
* `atlas/template.nii.gz` - defines template space: shape `112x112x112`, 1.6 mm
  isotropic voxels. Reference maps and every metric live in this space.
* `refs/<subject_id>_ref.nii.gz` - reference tract label maps (integer atlas
  labels, 0 = background) in template space, 15 files.
* `native/<subject_id>/pred_<model>.nii.gz` - the four model predictions for that
  subject, stored in the subject's OWN native acquisition grid. Every subject has
  a different grid (own shape, voxel size, origin and a small rotation of the
  acquisition axes). Model names: `cnn_a`, `cnn_b`, `transformer`, `unet3d`.
* `subjects.csv` - `subject_id,dataset,group,age_years,sex,scanner,voxel_size_mm`.
  `dataset` is one of `adult_ref` (in-domain adult test cohort), `adult_alt`
  (second adult cohort, different protocol) or `pediatric`; 5 subjects each.

## Task

1. **Resample** every prediction into template space with voxel-wise
   nearest-neighbour interpolation, e.g.
   `nibabel.processing.resample_from_to(img, template_img, order=0)`.
   Labels must remain integers; index-space comparison is wrong because each
   native grid has a different shape and affine.
2. **Per-tract overlap.** For every (subject, map, core tract) triple - `map`
   runs over the four models plus the ensemble you build - compute the Dice
   overlap in template space, the reference and prediction tract volumes in mm3
   (voxel count x 4.096, the template voxel volume), and the reference / 
   prediction presence flags.
3. **Build EWMTS** for each subject, in template space, from the four resampled
   prediction maps: a label `L` is assigned to a voxel when at least 3 of the 4
   models assign exactly `L` to that voxel. Then, for each label, delete every
   6-connected (face-adjacency) component whose volume is below 150 mm3. Save
   every fused map as `/app/output/ensemble/<subject_id>_ewmts.nii.gz` on the
   template grid.
4. **Aggregate** per model and per dataset: mean Dice per dataset, detection F1
   per dataset, the within-domain mean, the pooled cross-domain mean and the
   degradation delta between them.
5. **Recommend** which model the paper should use, based on the aggregated
   numbers, including whether the ensemble actually recovers cross-domain
   accuracy over the individual models.

## Constraints

1. **Scored panel and row set.** `per_tract_metrics.csv` contains exactly
   15 subjects x 5 maps x 20 core tracts = 1500 data rows with the exact header
   below, and no row whose `tract_id` belongs to a `benchmark_core=no` tract.
2. **Template geometry.** Every `/app/output/ensemble/<subject_id>_ewmts.nii.gz`
   carries the template shape `(112,112,112)` and the template affine (tolerance
   1e-4), with integer labels in `0..24`.
3. **Overlap definition.** `dice = 2*|A & B| / (|A| + |B|)` over template-space
   voxels, where `A` is the reference mask and `B` the map mask of that tract for
   that subject; `dice = 0.0` when `|A| + |B| == 0`. Every triple must be
   reported, including tracts that are absent from the reference or from the
   prediction.
4. **Presence and detection.** A tract counts as present in a map when its volume
   is `>= 250 mm3`. Per-dataset detection F1 is `2TP/(2TP+FP+FN)` over all
   (subject, tract) pairs of that dataset, with TP/FP/FN defined from the
   reference and prediction presence flags. Report it for the four models and for
   EWMTS.
5. **EWMTS rule.** Voting is done in template space on the four resampled
   integer maps (not in native space), requires at least 3 of 4 exact label
   agreements, and is followed by the 6-connected component filter in mm3 - the
   exact rule above is the ground truth for the grader.
6. **Aggregation rules.** A per-domain mean Dice averages only over the
   (subject, tract) pairs whose reference marks the tract present;
   `within_domain_mean_dice` covers `adult_ref`, `cross_domain_mean_dice` pools
   `adult_alt` and `pediatric`, and `degradation_delta = within - cross`. All
   reported summary values are rounded to 4 decimals.
7. **Ordering and formatting.** `per_tract_metrics.csv` is sorted by
   `subject_id`, then by model in the fixed order
   `cnn_a, cnn_b, transformer, unet3d, EWMTS`, then by ascending `tract_id`;
   `dice` has 6 decimals, volumes 2 decimals, presence flags are lowercase
   `true`/`false`. Ties in `recommendation.json` are broken by ascending model
   name (plain string ordering, so `EWMTS` sorts before `cnn_a`).
8. **No external data.** Every number must be derived from `/app/data`; no
   network access and no hand-written metric values.

## Output

1. `/app/output/per_tract_metrics.csv`, exact header:
   `subject_id,dataset,model,tract_id,tract_name,dice,ref_volume_mm3,pred_volume_mm3,present_in_ref,present_in_pred`
   worked format example (values illustrative only):
   `sub_01,adult_ref,cnn_a,7,OR_left,0.841233,2145.02,2011.13,true,true`
2. `/app/output/benchmark_summary.json`:
   `{"n_subjects": 15, "n_scored_tract_pairs": <int>, "models": {"<model>": {"mean_dice_by_domain": {"adult_ref": f, "adult_alt": f, "pediatric": f}, "detection_f1_by_domain": {"adult_ref": f, "adult_alt": f, "pediatric": f}, "within_domain_mean_dice": f, "cross_domain_mean_dice": f, "degradation_delta": f}}}`
   with all five model keys present (`cnn_a`, `cnn_b`, `transformer`, `unet3d`,
   `EWMTS`). `n_scored_tract_pairs` is the number of (subject, core tract) pairs
   whose reference marks the tract present.
3. `/app/output/ensemble/<subject_id>_ewmts.nii.gz` - 15 fused label maps on the
   template grid.
4. `/app/output/recommendation.json`:
   `{"best_within_domain_model": s, "best_cross_domain_model": s, "worst_degradation_model": s, "best_pediatric_model": s, "ewmts_recovers_accuracy": bool}`
   where each string is the model with the largest corresponding mean Dice (or
   the largest `degradation_delta` for `worst_degradation_model`), and
   `ewmts_recovers_accuracy` is true only if the EWMTS cross-domain mean Dice is
   strictly greater than that of every individual model.

## Scoring

The grader recomputes all gold values from `/app/data` with the rules above and
scores eight independently checkable groups with partial credit:
`structure` 0.10 (files, exact header, key set, sorting, no non-core leakage),
`dice` 0.22 (per-row Dice agreement, 0.02 absolute tolerance plus mean absolute
error), `volume` 0.10, `presence` 0.08 (presence flags), `detection_f1` 0.12
per (model, dataset) F1 cell, `summary` 0.13 (aggregated values and counts),
`ensemble` 0.15 (template geometry and per-label macro Dice against the gold
fusion), `recommendation` 0.10. The final score is the weighted sum and is
written to `/logs/verifier/reward.json`.
