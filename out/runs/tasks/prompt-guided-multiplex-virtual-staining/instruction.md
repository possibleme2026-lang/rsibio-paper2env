# Prompt-guided multiplex virtual staining from a single DNA channel

You are re-implementing the *inference half* of the decoupled semantic-supervision virtual-staining framework. In that framework, pixel-aligned multiplex supervision is replaced by **biological prompts**: instead of learning from paired multi-channel fluorescence, each subcellular structure is described symbolically and synthesised independently from a single acquired channel. Your job is to build such a prompt-driven synthesiser and to reproduce a held-out multiplex dataset from its nuclear (DNA) channel alone.

This is the *channel-deficient* regime: in the test fields only the DNA channel was acquired. Everything else must be synthesised.

## Inputs

- `/app/data/train/` — 40 fields, fully multiplexed (the only place where paired targets exist).
  Files are `R001_dna.png` … `R040_er.png`, 8-bit grayscale PNG, 160×160 px.
  Channel names: `dna`, `actin`, `tubulin`, `mito`, `golgi`, `er`.
- `/app/data/valid/` — 16 fields, fully multiplexed (`V001_*` … `V016_*`), same format. Use these for self-evaluation.
- `/app/data/test/` — 24 fields, **DNA channel only** (`T001_dna.png` … `T024_dna.png`). These are the fields you must complete.
- `/app/data/prompts.json` — the semantic supervision. One record per structure giving its morphology, its position in the elliptical nuclear coordinate frame `(u, v, rho)` and the superposition rule.
- `/app/data/protocol.md` — the exact metric definitions used by the evaluator (evaluation mask, masked PCC, radial profile, intensity statistics).

The images were synthesised by a deterministic multiplex-fluorescence simulator that is **not** shipped with the task: the generator was removed from the image after the data were written. Only `prompts.json` and the measurement protocol are available, exactly as a real annotator-facing prompt set would be.

## Task

1. **Explore.** Characterise the training set: per-channel intensity distributions inside the evaluation mask, background levels, and the relationship between the DNA channel and each target channel.
2. **Recover cell geometry.** Segment the nuclei in every field, and convert each nucleus into the elliptical frame required by `prompts.json` (centroid, semi-minor radius `r`, elongation `e`, major-axis angle `theta`). Note that any threshold-based segmentation yields a *systematically biased* nucleus radius — you must calibrate that bias against the training data.
3. **Implement the structures.** For each of the five target structures, render the cell-wise contribution described by its prompt (cortical bands, stress fibres, radiating filaments, punctate lattices, polarised ribbons, reticular meshes) and sum the contributions of all cells in the field.
4. **Calibrate intensities.** Grey levels are deliberately absent from `prompts.json`. Fit an affine intensity mapping per channel so that the rendered raw signal reproduces the per-channel grey-level distribution (at least the 50th and 95th percentiles of the masked pixels) of the training data.
5. **Validate and iterate.** Apply the full pipeline to `/app/data/valid/`, compute the masked PCC of `protocol.md` against the available validation targets, and tune your geometry calibration until the validation PCC stops improving.
6. **Predict.** Run the tuned pipeline on the 24 test DNA fields and write the results.

## Constraints

1. Write exactly 120 test predictions to `/app/output/test_predictions/{field}_{structure}.png` for the 24 test fields (`T001`…`T024`) × 5 structures (`actin`, `tubulin`, `mito`, `golgi`, `er`). Every file must be a 160×160 8-bit grayscale PNG.
2. Write exactly 80 predictions of the same form to `/app/output/valid_predictions/{field}_{structure}.png` for the 16 validation fields (`V001`…`V016`).
3. Do not modify anything under `/app/data/`; predictions must be computed from the supplied DNA channels as they are.
4. Every predicted channel must be a genuine image: standard deviation inside the evaluation mask ≥ 6 grey levels, and it must not be a copy of the DNA channel (mean absolute difference from `{field}_dna.png` > 8 grey levels).
5. The mean masked Pearson correlation (PCC, defined in `protocol.md`) over the 5 structures and 24 test fields must be ≥ 0.70, and every individual structure must reach ≥ 0.50.
6. The mean masked PCC must exceed the best of two naive baselines — (a) predicting the DNA channel for every structure, (b) predicting the training-set mean image of that structure — by at least 0.25.
7. Structure placement must be quantitatively correct, not merely correlated: for each structure the *radial profile* of the prediction (mean intensity versus normalised elliptical radius `rho`, 0.2-wide bins in [0, 2.6]) must correlate with the ground-truth radial profile with Pearson r ≥ 0.75.
8. Intensity calibration must be correct: for each structure, the mean masked 50th and 95th percentiles over the 24 test fields must lie within ±30 % of the ground-truth values.
9. Write `/app/output/report.json` containing `method` (short string), `valid_pcc` (object with the five per-structure PCC values you measured on the validation split) and `valid_pcc_mean`. The reported values are re-computed from your `/app/output/valid_predictions/` by the evaluator and must agree to within 0.02.

## Output

```
/app/output/test_predictions/T007_actin.png      # 160x160 uint8 PNG
/app/output/test_predictions/T007_tubulin.png
...
/app/output/valid_predictions/V003_golgi.png
...
/app/output/report.json
```

`report.json` example:

```json
{
  "method": "prompt-driven elliptical renderer, radius bias calibrated on train",
  "valid_pcc": {"actin": 0.93, "tubulin": 0.88, "mito": 0.90, "golgi": 0.91, "er": 0.86},
  "valid_pcc_mean": 0.896
}
```

## Scoring

The evaluator regenerates the hidden ground truth for the 24 test fields from a protected seed and checks eight independent constraint groups; the score is the fraction that pass:

1. `format` — file inventory, PNG shape/dtype, and integrity of the supplied DNA inputs.
2. `nondegenerate` — per-image variance and non-duplication of the DNA channel.
3. `pcc_avg` — mean masked PCC ≥ 0.70.
4. `pcc_per_channel` — every structure ≥ 0.50.
5. `beats_baselines` — margin of ≥ 0.25 over the two naive baselines.
6. `intensity_match` — masked 50th/95th percentile within ±30 %.
7. `profile_match` — radial-profile correlation ≥ 0.75 for every structure.
8. `self_report` — validation predictions complete, validation mean PCC ≥ 0.65, and `report.json` consistent with the recomputed values.

Partial credit is granted per group.
