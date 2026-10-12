# Evaluation protocol (mirrors the hidden evaluator)

All metrics are computed per field on 8-bit grey levels.

## 1. Evaluation mask

```
thr  = max(0.20 * max(dna), 8)
mask = binary_dilation(dna >= thr, disk(radius = 16))
if mask.sum() < 500: mask = all-True
```

`dna` is the observed DNA channel of the field. The mask covers the cell bodies and the perinuclear cytoplasm.

## 2. Masked Pearson correlation (PCC)

```
x = pred[mask] - mean(pred[mask])
y = gt[mask]   - mean(gt[mask])
PCC = sum(x*y) / sqrt(sum(x^2) * sum(y^2))      # 0 if either side is constant
```

Per-field PCC values are averaged over the 24 test fields; the *average PCC* of a structure is that mean, and the reported headline number is the mean over the five structures.

## 3. Naive baselines

* `dna_copy`   — the observed DNA channel used as prediction for every structure.
* `train_mean` — the pixel-wise mean image of that structure over the 40 training fields.

The best of the two is the baseline that a submission has to beat.

## 4. Radial profile

For every field the normalised elliptical radius `rho` of the *true* nucleus geometry is computed for all pixels (the smallest `rho` over the nuclei of the field). Pixels are binned into 13 bins of width 0.2 covering `rho` in [0, 2.6]; the mean intensity of a channel inside each bin (bins with fewer than 30 pixels are dropped) forms its radial profile. `profile_r` is the Pearson correlation between the predicted and the ground-truth profile.

## 5. Intensity statistics

For each structure, the 50th and the 95th percentile of the masked grey levels are averaged over the 24 test fields, both for the prediction and for the ground truth. Their ratio must stay inside [1/1.30, 1.30].

## 6. Scoring

Eight independent constraint groups are evaluated; `score = passed / 8`. See the task statement for the list.
