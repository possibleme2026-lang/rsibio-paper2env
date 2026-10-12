# Depth-resolved 2P vs 3P microscopy through intact murine cortical bone

You are the imaging-physics analyst of a bone-biology laboratory. The same intact mouse metatarsal was imaged twice through its intact cortical shell: once with two-photon (2P) excitation and once with three-photon (3P) excitation. Matching point-spread-function (PSF) bead calibration stacks were acquired with each modality. Your job is to quantify how image quality decays with depth for each modality and to determine the depth beyond which 3P outperforms 2P.

## Inputs (read-only, under /app/data)

| file | content |
|---|---|
| `2p_stack.tif` | 16-bit grayscale multi-page TIFF, 24 pages of 512x512. Page index k is the focal depth z = 10*k micrometres (0, 10, ..., 230 um). Lateral sampling 0.2 um/pixel, field of view 102.4 x 102.4 um. Each page shows the fluorescent osteocyte cell bodies that are in focus at that depth, attenuated by the overlying cortical bone. |
| `3p_stack.tif` | identical geometry, three-photon excitation. Same field of view and same cells as `2p_stack.tif`. |
| `beads_2p.tif` | 16-bit multi-page TIFF, 24 pages of 128x128, lateral sampling 0.1 um/pixel, same 10 um z-step. Each page holds one sub-resolution fluorescent bead at the exact centre of the field (centre pixel coordinate x = y = 63.5), imaged at that depth. |
| `beads_3p.tif` | the same bead calibration for three-photon excitation. |

Useful inspection tool installed in the image: `tiffinfo` (libtiff tools), plus numpy / scipy / pandas / tifffile / scikit-image.

**Do not modify, move or rewrite anything under /app/data.** The inputs are checksum-verified; any modification voids the run.

## Task

1. For every plane of each cell stack, estimate the background level and the noise level, segment the fluorescent signal, and compute four numbers: the detected-signal mean, the signal-to-background ratio (SBR), the signal-to-noise ratio (SNR), and the number of detected pixels.
2. For every plane of each bead stack, measure the lateral PSF full width at half maximum (FWHM) and the bead peak amplitude.
3. Fit an exponential attenuation model to the bead peak amplitude versus depth, separately for 2P and 3P, over the shallow depth range only.
4. Decide the depth beyond which 3P is better than 2P on **both** the SNR and the resolution criterion.
5. Write the per-depth table and the summary file described under Output.

All numeric definitions below are mandatory; the grader recomputes them independently from a protected seed.

### Metric definitions (use exactly these)

For a cell plane `P` (a 512x512 array of grey levels, as float):

* background level `b = median(P)`;
* noise level `sigma = 1.4826 * median(|P - b|)`;
* signal mask = every pixel with `P > b + 5*sigma`;
* let `n` be the number of masked pixels. If `n < 20` treat the plane as signal free: report `signal_mean = b`, `snr = 0.0`, `sbr = 0.0` (but still report the true count `n` in `mask_pixels`). Otherwise `signal_mean = mean(P[mask])`, `snr = (signal_mean - b)/sigma`, `sbr = (signal_mean - b)/b`.

For a bead plane `B`:

* work on the central 96x96 window `W = B[16:112, 16:112]`;
* background level `b = median(W)`, noise level `sigma = 1.4826 * median(|W - b|)`;
* weights `w = max(W - (b + 3*sigma), 0)`;
* centroid `x0 = sum(w*x)/sum(w)`, `y0 = sum(w*y)/sum(w)` (pixel coordinates, integers);
* second moments `sx2 = sum(w*(x-x0)^2)/sum(w)`, `sy2 = sum(w*(y-y0)^2)/sum(w)` (in pixel^2);
* lateral FWHM `= 2.3548 * sqrt((sx2 + sy2)/2) * 0.1` micrometres;
* bead peak amplitude `= sum(w) / (2*pi*sqrt(sx2*sy2))` in grey levels (this is the height of the Gaussian that has the same integral and widths).

Attenuation model: `peak(z) = A * exp(-mu * z)`. Estimate `mu` (in um^-1) by ordinary least squares of `ln(peak)` on `z`, using **only the 13 planes with z = 0, 10, ..., 120 um**. Report `mu` for both modalities and their ratio `mu_2p / mu_3p`.

3P advantage depth: the **smallest** sampled depth `z*` (one of 0, 10, ..., 230 um) such that for every sampled depth `z >= z*` BOTH conditions hold: `snr_3p(z) > snr_2p(z)` AND `fwhm_3p(z) < fwhm_2p(z)`. If no such depth exists report `-1`.

## Constraints

1. `/app/results/metrics.csv` must have exactly this header, in this order: `modality,depth_um,mask_pixels,signal_mean,sbr,snr,fwhm_lateral_um`.
2. The CSV must contain exactly 48 data rows: modality `2p` first (24 rows, depths 0,10,...,230 um ascending), then modality `3p` (24 rows, same grid).
3. `depth_um` values must be exactly the sampled grid 0, 10, ..., 230 (integers or floats are both accepted).
4. `snr` must follow the definition above, including the `n < 20` signal-free rule (2P planes deeper than about 160 um legitimately report `snr = 0.0`).
5. `fwhm_lateral_um` must come from the bead window defined above (central 96x96, `b + 3*sigma` threshold, intensity-weighted second moments), not from a profile eyeball or a different window.
6. `mu_*_um_inv` must be fitted on the 0-120 um range only; using the full 0-230 um range is wrong.
7. `/app/results/summary.json` must be a JSON object with the numeric keys `mu_2p_um_inv`, `mu_3p_um_inv`, `mu_ratio`, `advantage_depth_um`.
8. All reported numbers must be finite and non-negative (except `advantage_depth_um`, which may be -1).
9. The input TIFF files must be left byte-identical to their distributed state.

## Output

* `/app/results/metrics.csv` - the 48-row per-depth table, e.g.

```
modality,depth_um,mask_pixels,signal_mean,sbr,snr,fwhm_lateral_um
2p,0,23551,742.31,2.0944,29.692,0.4187
2p,10,22310,701.05,1.9123,28.042,0.4231
...
3p,230,16987,392.44,0.1810,15.698,1.5311
```

(the numbers above are illustrative only).

* `/app/results/summary.json`:

```json
{
  "mu_2p_um_inv": 0.02487,
  "mu_3p_um_inv": 0.00529,
  "mu_ratio": 4.70,
  "advantage_depth_um": 50.0
}
```

## Scoring

The grader regenerates the phantom from a protected seed and recomputes every gold number with an independent implementation. Partial credit is granted per row and per constraint:

* schema / row grid / ordering: 0.10
* `snr` per row (10% relative tolerance, 1.0 absolute floor): 0.18
* `sbr` per row (10% / 0.05): 0.10
* `signal_mean` per row (5% / 2.0): 0.10
* `mask_pixels` per row (10% / 50 px): 0.12
* `fwhm_lateral_um` per row (15% / 0.05 um): 0.15
* the three attenuation numbers (10% / 12%): 0.10
* `advantage_depth_um` within +/-10 um (one plane): 0.10
* physical validity: per modality the SNR series must be non-increasing and the FWHM series non-decreasing with depth (5% slack), 2P must beat 3P at 0 um and 3P must beat 2P at 230 um: 0.05

Score is the weighted fraction of constraints passed; `reward` equals `score`.
