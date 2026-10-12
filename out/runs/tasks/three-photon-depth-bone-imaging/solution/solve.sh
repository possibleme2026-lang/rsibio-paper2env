#!/bin/bash
set -euo pipefail
mkdir -p /app/results
python3 - <<'PY'
import json

import numpy as np
import pandas as pd
import tifffile

DEPTHS = np.arange(0.0, 231.0, 10.0)
BEAD_UM = 0.1
MIN_MASK_PIXELS = 20


def cell_metrics(stack):
    out = []
    for k, z in enumerate(DEPTHS):
        p = np.asarray(stack[k], dtype=np.float64)
        b = float(np.median(p))
        s = 1.4826 * float(np.median(np.abs(p - b)))
        mask = p > b + 5.0 * s
        npx = int(mask.sum())
        m = float(p[mask].mean()) if npx >= MIN_MASK_PIXELS else b
        out.append((float(z), npx, m, (m - b) / b, (m - b) / s))
    return out


def bead_metrics(stack):
    lo, hi = 16, 112
    yy, xx = np.mgrid[lo:hi, lo:hi]
    fwhm = []
    peak = []
    for k in range(DEPTHS.size):
        wimg = np.asarray(stack[k], dtype=np.float64)[lo:hi, lo:hi]
        b = float(np.median(wimg))
        s = 1.4826 * float(np.median(np.abs(wimg - b)))
        w = np.clip(wimg - (b + 3.0 * s), 0.0, None)
        tot = float(w.sum())
        x0 = float((w * xx).sum() / tot)
        y0 = float((w * yy).sum() / tot)
        sx2 = float((w * (xx - x0) ** 2).sum() / tot)
        sy2 = float((w * (yy - y0) ** 2).sum() / tot)
        fwhm.append(2.3548 * float(np.sqrt(0.5 * (sx2 + sy2))) * BEAD_UM)
        peak.append(tot / (2.0 * np.pi * float(np.sqrt(sx2 * sy2))))
    return np.array(fwhm), np.array(peak)


cells = {m: cell_metrics(tifffile.imread('/app/data/' + m + '_stack.tif')) for m in ('2p', '3p')}
beads = {m: bead_metrics(tifffile.imread('/app/data/beads_' + m + '.tif')) for m in ('2p', '3p')}

rows = []
for m in ('2p', '3p'):
    for (z, npx, sm, sbr, snr), fw in zip(cells[m], beads[m][0]):
        rows.append({'modality': m, 'depth_um': z, 'mask_pixels': npx, 'signal_mean': sm,
                     'sbr': sbr, 'snr': snr, 'fwhm_lateral_um': fw})
cols = ['modality', 'depth_um', 'mask_pixels', 'signal_mean', 'sbr', 'snr', 'fwhm_lateral_um']
pd.DataFrame(rows, columns=cols).to_csv('/app/results/metrics.csv', index=False)

sel = DEPTHS <= 120.0
mus = {}
for m in ('2p', '3p'):
    slope = np.polyfit(DEPTHS[sel], np.log(beads[m][1][sel]), 1)[0]
    mus[m] = float(-slope)

snr = {m: np.array([r[4] for r in cells[m]]) for m in ('2p', '3p')}
fw = {m: beads[m][0] for m in ('2p', '3p')}
adv = -1.0
for i in range(DEPTHS.size):
    if all(snr['3p'][j] > snr['2p'][j] and fw['3p'][j] < fw['2p'][j]
           for j in range(i, DEPTHS.size)):
        adv = float(DEPTHS[i])
        break

summary = {'mu_2p_um_inv': mus['2p'], 'mu_3p_um_inv': mus['3p'],
           'mu_ratio': mus['2p'] / mus['3p'], 'advantage_depth_um': adv,
           'fit_depth_range_um': [0.0, 120.0]}
with open('/app/results/summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))
PY
