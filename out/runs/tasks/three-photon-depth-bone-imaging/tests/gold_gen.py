#!/usr/bin/env python3
"""Hidden gold generator for the 2P/3P cortical-bone imaging task.

Contains a byte-identical copy of the phantom renderer that produced the TIFFs
in /app/data (same seed, same call order) plus an independent reference
implementation of every metric the agent has to report.  Nothing in this file
is visible to the agent during the solve phase.
"""
import os

import numpy as np
import tifffile

SEED = 20260731
DEPTHS = np.arange(0.0, 231.0, 10.0)

CELL_N = 512
CELL_UM = 0.2
BEAD_N = 128
BEAD_UM = 0.1

CELL_COUNT = 90
Z_SIGMA = 8.0
NOISE_CELL = 25.0

BEAD_BG = 300.0
NOISE_BEAD = 2.0

MIN_MASK_PIXELS = 20
BEAD_WINDOW = (16, 112)
FIT_ZMAX = 120.0


def cell_background(z, modality):
    if modality == '2p':
        return 240.0 + 1.2 * z
    return 200.0 + 0.6 * z


def psf_fwhm_um(z, modality):
    if modality == '2p':
        return 0.42 * np.sqrt(1.0 + (z / 60.0) ** 2)
    return 0.46 * np.sqrt(1.0 + (z / 260.0) ** 2)


def psf_sigma_um(z, modality):
    return psf_fwhm_um(z, modality) / 2.3548


def cell_peak(z, modality):
    if modality == '2p':
        return 6000.0 * np.exp(-z / 40.0)
    return 2600.0 * np.exp(-z / 190.0)


def bead_peak(z, modality):
    if modality == '2p':
        return 40000.0 * np.exp(-z / 40.0)
    return 16000.0 * np.exp(-z / 190.0)


def add_gaussian(img, cx, cy, sx, sy, theta, amp):
    r = int(np.ceil(4.0 * max(sx, sy))) + 1
    x0 = max(0, int(np.floor(cx)) - r)
    x1 = min(img.shape[1], int(np.ceil(cx)) + r + 1)
    y0 = max(0, int(np.floor(cy)) - r)
    y1 = min(img.shape[0], int(np.ceil(cy)) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - cx
    dy = yy - cy
    ct = np.cos(theta)
    st = np.sin(theta)
    u = ct * dx + st * dy
    v = -st * dx + ct * dy
    img[y0:y1, x0:x1] += amp * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))


def make_population(rng):
    n = CELL_COUNT
    return {
        'x': rng.uniform(6.0, CELL_N * CELL_UM - 6.0, n),
        'y': rng.uniform(6.0, CELL_N * CELL_UM - 6.0, n),
        'z': rng.uniform(-25.0, 265.0, n),
        'theta': rng.uniform(0.0, np.pi, n),
        'smaj': 3.0 * rng.uniform(0.75, 1.35, n),
        'smin': 1.5 * rng.uniform(0.75, 1.35, n),
    }


def render_cell_plane(pop, z, modality, rng):
    img = np.zeros((CELL_N, CELL_N), dtype=np.float64)
    amp0 = cell_peak(z, modality)
    s_psf = psf_sigma_um(z, modality)
    for i in range(pop['x'].size):
        w = np.exp(-0.5 * ((pop['z'][i] - z) / Z_SIGMA) ** 2)
        if w < 0.03:
            continue
        smaj = np.sqrt(pop['smaj'][i] ** 2 + s_psf ** 2)
        smin = np.sqrt(pop['smin'][i] ** 2 + s_psf ** 2)
        add_gaussian(img, pop['x'][i] / CELL_UM, pop['y'][i] / CELL_UM,
                     smaj / CELL_UM, smin / CELL_UM, pop['theta'][i], amp0 * w)
    img += cell_background(z, modality)
    img += rng.normal(0.0, NOISE_CELL, img.shape)
    return np.clip(img, 0.0, 65535.0).astype(np.uint16)


def render_bead_plane(z, modality, rng):
    img = np.zeros((BEAD_N, BEAD_N), dtype=np.float64)
    s = psf_sigma_um(z, modality) / BEAD_UM
    add_gaussian(img, (BEAD_N - 1) / 2.0, (BEAD_N - 1) / 2.0, s, s, 0.0,
                 bead_peak(z, modality))
    img += BEAD_BG
    img += rng.normal(0.0, NOISE_BEAD, img.shape)
    return np.clip(img, 0.0, 65535.0).astype(np.uint16)


def render_all(seed=SEED):
    rng = np.random.default_rng(seed)
    pop = make_population(rng)
    cells = {}
    for modality in ('2p', '3p'):
        cells[modality] = np.stack([render_cell_plane(pop, z, modality, rng) for z in DEPTHS])
    beads = {}
    for modality in ('2p', '3p'):
        beads[modality] = np.stack([render_bead_plane(z, modality, rng) for z in DEPTHS])
    return cells, beads


def analyse_cell_stack(stack):
    out = []
    for k, z in enumerate(DEPTHS):
        plane = np.asarray(stack[k], dtype=np.float64)
        med = float(np.median(plane))
        mad = float(np.median(np.abs(plane - med)))
        sigma = 1.4826 * mad
        mask = plane > (med + 5.0 * sigma)
        npx = int(mask.sum())
        if npx < MIN_MASK_PIXELS:
            sig_mean = med
        else:
            sig_mean = float(plane[mask].mean())
        snr = (sig_mean - med) / sigma if sigma > 0 else 0.0
        sbr = (sig_mean - med) / med if med > 0 else 0.0
        out.append({'depth_um': float(z), 'mask_pixels': npx, 'signal_mean': sig_mean,
                    'sbr': float(sbr), 'snr': float(snr)})
    return out


def analyse_bead_stack(stack):
    lo, hi = BEAD_WINDOW
    yy, xx = np.mgrid[lo:hi, lo:hi]
    out = []
    for k, z in enumerate(DEPTHS):
        plane = np.asarray(stack[k], dtype=np.float64)
        win = plane[lo:hi, lo:hi]
        med = float(np.median(win))
        sigma = 1.4826 * float(np.median(np.abs(win - med)))
        w = np.clip(win - (med + 3.0 * sigma), 0.0, None)
        tot = float(w.sum())
        if tot <= 0.0:
            out.append({'depth_um': float(z), 'fwhm_lateral_um': -1.0, 'peak': 0.0})
            continue
        x0 = float((w * xx).sum() / tot)
        y0 = float((w * yy).sum() / tot)
        sx2 = float((w * (xx - x0) ** 2).sum() / tot)
        sy2 = float((w * (yy - y0) ** 2).sum() / tot)
        fwhm = 2.3548 * float(np.sqrt(0.5 * (sx2 + sy2))) * BEAD_UM
        peak = tot / (2.0 * np.pi * float(np.sqrt(sx2 * sy2)))
        out.append({'depth_um': float(z), 'fwhm_lateral_um': float(fwhm), 'peak': float(peak)})
    return out


def fit_mu(peaks):
    sel = [i for i, z in enumerate(DEPTHS) if z <= FIT_ZMAX + 1e-9 and peaks[i] > 0.0]
    x = np.array([DEPTHS[i] for i in sel], dtype=np.float64)
    y = np.log(np.array([peaks[i] for i in sel], dtype=np.float64))
    slope = float(np.polyfit(x, y, 1)[0])
    return -slope


def advantage_depth(snr2, snr3, fw2, fw3):
    n = len(DEPTHS)
    ok = [bool(snr3[i] > snr2[i] and 0.0 < fw3[i] < fw2[i]) for i in range(n)]
    for i in range(n):
        if all(ok[i:]):
            return float(DEPTHS[i])
    return -1.0


def compute_gold():
    cells, beads = render_all(SEED)
    cm = {m: analyse_cell_stack(cells[m]) for m in ('2p', '3p')}
    bm = {m: analyse_bead_stack(beads[m]) for m in ('2p', '3p')}
    rows = []
    for m in ('2p', '3p'):
        for i, z in enumerate(DEPTHS):
            rows.append({'modality': m, 'depth_um': float(z),
                         'mask_pixels': cm[m][i]['mask_pixels'],
                         'signal_mean': cm[m][i]['signal_mean'],
                         'sbr': cm[m][i]['sbr'],
                         'snr': cm[m][i]['snr'],
                         'fwhm_lateral_um': bm[m][i]['fwhm_lateral_um']})
    mu2 = fit_mu([d['peak'] for d in bm['2p']])
    mu3 = fit_mu([d['peak'] for d in bm['3p']])
    adv = advantage_depth([d['snr'] for d in cm['2p']], [d['snr'] for d in cm['3p']],
                          [d['fwhm_lateral_um'] for d in bm['2p']],
                          [d['fwhm_lateral_um'] for d in bm['3p']])
    return {'rows': rows, 'mu_2p': mu2, 'mu_3p': mu3,
            'mu_ratio': (mu2 / mu3 if mu3 != 0 else float('nan')),
            'advantage_depth_um': adv}


def verify_inputs(data_dir):
    cells, beads = render_all(SEED)
    expected = {
        '2p_stack.tif': cells['2p'],
        '3p_stack.tif': cells['3p'],
        'beads_2p.tif': beads['2p'],
        'beads_3p.tif': beads['3p'],
    }
    for name, arr in expected.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            return False, 'input file missing: ' + name
        try:
            got = tifffile.imread(path)
        except Exception as exc:
            return False, 'cannot read %s: %s' % (name, exc)
        if got.shape != arr.shape or not np.array_equal(got, arr):
            return False, 'input file modified: ' + name
    return True, ''
