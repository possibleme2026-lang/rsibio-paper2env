#!/usr/bin/env python3
"""Deterministic phantom generator for the 2P/3P cortical-bone imaging task.

Renders two osteocyte z-stacks (512x512x24) and two PSF bead z-stacks
(128x128x24) as 16-bit multi-page TIFFs.  The seed is fixed and the renderer is
byte-identical to the copy inside tests/gold_gen.py, so the hidden gold can be
regenerated exactly at verification time.  This file is deleted from the image
after the build step that creates the data.
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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cells, beads = render_all(SEED)
    for modality in ('2p', '3p'):
        tifffile.imwrite(os.path.join(here, modality + '_stack.tif'), cells[modality],
                         photometric='minisblack')
        tifffile.imwrite(os.path.join(here, 'beads_' + modality + '.tif'), beads[modality],
                         photometric='minisblack')
    print('phantom written to', here)


if __name__ == '__main__':
    main()
