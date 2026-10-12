#!/usr/bin/env python3
"""Deterministic multiplex-fluorescence simulator.

Writes /app/data/{train,valid,test}.  The test fields are written with the DNA
channel only: the agent must synthesise the remaining five structures.  This
file is deleted from the image right after the data have been generated.
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

IMG = 160
CHANNELS = ("dna", "actin", "tubulin", "mito", "golgi", "er")
TARGETS = ("actin", "tubulin", "mito", "golgi", "er")
BG = {"dna": 6.0, "actin": 10.0, "tubulin": 9.0, "mito": 8.0,
      "golgi": 7.0, "er": 12.0}
TRAIN_SEEDS = [1000 + i for i in range(40)]
VALID_SEEDS = [2000 + i for i in range(16)]
TEST_SEEDS = [3000 + i for i in range(24)]
PUBLIC_TARGET_OFFSET = 1000003

_YY, _XX = np.mgrid[0:IMG, 0:IMG].astype(np.float32)


def field_name(split, i):
    return {"train": "R%03d", "valid": "V%03d", "test": "T%03d"}[split] % (i + 1)


def _cells(rng):
    cells = []
    for gy in range(3):
        for gx in range(3):
            cells.append({
                "cy": 28.0 + 52.0 * gy + float(rng.uniform(-7.0, 7.0)),
                "cx": 28.0 + 52.0 * gx + float(rng.uniform(-7.0, 7.0)),
                "r": float(rng.uniform(6.0, 9.0)),
                "e": float(rng.uniform(1.15, 1.65)),
                "th": float(rng.uniform(0.0, np.pi)),
                "amp": float(rng.uniform(0.80, 1.00)),
            })
    return cells


def _uv(c):
    ct, st = np.cos(c["th"]), np.sin(c["th"])
    dx = _XX - c["cx"]
    dy = _YY - c["cy"]
    u = ct * dx + st * dy
    v = -st * dx + ct * dy
    a = c["r"] * c["e"]
    b = c["r"]
    rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    return u, v, rho, a, b


def _gauss(cy, cx, s):
    return np.exp(-0.5 * (((_YY - cy) ** 2 + (_XX - cx) ** 2) / (s * s)))


def _seg(cy, cx, phi, L, w):
    dx = _XX - cx
    dy = _YY - cy
    u = np.cos(phi) * dx + np.sin(phi) * dy
    v = -np.sin(phi) * dx + np.cos(phi) * dy
    return np.exp(-0.5 * (u / L) ** 2) * np.exp(-0.5 * (v / w) ** 2)


def raw_dna(cells):
    img = np.zeros((IMG, IMG), np.float32)
    for c in cells:
        _, _, rho, _, _ = _uv(c)
        img += c["amp"] * np.exp(-0.5 * rho ** 4)
    return img


def raw_actin(cells):
    img = np.zeros((IMG, IMG), np.float32)
    for c in cells:
        u, v, rho, a, b = _uv(c)
        ring = np.exp(-0.5 * ((rho - 1.60) / 0.28) ** 2) * (rho < 2.5)
        fib = np.zeros_like(img)
        for off in (-0.62, -0.28, 0.28, 0.62):
            fib += (np.exp(-0.5 * ((v - off * b) / (0.20 * b)) ** 2)
                    * np.exp(-0.5 * (u / (1.50 * a)) ** 2))
        img += c["amp"] * (0.95 * ring + 0.60 * fib)
    return img


def raw_tubulin(cells):
    img = np.zeros((IMG, IMG), np.float32)
    for c in cells:
        u, v, rho, a, b = _uv(c)
        inner = np.exp(-0.5 * ((rho - 0.95) / 0.30) ** 2)
        seg = np.zeros_like(img)
        for m in range(9):
            phi = c["th"] + 2.0 * np.pi * m / 9.0 + 0.13 * m
            d = 1.30 * b
            seg += _seg(c["cy"] + d * np.sin(phi), c["cx"] + d * np.cos(phi),
                        phi, 0.95 * a, 0.15 * b)
        img += c["amp"] * (0.55 * inner + 0.60 * seg)
    return img


def raw_mito(cells):
    img = np.zeros((IMG, IMG), np.float32)
    for c in cells:
        u, v, rho, a, b = _uv(c)
        env = np.exp(-0.5 * ((rho - 1.30) / 0.40) ** 2)
        pct = np.zeros_like(img)
        for i in range(-3, 4):
            for j in range(-3, 4):
                uu = 0.55 * i + 0.275 * (j % 2)
                vv = 0.42 * j
                if 1.05 < np.hypot(uu, vv) < 1.85:
                    pct += _gauss(c["cy"] + vv * b, c["cx"] + uu * a, 0.85)
        img += c["amp"] * (0.90 * env + 0.32 * pct)
    return img


def raw_golgi(cells):
    img = np.zeros((IMG, IMG), np.float32)
    for c in cells:
        u, v, rho, a, b = _uv(c)
        side = 1.0 / (1.0 + np.exp(-v / (0.35 * b)))
        img += c["amp"] * np.exp(-0.5 * ((rho - 1.35) / 0.28) ** 2) * side
    return img


def raw_er(cells):
    img = np.zeros((IMG, IMG), np.float32)
    for c in cells:
        u, v, rho, a, b = _uv(c)
        env = np.exp(-0.5 * ((rho - 1.35) / 0.45) ** 2)
        mesh = 0.5 + 0.5 * np.sin(6.0 * u / a) * np.sin(5.0 * v / b)
        img += c["amp"] * 0.85 * env * (0.75 + 0.50 * mesh)
    return img


RAW = {"dna": raw_dna, "actin": raw_actin, "tubulin": raw_tubulin,
       "mito": raw_mito, "golgi": raw_golgi, "er": raw_er}


def compute_scales():
    pool = {ch: [] for ch in CHANNELS}
    for s in TRAIN_SEEDS:
        cells = _cells(np.random.default_rng(s))
        for ch in CHANNELS:
            pool[ch].append(RAW[ch](cells).ravel())
    return {ch: 225.0 / max(float(np.percentile(np.concatenate(pool[ch]), 99.5)), 1e-6)
            for ch in CHANNELS}


def _finish(raw, ch, scales, rng, illum):
    v = raw * scales[ch] * illum + BG[ch]
    v = ndi.gaussian_filter(v.astype(np.float32), 0.55)
    v = v + rng.normal(0.0, 2.2, v.shape).astype(np.float32)
    return np.clip(v, 0, 255).astype(np.uint8)


def render_field(seed, scales, targets_rng=None):
    rng = np.random.default_rng(seed)
    cells = _cells(rng)
    illum = float(rng.uniform(0.90, 1.10))
    out = {"dna": _finish(RAW["dna"](cells), "dna", scales, rng, illum)}
    if targets_rng is not None:
        for ch in TARGETS:
            out[ch] = _finish(RAW[ch](cells), ch, scales, targets_rng, illum)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/app/data")
    a = ap.parse_args()
    scales = compute_scales()
    for split, seeds in (("train", TRAIN_SEEDS), ("valid", VALID_SEEDS),
                         ("test", TEST_SEEDS)):
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        for i, s in enumerate(seeds):
            name = field_name(split, i)
            if split == "test":
                img = render_field(s, scales, None)
                Image.fromarray(img["dna"]).save(os.path.join(d, name + "_dna.png"))
            else:
                img = render_field(s, scales,
                                   np.random.default_rng(s + PUBLIC_TARGET_OFFSET))
                for ch in CHANNELS:
                    Image.fromarray(img[ch]).save(
                        os.path.join(d, "%s_%s.png" % (name, ch)))
    print("data written to", a.out)


if __name__ == "__main__":
    main()
