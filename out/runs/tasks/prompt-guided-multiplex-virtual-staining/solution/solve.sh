#!/bin/bash
set -euo pipefail
mkdir -p /app/output/test_predictions /app/output/valid_predictions
cat > /tmp/solve.py <<'PY'
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import disk

IMG = 160
TARGETS = ("actin", "tubulin", "mito", "golgi", "er")
DATA, OUT = "/app/data", "/app/output"
YY, XX = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
GRID = [1.00, 1.04, 1.08, 1.12, 1.16, 1.20, 1.24]


def load(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.uint8)


def save(a, p):
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(p)


def ids(split):
    return sorted({f.split("_")[0] for f in os.listdir(os.path.join(DATA, split))})


def mask_of(dna):
    m = ndi.binary_dilation(dna >= max(0.20 * float(dna.max()), 8.0), disk(16))
    return m if m.sum() >= 500 else np.ones(dna.shape, bool)


def cells_of(dna):
    m = ndi.binary_opening(dna >= max(0.45 * float(dna.max()), 25.0), disk(2))
    lab, n = ndi.label(m)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.size < 60:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        dy, dx = ys - cy, xs - cx
        cov = np.array([[(dx * dx).mean(), (dx * dy).mean()],
                        [(dx * dy).mean(), (dy * dy).mean()]])
        w, V = np.linalg.eigh(cov)
        b = 2.0 * np.sqrt(max(w[0], 1e-6))
        a = 2.0 * np.sqrt(max(w[1], 1e-6))
        out.append({"cy": cy, "cx": cx, "r": b, "e": max(a / b, 1.0),
                    "th": float(np.arctan2(V[1, 1], V[0, 1])),
                    "p95": float(np.percentile(dna[lab == i], 95))})
    if out:
        med = float(np.median([c["p95"] for c in out]))
        for c in out:
            c["amp"] = float(np.clip(c["p95"] / max(med, 1e-6), 0.80, 1.05))
    return out


def render(ch, cells, s):
    img = np.zeros((IMG, IMG), np.float32)
    for c in cells:
        ct, st = np.cos(c["th"]), np.sin(c["th"])
        dx = XX - c["cx"]
        dy = YY - c["cy"]
        u = ct * dx + st * dy
        v = -st * dx + ct * dy
        a = c["r"] * c["e"] / s
        b = c["r"] / s
        rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
        amp = c.get("amp", 1.0)
        if ch == "actin":
            ring = np.exp(-0.5 * ((rho - 1.60) / 0.28) ** 2) * (rho < 2.5)
            fib = np.zeros_like(img)
            for off in (-0.62, -0.28, 0.28, 0.62):
                fib += (np.exp(-0.5 * ((v - off * b) / (0.20 * b)) ** 2)
                        * np.exp(-0.5 * (u / (1.50 * a)) ** 2))
            img += amp * (0.95 * ring + 0.60 * fib)
        elif ch == "tubulin":
            inner = np.exp(-0.5 * ((rho - 0.95) / 0.30) ** 2)
            seg = np.zeros_like(img)
            for m in range(9):
                phi = c["th"] + 2.0 * np.pi * m / 9.0 + 0.13 * m
                d = 1.30 * b
                cy2 = c["cy"] + d * np.sin(phi)
                cx2 = c["cx"] + d * np.cos(phi)
                uu = np.cos(phi) * (XX - cx2) + np.sin(phi) * (YY - cy2)
                vv = -np.sin(phi) * (XX - cx2) + np.cos(phi) * (YY - cy2)
                seg += (np.exp(-0.5 * (uu / (0.95 * a)) ** 2)
                        * np.exp(-0.5 * (vv / (0.15 * b)) ** 2))
            img += amp * (0.55 * inner + 0.60 * seg)
        elif ch == "mito":
            env = np.exp(-0.5 * ((rho - 1.30) / 0.40) ** 2)
            pct = np.zeros_like(img)
            for i in range(-3, 4):
                for j in range(-3, 4):
                    uu = 0.55 * i + 0.275 * (j % 2)
                    vv = 0.42 * j
                    if 1.05 < np.hypot(uu, vv) < 1.85:
                        pct += np.exp(-0.5 * (((YY - (c["cy"] + vv * b)) ** 2 +
                                               (XX - (c["cx"] + uu * a)) ** 2) / 0.85 ** 2))
            img += amp * (0.90 * env + 0.32 * pct)
        elif ch == "golgi":
            side = 1.0 / (1.0 + np.exp(-v / (0.35 * b)))
            img += amp * np.exp(-0.5 * ((rho - 1.35) / 0.28) ** 2) * side
        elif ch == "er":
            env = np.exp(-0.5 * ((rho - 1.35) / 0.45) ** 2)
            mesh = 0.5 + 0.5 * np.sin(6.0 * u / a) * np.sin(5.0 * v / b)
            img += amp * 0.85 * env * (0.75 + 0.50 * mesh)
    return img


def fit_affine(raw_list, gt_list, mask_list):
    r = np.concatenate([x[m] for x, m in zip(raw_list, mask_list)])
    g = np.concatenate([x[m].astype(np.float64) for x, m in zip(gt_list, mask_list)])
    pr = np.percentile(r, [50, 95])
    pg = np.percentile(g, [50, 95])
    if pr[1] - pr[0] < 1e-6:
        return 1.0, float(pg[0])
    gain = float((pg[1] - pg[0]) / (pr[1] - pr[0]))
    return gain, float(pg[0] - gain * pr[0])


def pcc(a, b, m):
    x = a[m].astype(np.float64) - a[m].mean()
    y = b[m].astype(np.float64) - b[m].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 1e-9 else 0.0


def main():
    tr, va, te = ids("train"), ids("valid"), ids("test")
    cal = tr[:12]
    geo = {f: cells_of(load(os.path.join(DATA, "train", f + "_dna.png"))) for f in tr}
    geo.update({f: cells_of(load(os.path.join(DATA, "valid", f + "_dna.png"))) for f in va})
    geo.update({f: cells_of(load(os.path.join(DATA, "test", f + "_dna.png"))) for f in te})
    msk = {f: mask_of(load(os.path.join(DATA, "train", f + "_dna.png"))) for f in tr}
    msk.update({f: mask_of(load(os.path.join(DATA, "valid", f + "_dna.png"))) for f in va})
    msk.update({f: mask_of(load(os.path.join(DATA, "test", f + "_dna.png"))) for f in te})

    best, gain, bias = {}, {}, {}
    for ch in TARGETS:
        sel = (-2.0, 1.12)
        for s in GRID:
            raws, gts, ms = [], [], []
            for f in cal:
                raws.append(render(ch, geo[f], s))
                gts.append(load(os.path.join(DATA, "train", f + "_" + ch + ".png")))
                ms.append(msk[f])
            g, b = fit_affine(raws, gts, ms)
            sc = float(np.mean([pcc(np.clip(g * r + b, 0, 255), gt, m)
                                for r, gt, m in zip(raws, gts, ms)]))
            if sc > sel[0]:
                sel = (sc, s)
        best[ch] = sel[1]
        raws, gts, ms = [], [], []
        for f in cal:
            raws.append(render(ch, geo[f], best[ch]))
            gts.append(load(os.path.join(DATA, "train", f + "_" + ch + ".png")))
            ms.append(msk[f])
        gain[ch], bias[ch] = fit_affine(raws, gts, ms)
        print(ch, "scale=%.2f pcc=%.3f" % (best[ch], sel[0]))

    def emit(fields, folder):
        for f in fields:
            for ch in TARGETS:
                raw = render(ch, geo[f], best[ch])
                save(gain[ch] * raw + bias[ch], os.path.join(folder, "%s_%s.png" % (f, ch)))

    emit(te, os.path.join(OUT, "test_predictions"))
    emit(va, os.path.join(OUT, "valid_predictions"))

    rep = {"method": "prompt-driven elliptical renderer; radius bias calibrated on train",
           "valid_pcc": {}, "valid_pcc_mean": 0.0}
    vals = []
    for ch in TARGETS:
        cs = [pcc(load(os.path.join(OUT, "valid_predictions", "%s_%s.png" % (f, ch))),
                  load(os.path.join(DATA, "valid", "%s_%s.png" % (f, ch))), msk[f]) for f in va]
        rep["valid_pcc"][ch] = round(float(np.mean(cs)), 4)
        vals.append(float(np.mean(cs)))
    rep["valid_pcc_mean"] = round(float(np.mean(vals)), 4)
    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep, indent=2))


main()
PY
python /tmp/solve.py
