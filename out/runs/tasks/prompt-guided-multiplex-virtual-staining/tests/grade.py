#!/usr/bin/env python3
"""Deterministic grader for bioforge/prompt-guided-multiplex-virtual-staining."""
import json
import os
import sys
import traceback

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import disk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gold_gen

IMG = gold_gen.IMG
TARGETS = gold_gen.TARGETS
DATA = "/app/data"
OUT = "/app/output"
REWARD = "/logs/verifier/reward.json"
NAMES = ["format", "nondegenerate", "pcc_avg", "pcc_per_channel",
         "beats_baselines", "intensity_match", "profile_match", "self_report"]


def load_u8(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def eval_mask(dna):
    thr = max(0.20 * float(dna.max()), 8.0)
    m = ndi.binary_dilation(dna >= thr, disk(16))
    if m.sum() < 500:
        m = np.ones(dna.shape, bool)
    return m


def pcc(a, b, m):
    x = a[m].astype(np.float64)
    y = b[m].astype(np.float64)
    if x.size < 20:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if den <= 1e-9:
        return 0.0
    return float((x * y).sum() / den)


def load_preds(folder, fields):
    out = {}
    ok = True
    for f in fields:
        for ch in TARGETS:
            p = os.path.join(folder, "%s_%s.png" % (f, ch))
            if not os.path.isfile(p):
                ok = False
                continue
            try:
                a = load_u8(p)
            except Exception:
                ok = False
                continue
            if a.shape != (IMG, IMG):
                ok = False
                continue
            out[(f, ch)] = a
    return out, ok


def radial_profile(cells, img):
    best = None
    for c in cells:
        _, _, rho, _, _ = gold_gen._uv(c)
        best = rho if best is None else np.minimum(best, rho)
    edges = np.arange(0.0, 2.61, 0.2)
    idx = np.digitize(best.ravel(), edges) - 1
    vals = img.ravel().astype(np.float64)
    prof = np.full(len(edges) - 1, np.nan)
    for b in range(len(prof)):
        sel = idx == b
        if sel.sum() >= 30:
            prof[b] = vals[sel].mean()
    return prof


def prof_corr(p, g):
    m = np.isfinite(p) & np.isfinite(g)
    if m.sum() < 5:
        return 0.0
    a = p[m] - p[m].mean()
    b = g[m] - g[m].mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if den <= 1e-9:
        return 0.0
    return float((a * b).sum() / den)


def run_checks():
    res = {k: False for k in NAMES}
    info = {}

    gold = gold_gen.build()
    test_fields = gold["test_fields"]
    valid_fields = gold["valid_fields"]

    tp, tp_ok = load_preds(os.path.join(OUT, "test_predictions"), test_fields)
    vp, vp_ok = load_preds(os.path.join(OUT, "valid_predictions"), valid_fields)

    # ---------------- 1. format / input integrity ------------------------
    integrity = 0.0
    for f in test_fields:
        env = os.path.join(DATA, "test", f + "_dna.png")
        if not os.path.isfile(env):
            integrity = 255.0
            break
        a = load_u8(env).astype(np.float64)
        b = gold["test_dna"][f].astype(np.float64)
        integrity = max(integrity, float(np.abs(a - b).mean()))
    res["format"] = bool(tp_ok and vp_ok and integrity < 5.0)
    info["format"] = {"test_files": len(tp), "valid_files": len(vp),
                      "max_mean_abs_dna_diff": round(integrity, 3)}

    # ---------------- 2. non-degenerate ----------------------------------
    nd_ok = True
    nd_bad = 0
    dna_cache = {}
    for (f, ch), pred in tp.items():
        if f not in dna_cache:
            dna_cache[f] = gold["test_dna"][f]
        dna = dna_cache[f]
        m = gold["masks"][f]
        if float(pred[m].std()) < 6.0:
            nd_bad += 1
            continue
        if float(np.abs(pred.astype(np.float64) - dna.astype(np.float64))[m].mean()) <= 8.0:
            nd_bad += 1
    nd_ok = (nd_bad == 0) and len(tp) == len(test_fields) * len(TARGETS)
    res["nondegenerate"] = bool(nd_ok)
    info["nondegenerate"] = {"violations": nd_bad}

    # ---------------- 3/4/5. correlation ---------------------------------
    per_field = {ch: [] for ch in TARGETS}
    dna_base, mean_base = [], []
    for f in test_fields:
        m = gold["masks"][f]
        dna = gold["test_dna"][f]
        for ch in TARGETS:
            gt = gold["test_gt"][f][ch]
            pred = tp.get((f, ch))
            per_field[ch].append(-1.0 if pred is None else pcc(pred, gt, m))
            dna_base.append(pcc(dna, gt, m))
            mean_base.append(pcc(gold["train_mean"][ch], gt, m))

    per_ch = {ch: float(np.mean(per_field[ch])) for ch in TARGETS}
    avg = float(np.mean([per_ch[ch] for ch in TARGETS]))
    base = max(float(np.mean(dna_base)), float(np.mean(mean_base)))

    res["pcc_avg"] = bool(avg >= 0.70)
    info["pcc_avg"] = round(avg, 4)

    res["pcc_per_channel"] = bool(all(per_ch[ch] >= 0.50 for ch in TARGETS))
    info["pcc_per_channel"] = {k: round(v, 4) for k, v in per_ch.items()}

    res["beats_baselines"] = bool(avg >= base + 0.25)
    info["beats_baselines"] = {"agent": round(avg, 4), "best_baseline": round(base, 4),
                               "dna_copy": round(float(np.mean(dna_base)), 4),
                               "train_mean": round(float(np.mean(mean_base)), 4)}

    # ---------------- 6. intensity ---------------------------------------
    inten_ok = True
    inten = {}
    for ch in TARGETS:
        p50p, p95p, p50g, p95g = [], [], [], []
        for f in test_fields:
            m = gold["masks"][f]
            gt = gold["test_gt"][f][ch]
            pred = tp.get((f, ch))
            if pred is None:
                inten_ok = False
                continue
            p50p.append(np.percentile(pred[m], 50))
            p95p.append(np.percentile(pred[m], 95))
            p50g.append(np.percentile(gt[m], 50))
            p95g.append(np.percentile(gt[m], 95))
        a50, a95 = float(np.mean(p50p)), float(np.mean(p95p))
        b50, b95 = float(np.mean(p50g)), float(np.mean(p95g))
        r50 = abs(np.log((a50 + 1e-6) / (b50 + 1e-6)))
        r95 = abs(np.log((a95 + 1e-6) / (b95 + 1e-6)))
        inten[ch] = [round(a50, 1), round(b50, 1), round(a95, 1), round(b95, 1)]
        if r50 > np.log(1.30) or r95 > np.log(1.30):
            inten_ok = False
    res["intensity_match"] = bool(inten_ok)
    info["intensity_match"] = inten

    # ---------------- 7. radial profile ----------------------------------
    prof_ok = True
    profs = {}
    for ch in TARGETS:
        cs = []
        for f in test_fields:
            pred = tp.get((f, ch))
            if pred is None:
                prof_ok = False
                continue
            cells = gold["cells"][f]
            cs.append(prof_corr(radial_profile(cells, pred),
                                radial_profile(cells, gold["test_gt"][f][ch])))
        profs[ch] = round(float(np.mean(cs)), 4) if cs else 0.0
        if not cs or float(np.mean(cs)) < 0.75:
            prof_ok = False
    res["profile_match"] = bool(prof_ok)
    info["profile_match"] = profs

    # ---------------- 8. self report -------------------------------------
    vpcc = {}
    for ch in TARGETS:
        vals = []
        for f in valid_fields:
            pred = vp.get((f, ch))
            m = gold["valid_masks"][f]
            gt = gold["valid_gt"][f][ch]
            vals.append(-1.0 if pred is None else pcc(pred, gt, m))
        vpcc[ch] = float(np.mean(vals))
    vmean = float(np.mean(list(vpcc.values())))
    report_ok = False
    rep_note = {}
    if not vp_ok or len(vp) != len(valid_fields) * len(TARGETS):
        rep_note["error"] = "incomplete validation predictions"
    elif vmean < 0.65:
        rep_note["error"] = "validation mean PCC below 0.65"
        rep_note["valid_mean"] = round(vmean, 4)
    else:
        try:
            with open(os.path.join(OUT, "report.json")) as fh:
                rep = json.load(fh)
            rep_v = rep.get("valid_pcc", {})
            deltas = {}
            ok = True
            for ch in TARGETS:
                d = abs(float(rep_v.get(ch, -9.0)) - vpcc[ch])
                deltas[ch] = round(d, 4)
                if d > 0.02:
                    ok = False
            rep_note = {"valid_mean": round(vmean, 4), "deltas": deltas}
            report_ok = ok
        except Exception as exc:
            rep_note = {"error": "report.json unreadable: %s" % exc}
    res["self_report"] = bool(report_ok)
    info["self_report"] = rep_note
    return res, info


def main():
    checks = {k: False for k in NAMES}
    info = {}
    try:
        checks, info = run_checks()
    except Exception:
        traceback.print_exc()
    npass = sum(1 for k in NAMES if checks.get(k))
    score = npass / float(len(NAMES))
    payload = {"score": round(score, 6), "reward": round(score, 6),
               "checks": checks, "detail": info, "passed": npass, "total": len(NAMES)}
    os.makedirs(os.path.dirname(REWARD), exist_ok=True)
    with open(REWARD, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
