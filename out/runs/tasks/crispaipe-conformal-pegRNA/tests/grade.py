#!/usr/bin/env python3
import json
import math
import os
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_gen import get_gold

OUT_DIR = '/app/output'
NOMINAL = 0.90
N_CONSTRAINTS = 9

def load_predictions():
    path = os.path.join(OUT_DIR, 'predictions.csv')
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def load_regions():
    path = os.path.join(OUT_DIR, 'regions.jsonl')
    if not os.path.exists(path):
        return None
    regions = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                return None
            pid = obj.get('peg_id')
            reg = obj.get('region')
            if pid is None or reg is None:
                return None
            regions[pid] = reg
    return regions

def load_calibration():
    path = os.path.join(OUT_DIR, 'calibration.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None

def region_area_and_hull(vertices):
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3:
        return 0.0, None
    if np.any(v < -1e-9):
        return 0.0, None
    s = v.sum(axis=1)
    if np.any(np.abs(s - 1.0) > 1e-6):
        return 0.0, None
    pts2 = v[:, :2]
    try:
        hull = ConvexHull(pts2)
    except Exception:
        return 0.0, None
    return float(hull.area), hull

def point_in_hull(pt2, hull, tol=1e-7):
    if hull is None:
        return False
    return bool(np.all(hull.equations[:, :2] @ pt2 + hull.equations[:, 2] <= tol))

def main():
    results = {}
    gold = get_gold().set_index('peg_id')
    gold_ids = list(gold.index)
    gold_y = gold[['true_edited', 'true_unedited', 'true_indel']].values.astype(float)

    preds = load_predictions()
    regions = load_regions()
    calib = load_calibration()

    c1 = False
    if preds is not None and set(preds.columns) >= {'peg_id', 'p_edited', 'p_unedited', 'p_indel'}:
        if len(preds) == len(gold_ids) and set(preds['peg_id']) == set(gold_ids) and preds['peg_id'].is_unique:
            c1 = True
    results['C1_predictions_format'] = c1

    c2 = False
    P = None
    if c1:
        P = preds.set_index('peg_id').loc[gold_ids][['p_edited', 'p_unedited', 'p_indel']].values.astype(float)
        if np.all(np.isfinite(P)) and np.all(P >= -1e-9) and np.all(np.abs(P.sum(axis=1) - 1.0) <= 1e-3):
            c2 = True
    results['C2_probability_validity'] = c2

    c3 = False
    c4 = False
    c5 = False
    if c2:
        try:
            rho_edit = spearmanr(P[:, 0], gold_y[:, 0]).correlation
            rho_unedit = spearmanr(P[:, 1], gold_y[:, 1]).correlation
            rho_indel = spearmanr(P[:, 2], gold_y[:, 2]).correlation
            c3 = bool(np.isfinite(rho_edit) and rho_edit >= 0.50)
            c4 = bool(np.isfinite(rho_unedit) and rho_unedit >= 0.40 and np.isfinite(rho_indel) and rho_indel >= 0.15)
            mae_edit = float(np.mean(np.abs(P[:, 0] - gold_y[:, 0])))
            mae_indel = float(np.mean(np.abs(P[:, 2] - gold_y[:, 2])))
            c5 = bool(mae_edit <= 0.15 and mae_indel <= 0.12)
        except Exception:
            pass
    results['C3_spearman_edited'] = c3
    results['C4_spearman_unedited_indel'] = c4
    results['C5_mae'] = c5

    c6 = False
    hulls = {}
    areas = []
    if regions is not None and set(regions.keys()) == set(gold_ids):
        valid = True
        for pid in gold_ids:
            try:
                area, hull = region_area_and_hull(regions[pid])
            except Exception:
                valid = False
                break
            if hull is None or area <= 1e-12:
                valid = False
                break
            hulls[pid] = hull
            areas.append(area)
        if valid and len(areas) == len(gold_ids):
            c6 = True
    results['C6_region_geometry'] = c6

    c7 = False
    coverage = None
    if c6:
        covered = 0
        for pid, y in zip(gold_ids, gold_y):
            hull = hulls[pid]
            pt2 = np.array([y[0], y[1]], dtype=float)
            if point_in_hull(pt2, hull):
                covered += 1
        coverage = covered / float(len(gold_ids))
        c7 = bool(0.80 <= coverage <= 0.98)
    results['C7_conformal_coverage'] = c7

    c8 = False
    if c6 and len(areas) > 0:
        arr = np.asarray(areas, dtype=float)
        med_area = float(np.median(arr))
        frac_small = float(np.mean(arr <= 0.40))
        c8 = bool(med_area <= 0.25 and frac_small >= 0.80)
    results['C8_region_efficiency'] = c8

    c9 = False
    if isinstance(calib, dict):
        try:
            nc = float(calib.get('nominal_coverage', -1))
            cs = int(calib.get('calib_size', 0))
            qh = float(calib.get('q_hat', float('nan')))
            c9 = bool(abs(nc - NOMINAL) <= 0.01 and cs >= 250 and np.isfinite(qh) and 0.0 < qh < 1e6)
        except Exception:
            pass
    results['C9_calibration_metadata'] = c9

    passed = sum(1 for v in results.values() if v)
    score = passed / float(N_CONSTRAINTS)
    reward = score

    print('Constraint results:')
    for k, v in results.items():
        print('  ' + k + ': ' + ('PASS' if v else 'FAIL'))
    print('Passed ' + str(passed) + '/' + str(N_CONSTRAINTS) + ' -> score=' + str(round(score, 4)))

    os.makedirs('/logs/verifier', exist_ok=True)
    with open('/logs/verifier/reward.json', 'w') as fh:
        json.dump({'score': score, 'reward': reward}, fh)
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        os.makedirs('/logs/verifier', exist_ok=True)
        with open('/logs/verifier/reward.json', 'w') as fh:
            json.dump({'score': 0.0, 'reward': 0.0}, fh)
        sys.exit(0)
