#!/usr/bin/env python3
"""Deterministic generator for the synthetic neutrophil-NBT / TBARS-MDA cohort.

Running this script recreates every raw measurement table used by the task.
No derived ground-truth quantity is ever written to disk. The values are
synthetic but mimic the structure and scale of the underlying study: 100
subjects (21 fertile controls, 79 infertile women), ~18,000 segmented
microscopy objects and three 96-well TBARS plates.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SEED = 20260802
PIXEL_SIZE_UM = 0.25
DILUTION = 2.0
STD_CONC = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def main(out_dir):
    rng = np.random.default_rng(SEED)
    os.makedirs(out_dir, exist_ok=True)

    n_ctrl, n_case = 21, 79
    n = n_ctrl + n_case
    groups = np.array(['fertile_control'] * n_ctrl + ['infertile'] * n_case)
    groups = groups[rng.permutation(n)]
    subject_ids = ['S{:03d}'.format(i) for i in range(1, n + 1)]

    meta = pd.DataFrame({
        'subject_id': subject_ids,
        'group': groups,
        'age_years': np.round(np.clip(rng.normal(29.0, 5.0, n), 18.0, 45.0), 1),
        'specimen_type': rng.choice(['blood', 'serum'], size=n, p=[0.6, 0.4]),
    })
    meta.to_csv(os.path.join(out_dir, 'subject_metadata.csv'), index=False)

    burden = rng.normal(0.0, 1.0, n) + np.where(groups == 'infertile', 1.10, 0.0)
    p_pos = 1.0 / (1.0 + np.exp(-(-0.55 + 0.85 * burden + rng.normal(0.0, 0.40, n))))
    p_pos = np.clip(p_pos, 0.05, 0.95)
    size_factor = np.exp(rng.normal(0.0, 0.40, n))
    mda_true = np.clip(3.0 + 1.7 * burden + rng.normal(0.0, 1.3, n), 0.6, 12.0)

    cell_rows, field_rows = [], []
    for i, sid in enumerate(subject_ids):
        n_fields = int(rng.integers(3, 9))
        for f in range(1, n_fields + 1):
            fid = 'F{:02d}'.format(f)
            field_rows.append({
                'subject_id': sid, 'field_id': fid,
                'focus_score': round(float(np.clip(rng.normal(0.78, 0.11), 0.05, 0.99)), 4),
                'background_od': round(float(np.clip(rng.normal(0.055, 0.012), 0.01, 0.12)), 4),
            })
            n_cells = int(rng.integers(20, 46))
            for c in range(1, n_cells + 1):
                u = rng.random()
                if u < 0.055:
                    obj = 'lymphocyte'
                elif u < 0.095:
                    obj = 'debris'
                elif u < 0.112:
                    obj = 'platelet_aggregate'
                else:
                    obj = 'neutrophil'

                if obj == 'neutrophil':
                    if rng.random() < 0.04:
                        area_px = float(rng.uniform(50.0, 190.0))
                    else:
                        area_px = float(np.clip(rng.normal(1800.0, 300.0), 250.0, 3400.0))
                    if rng.random() < p_pos[i]:
                        fa_um2 = float(np.clip(55.0 * size_factor[i] * np.exp(rng.normal(0.0, 0.45)), 26.0, 400.0))
                        od = float(np.clip(rng.normal(0.62, 0.15), 0.25, 1.20))
                    else:
                        fa_um2 = float(np.clip(np.exp(rng.normal(np.log(6.0), 0.60)), 0.5, 24.0))
                        od = float(np.clip(rng.normal(0.07, 0.03), 0.01, 0.20))
                elif obj == 'lymphocyte':
                    area_px = float(np.clip(rng.normal(760.0, 130.0), 300.0, 1400.0))
                    fa_um2 = float(np.clip(rng.normal(3.0, 2.5), 0.0, 15.0))
                    od = float(np.clip(rng.normal(0.04, 0.02), 0.0, 0.12))
                elif obj == 'debris':
                    area_px = float(rng.uniform(30.0, 400.0))
                    fa_um2 = float(np.clip(rng.normal(8.0, 9.0), 0.0, 45.0))
                    od = float(np.clip(rng.normal(0.05, 0.04), 0.0, 0.20))
                else:
                    area_px = float(rng.uniform(40.0, 300.0))
                    fa_um2 = float(np.clip(rng.normal(2.0, 3.0), 0.0, 20.0))
                    od = float(np.clip(rng.normal(0.03, 0.02), 0.0, 0.10))

                cell_rows.append({
                    'subject_id': sid, 'field_id': fid,
                    'cell_id': 'C{:03d}'.format(c),
                    'object_class': obj,
                    'cell_area_px': round(area_px, 1),
                    'formazan_area_px': round(fa_um2 / (PIXEL_SIZE_UM ** 2), 1),
                    'mean_deposit_od': round(od, 4),
                })

    pd.DataFrame(cell_rows).to_csv(os.path.join(out_dir, 'nbt_cell_measurements.csv'), index=False)
    pd.DataFrame(field_rows).to_csv(os.path.join(out_dir, 'field_qc.csv'), index=False)

    # ---------------- TBARS plates ----------------
    rng_p = np.random.default_rng(SEED + 7)
    wells = ['{}{}'.format(r, c) for r in 'ABCDEFGH' for c in range(1, 13)]
    std_wells, sample_wells = wells[:8], wells[8:]
    std_ids = ['STD_{:g}'.format(c) for c in STD_CONC]
    plate_of = {sid: (idx % 3) + 1 for idx, sid in enumerate(subject_ids)}
    sat_subjects = rng_p.choice(subject_ids, size=5, replace=False)
    saturated = {(sid, int(rng_p.integers(1, 3))) for sid in sat_subjects}

    readings, wellmap = [], []
    for plate in (1, 2, 3):
        pid = 'plate_{}'.format(plate)
        slope = 0.065 + float(rng_p.normal(0.0, 0.0035))
        intercept = 0.045 + float(rng_p.normal(0.0, 0.006))
        for w, cid, cc in zip(std_wells, std_ids, STD_CONC):
            readings.append({'plate_id': pid, 'well': w,
                             'absorbance_532': round(intercept + slope * cc + float(rng_p.normal(0.0, 0.004)), 4),
                             'absorbance_600': round(0.035 + float(rng_p.normal(0.0, 0.004)), 4)})
            wellmap.append({'plate_id': pid, 'well': w, 'well_type': 'standard',
                            'sample_id': cid, 'mda_nmol_per_ml': cc, 'replicate': ''})
        k = 0
        for sid in [s for s in subject_ids if plate_of[s] == plate]:
            i = subject_ids.index(sid)
            for rep in (1, 2):
                w = sample_wells[k]
                k += 1
                a532 = intercept + slope * (mda_true[i] / DILUTION) + float(rng_p.normal(0.0, 0.006))
                if (sid, rep) in saturated:
                    a532 = 2.25 + float(rng_p.uniform(0.0, 0.4))
                readings.append({'plate_id': pid, 'well': w, 'absorbance_532': round(a532, 4),
                                 'absorbance_600': round(0.035 + float(rng_p.normal(0.0, 0.004)), 4)})
                wellmap.append({'plate_id': pid, 'well': w, 'well_type': 'sample',
                                'sample_id': sid, 'mda_nmol_per_ml': '', 'replicate': rep})

    pd.DataFrame(readings).to_csv(os.path.join(out_dir, 'tbars_plate_readings.csv'), index=False)
    pd.DataFrame(wellmap).to_csv(os.path.join(out_dir, 'tbars_well_map.csv'), index=False)

    with open(os.path.join(out_dir, 'assay_parameters.json'), 'w') as fh:
        json.dump({'pixel_size_um': PIXEL_SIZE_UM,
                   'tbars_dilution_factor': DILUTION,
                   'absorbance_wavelengths_nm': {'signal': 532, 'reference': 600}}, fh, indent=2)

    print('wrote synthetic cohort to {}: {} objects, {} fields, {} wells, {} saturated wells'.format(
        out_dir, len(cell_rows), len(field_rows), len(readings), len(saturated)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir', nargs='?', default=os.path.dirname(os.path.abspath(__file__)))
    main(ap.parse_args().out_dir)
