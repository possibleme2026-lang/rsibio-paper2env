#!/usr/bin/env python3
'''Deterministic generator for the synthetic HNSCC chemoradiation cohort.

Running this file recreates mouse_cohort.csv and pet_ct_roi.csv next to it from
seed 20260803.  Only the INPUTS are produced here; every analysis result used
for grading is recomputed independently at verification time.
'''

import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260803
MODELS = ('MOC1', 'MOC2', 'MLM1')
N_BY_MODEL = {'MOC1': 40, 'MOC2': 41, 'MLM1': 40}
TIME_DAYS = (0, 3, 7, 11, 14)
MODEL_LOG_HAZARD = {'MOC1': 0.0, 'MOC2': 0.45, 'MLM1': -0.30}
OUT_DIR = Path(__file__).resolve().parent


def main():
    rng = np.random.default_rng(SEED)
    cohort_rows, pet_rows = [], []
    for model in MODELS:
        for idx in range(N_BY_MODEL[model]):
            mouse_id = '{}-{:02d}'.format(model, idx + 1)

            v0 = float(np.clip(rng.lognormal(4.15, 0.22), 50.0, 150.0))
            u = float(rng.uniform())
            shrink = 0.02 + 0.13 * u + float(rng.normal(0.0, 0.006))
            grow = 0.085 - 0.075 * u + float(rng.normal(0.0, 0.008))
            if model == 'MOC2':
                shrink -= 0.012
                grow += 0.022
            elif model == 'MLM1':
                shrink += 0.015
                grow -= 0.016

            vol = {0: v0}
            vol[3] = vol[0] * math.exp(-3.0 * shrink)
            vol[7] = vol[3] * math.exp(4.0 * grow)
            vol[11] = vol[3] * math.exp(8.0 * grow)
            vol[14] = vol[3] * math.exp(11.0 * grow)
            vol = {d: float(x * math.exp(float(rng.normal(0.0, 0.03))))
                   for d, x in vol.items()}

            roi_volume = float(vol[14] * math.exp(float(rng.normal(0.0, 0.05))))
            log_roi = math.log(roi_volume)

            body_weight = float(np.clip(rng.normal(27.0, 2.5), 20.0, 35.0))
            dose = float(np.clip(rng.normal(9.5, 1.0), 6.0, 13.0))
            uptake = float(np.clip(rng.normal(60.0, 4.0), 45.0, 75.0))

            suv_max_true = float(math.exp(1.0
                                          + 0.45 * (log_roi - math.log(160.0))
                                          + 0.22 * float(rng.normal())
                                          + float(rng.normal(0.0, 0.06))))
            suv_mean_true = float(np.clip(0.55 * suv_max_true
                                          + float(rng.normal(0.0, 0.35)),
                                          0.35, 0.95 * suv_max_true))

            conc_max = suv_max_true * dose * 1000.0 / body_weight
            conc_mean = suv_mean_true * dose * 1000.0 / body_weight

            lp = (0.70 * (log_roi - math.log(160.0))
                  + 0.35 * (suv_max_true - 2.7)
                  + MODEL_LOG_HAZARD[model]
                  + float(rng.normal(0.0, 0.20)))
            scale = 38.0 * math.exp(-lp / 1.5)
            t_event = float(rng.weibull(1.5)) * scale
            t_censor = float(rng.uniform(60.0, 100.0))
            observed = min(t_event, t_censor)
            event = 1 if t_event <= t_censor else 0

            row = {'mouse_id': mouse_id, 'model': model,
                   'treatment': 'cisplatin_5mgkg+RT_8Gy'}
            for d in TIME_DAYS:
                row['day{}_volume_mm3'.format(d)] = round(vol[d], 2)
            row['survival_days'] = round(observed, 1)
            row['event'] = int(event)
            cohort_rows.append(row)

            pet_rows.append({
                'mouse_id': mouse_id,
                'scan_day': 14,
                'body_weight_g': round(body_weight, 2),
                'injected_dose_mbq': round(dose, 3),
                'uptake_time_min': round(uptake, 1),
                'roi_volume_mm3': round(roi_volume, 2),
                'roi_activity_mean_kbq_ml': round(conc_mean, 2),
                'roi_activity_max_kbq_ml': round(conc_max, 2),
            })

    pd.DataFrame(cohort_rows).to_csv(OUT_DIR / 'mouse_cohort.csv', index=False)
    pd.DataFrame(pet_rows).to_csv(OUT_DIR / 'pet_ct_roi.csv', index=False)
    print('wrote {} mice'.format(len(cohort_rows)))


if __name__ == '__main__':
    main()
