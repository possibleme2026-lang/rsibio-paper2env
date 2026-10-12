# Cold-stress early warning for a Nile tilapia nursery pond from 6-hourly manual thermometer readings

A subtropical nursery pond in Cumilla (Bangladesh) keeps only manual records: a technician reads air temperature and pond-water temperature four times a day (00:00, 06:00, 12:00, 18:00) and transcribes solar radiation, relative humidity and wind speed from a nearby weather station. There is no automatic logger, about 3-6% of the entries were missed, and the nursery must know *in advance* when the water will fall into the cold-stress range for Nile tilapia (*Oreochromis niloticus*) fry - a 6-hourly water-temperature reading below 20 degC - so that aeration, water exchange and feeding can be adjusted 6, 12, 18 and 24 hours ahead.

Build and evaluate that low-resource early-warning framework from the raw record.

## Inputs

- `/app/data/pond_temperatures.csv` - 5,844 rows of 6-hourly observations from 2022-01-01 00:00 to 2025-12-31 18:00 with columns `timestamp`, `air_temp_c`, `water_temp_c`, `solar_rad_wm2`, `humidity_pct`, `wind_ms`. Missing manual/station entries are empty fields (NaN); timestamps are never missing and the grid is complete.
- The air-water coupling in this pond is strong but weakens in winter, so the boundary zone around 20 degC is where errors hurt most.

## Task

1. **Fit** a water-temperature forecaster on the 2022-2023 readings only (any tabular regression you can justify; the air, radiation, humidity and wind columns carry usable coupling).
2. **Develop** on 2024: for each lead time, pick one cold-stress alert threshold that maximises F1 on that year's forecasts. Freeze model and thresholds afterwards.
3. **Test** on 2025 as an independent year: for every eligible 2025 origin emit 6-, 12-, 18- and 24-hour-ahead water-temperature forecasts and binary cold-stress alerts.
4. **Report** the per-lead test-year metrics and the thresholds you froze.

## Output

`/app/output/predictions_2025.csv` with exactly these columns:

```
origin_time,target_time,lead_hours,water_temp_pred_c,alert_pred
2025-01-03 00:00:00,2025-01-03 06:00:00,6,19.42,1
2025-01-03 00:00:00,2025-01-03 12:00:00,12,19.11,1
2025-01-03 00:00:00,2025-01-03 18:00:00,18,18.95,1
2025-01-03 00:00:00,2025-01-04 00:00:00,24,19.30,1
```

- one row per (origin, lead) pair, `lead_hours` in {6, 12, 18, 24}, and `target_time` must equal `origin_time + lead_hours`;
- eligible origins: 2025 grid timestamps from 2025-01-01 00:00 through 2025-12-30 18:00 that have a recorded water temperature at the origin; skip a pair only when the water reading at `target_time` is missing; cover at least 95% of all such pairs with no duplicate (origin, lead) rows;
- `water_temp_pred_c` numeric in [-5, 45], `alert_pred` in {0, 1}.

`/app/output/metrics.json`:

```json
{
  "model": "hist-gradient-boosting",
  "train_years": [2022, 2023],
  "dev_year": 2024,
  "test_year": 2025,
  "cold_stress_threshold_c": 20.0,
  "persistence_mae_6h_c": 1.16,
  "leads": {
    "6":  {"threshold_c": 19.55, "mae_c": 0.71, "rmse_c": 0.94, "bias_c": 0.06,
           "f1": 0.78, "precision": 0.74, "recall": 0.83,
           "n_alerts": 121, "n_cold_readings": 108},
    "12": {"...": "same keys"},
    "18": {"...": "same keys"},
    "24": {"...": "same keys"}
  }
}
```

## Constraints (each is graded independently)

1. **Split discipline** - model fitting uses 2022-2023 only, threshold selection uses 2024 only, and every row of `predictions_2025.csv` has a 2025 origin; do not refit on the test year.
2. **Format and coverage** - exact columns and JSON keys as above, at least 95% coverage of the eligible (origin, lead) grid, no duplicate (origin, lead) rows, no missing forecast values, all forecasts inside [-5, 45] degC, and the input record left untouched.
3. **Alert rule** - `alert_pred` must be exactly `1` when `water_temp_pred_c < threshold_c` for that lead and `0` otherwise, where `threshold_c` is the frozen value reported in `metrics.json` and lies in [17, 22.5] degC.
4. **Metric honesty** - every number in `metrics.json` is recomputed from your own submitted rows: MAE/RMSE/bias within 0.05 degC, F1/precision/recall within 0.05 of the recomputed value at every lead.
5. **Persistence skill** - the 6-h MAE must beat the same-hour-yesterday baseline (`water_temp_c` at `target_time - 24 h`), recomputed on exactly the same rows, by at least 15%.
6. **Accuracy floors** - at every lead the test-year MAE must be within 15% (+0.03 degC) of a hidden reference model fitted with the same split protocol.
7. **Detection floors** - the 6-h cold-stress F1 must be within 0.10 of the hidden reference F1, and the 6-h sensitivity (recall on readings < 20 degC) within 0.10 of the reference sensitivity.
8. **Lead-time skill** - the 12-, 18- and 24-h F1 scores must each be within 0.10 of the hidden reference at the same lead.
9. **Boundary-zone calibration** - at 6-h lead, the mean signed error (predicted - observed) over test readings with observed water temperature between 18 and 22 degC must satisfy |bias| <= 0.8 degC.
10. **Plausibility / anti-leakage** - at every lead MAE >= 0.25 degC and F1 <= 0.995 (a forecast that tracks observations more closely than the intrinsic noise of the record is treated as future-data leakage), and the 6-h alert count must be between 0.4 and 3.0 times the number of true cold readings (no all-alert or no-alert collapse).

## Scoring

Weighted fraction of the independent checks (total weight 17.5): input-record integrity 1.0, output format/coverage 2.5, alert-rule consistency 1.0, metric honesty 2.0, persistence skill 1.5, MAE floors 1.5, 6-h F1 floor 2.0, 6-h sensitivity floor 1.5, 12/18/24-h F1 floors 1.5, boundary-zone bias 1.0, plausibility/anti-leakage + alert-rate 2.0. The verifier writes `{"score": <float>, "reward": <float>}` to `/logs/verifier/reward.json` and a per-check breakdown to stdout and `/logs/verifier/report.json`.
