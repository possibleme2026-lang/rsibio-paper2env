#!/usr/bin/env python3
"""Deterministic synthetic 6-hourly pond / air temperature record (2022-2025).

Synthetic-but-realistic stand-in for the manual readings described in the paper:
an inland subtropical nursery pond (Cumilla, Bangladesh) with a strong annual
cycle, synoptic cold fronts that are stronger in winter, a damped diurnal water
cycle and 3-6% missing manual / station entries.  Re-running with the same seed
always reproduces the same file, so the verifier can regenerate it independently.
"""
import argparse

import numpy as np
import pandas as pd

SEED = 20260801
START = "2022-01-01 00:00"
END = "2025-12-31 18:00"


def build_frame(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, END, freq="6h")
    n = len(idx)
    doy = idx.dayofyear.to_numpy(dtype=float)
    hod = idx.hour.to_numpy(dtype=float)

    # synoptic weather anomaly: AR(1), variance larger in winter (cold fronts)
    innov = 0.42 + 0.34 * np.cos(2 * np.pi * (doy - 15.0) / 365.25)
    anom = np.zeros(n)
    for i in range(1, n):
        anom[i] = 0.945 * anom[i - 1] + innov[i] * rng.normal()

    air = (26.1 + 6.3 * np.cos(2 * np.pi * (doy - 200.0) / 365.25)
           + 4.1 * np.sin(2 * np.pi * (hod - 9.0) / 24.0)
           + anom + rng.normal(0.0, 0.40, n))

    # pond water: damped diurnal cycle, slow first-order response, mixing noise
    water_anom = np.zeros(n)
    for i in range(1, n):
        water_anom[i] = 0.86 * water_anom[i - 1] + 0.15 * anom[i] + 0.75 * rng.normal()
    water = (26.1 + 6.3 * np.cos(2 * np.pi * (doy - 218.0) / 365.25)
             + 1.55 * np.sin(2 * np.pi * (hod - 13.0) / 24.0)
             + water_anom + rng.normal(0.0, 0.15, n))

    clear = np.clip(np.sin(np.pi * (hod - 6.0) / 12.0), 0.0, None)
    cloud = np.clip(0.85 - 0.12 * anom + rng.normal(0.0, 0.15, n), 0.15, 1.0)
    solar = 880.0 * clear ** 1.15 * cloud
    humidity = np.clip(79.0 - 1.1 * (air - 26.1) + rng.normal(0.0, 6.5, n), 28.0, 99.0)
    wind = np.clip(2.1 + 0.55 * np.abs(anom) + rng.normal(0.0, 0.9, n), 0.1, 11.0)

    df = pd.DataFrame({"timestamp": idx, "air_temp_c": air, "water_temp_c": water,
                       "solar_rad_wm2": solar, "humidity_pct": humidity, "wind_ms": wind})
    for col, rate in (("air_temp_c", 0.030), ("water_temp_c", 0.030),
                      ("solar_rad_wm2", 0.045), ("humidity_pct", 0.055), ("wind_ms", 0.060)):
        df.loc[rng.random(n) < rate, col] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/app/data/pond_temperatures.csv")
    args = ap.parse_args()
    df = build_frame()
    df.to_csv(args.out, index=False, float_format="%.2f")
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
