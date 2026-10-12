import os
import numpy as np
import pandas as pd

SEED = 20260728
N_PARTICIPANTS = 12
N_DAYS = 8
SAMPLES_PER_DAY = 8640
SITES = [f"site_{i:02d}" for i in range(1, 9)]

ATT_CHEST = {
    "indoor_day": 0.90,
    "outdoor_day": 0.96,
    "commute": 0.92,
    "home_evening": 0.88,
    "indoor_night": 0.78,
    "outdoor_night": 0.76,
}
ATT_WRIST = {
    "indoor_day": 0.65,
    "outdoor_day": 0.45,
    "commute": 0.68,
    "home_evening": 0.62,
    "indoor_night": 0.60,
    "outdoor_night": 0.55,
}


def generate_dataframe():
    rng = np.random.default_rng(SEED)
    rows = []
    for p in range(N_PARTICIPANTS):
        pid = f"P{p+1:02d}"
        site = SITES[p % len(SITES)]
        p_eff_chest = rng.lognormal(mean=-0.02, sigma=0.08)
        p_eff_wrist = rng.lognormal(mean=-0.02, sigma=0.12)
        for d in range(N_DAYS):
            day_id = f"{pid}_D{d+1:02d}"
            d_eff_chest = rng.lognormal(mean=0.0, sigma=0.05)
            d_eff_wrist = rng.lognormal(mean=0.0, sigma=0.08)
            low_cov = ((p + d) % 12 == 0)
            for t in range(SAMPLES_PER_DAY):
                hour = t * 10.0 / 3600.0
                dt = pd.Timestamp("2026-07-01") + pd.Timedelta(days=d) + pd.Timedelta(seconds=10 * t)
                r = rng.random()
                if hour < 6:
                    ctx = "outdoor_night" if r < 0.3 else "indoor_night"
                elif hour < 9:
                    ctx = "commute"
                elif hour < 17:
                    if site in ["site_02", "site_04", "site_06", "site_08"] and r < 0.4:
                        ctx = "outdoor_day"
                    else:
                        ctx = "indoor_day"
                elif hour < 22:
                    ctx = "home_evening"
                else:
                    ctx = "indoor_night"
                if ctx == "outdoor_day":
                    base = rng.lognormal(mean=7.5, sigma=0.4)
                elif ctx == "indoor_day":
                    base = rng.lognormal(mean=4.5, sigma=0.6)
                elif ctx == "commute":
                    base = rng.lognormal(mean=5.5, sigma=0.7)
                elif ctx == "home_evening":
                    base = rng.lognormal(mean=3.5, sigma=0.7)
                elif ctx == "outdoor_night":
                    base = rng.lognormal(mean=0.0, sigma=0.5)
                else:
                    base = rng.lognormal(mean=-0.5, sigma=0.5)
                if "night" in ctx and rng.random() < 0.01:
                    base *= rng.lognormal(mean=2.0, sigma=0.5)
                glasses = max(0.05, base)
                att_c = ATT_CHEST[ctx]
                att_w = ATT_WRIST[ctx]
                noise_c = rng.lognormal(mean=0, sigma=0.12)
                noise_w = rng.lognormal(mean=0, sigma=0.18)
                chest = max(0.01, glasses * att_c * p_eff_chest * d_eff_chest * noise_c)
                wrist = max(0.01, glasses * att_w * p_eff_wrist * d_eff_wrist * noise_w)
                quality = 1 if (low_cov and t % 2 == 0) else 0
                if rng.random() < 0.01:
                    quality = 1
                if quality == 1:
                    glasses = chest = wrist = -1.0
                rows.append((dt, pid, site, day_id, ctx, quality, glasses, chest, wrist))
    df = pd.DataFrame(
        rows,
        columns=[
            "datetime",
            "participant_id",
            "site",
            "participant_day",
            "context",
            "quality_flag",
            "glasses_MEDI",
            "chest_MEDI",
            "wrist_MEDI",
        ],
    )
    return df


if __name__ == "__main__":
    df = generate_dataframe()
    outpath = os.path.join(os.path.dirname(__file__), "dosimetry.csv")
    df.to_csv(outpath, index=False)
