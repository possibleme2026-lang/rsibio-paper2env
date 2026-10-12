import json, numpy as np, pandas as pd, os, argparse
from tlr_sim import simulate, SPECIES

GOLD_PARAMS = {
    "k1": 0.5, "k2": 0.05,
    "k3": 0.4, "k4": 0.08,
    "k5": 0.3, "k6": 0.06,
    "k7": 0.2, "k8": 0.1,
    "k9": 0.25, "k10": 0.15,
    "k11": 0.4, "k12": 0.2,
    "k13": 0.35, "k14": 0.25,
    "k15": 0.5, "k16": 0.3, "k17": 0.1,
    "k_fb": 2.5, "K_fb": 0.3, "n_fb": 1.0
}
INITIAL = {
    "M": 100.0, "M_ub": 0.0,
    "I": 80.0, "I_ub": 0.0,
    "T": 60.0, "T_ub": 0.0,
    "B": 50.0, "B_act": 0.0,
    "E": 30.0, "E_act": 0.0,
    "TANK": 40.0, "NFkB": 0.0,
    "MAPK": 0.0, "Cyt": 0.0
}
TIME_POINTS = np.array([0,5,15,30,60,120,240,360,480,720], dtype=float)
CONDITIONS = ["WT", "IKKepsilon_KO", "TANK_KO"]

def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    gold_model = {
        "parameters": GOLD_PARAMS,
        "initial": INITIAL,
        "feedback_enabled": True,
        "feedback_enzyme": "E_act",
        "feedback_targets": ["I_ub", "T_ub"]
    }
    all_rows = []
    for cond in CONDITIONS:
        y = simulate(gold_model, cond, TIME_POINTS)
        for i, t in enumerate(TIME_POINTS):
            for j, sp in enumerate(SPECIES):
                all_rows.append({"condition": cond, "time": t, "species": sp, "value": y[i, j]})
    df = pd.DataFrame(all_rows)
    wt_max = df[df.condition=="WT"].groupby("species")["value"].max()
    df["value"] = df.apply(lambda r: r["value"] / wt_max[r["species"]], axis=1)
    rng = np.random.default_rng(42)
    df["value"] = df["value"] * rng.lognormal(mean=0, sigma=0.1, size=len(df))
    df["value"] = df["value"].clip(0, 1.5)
    piv = df.pivot_table(index="time", columns=["condition", "species"], values="value")
    piv = piv.reindex(columns=pd.MultiIndex.from_product([CONDITIONS, SPECIES]))
    piv.to_csv(os.path.join(outdir, "experimental_data.csv"))
    for cond in CONDITIONS:
        sub = df[df.condition==cond].pivot(index="time", columns="species", values="value")
        sub = sub.reindex(columns=SPECIES)
        sub.to_csv(os.path.join(outdir, f"{cond}_data.csv"))
    baseline = {
        "parameters": {k:v for k,v in GOLD_PARAMS.items() if k not in ["k_fb", "K_fb", "n_fb"]},
        "initial": INITIAL,
        "feedback_enabled": False,
        "feedback_enzyme": "E_act",
        "feedback_targets": []
    }
    with open(os.path.join(outdir, "baseline_model.json"), "w") as f:
        json.dump(baseline, f, indent=2)
    copy_df = pd.DataFrame({"species": list(INITIAL.keys()), "copy_number": list(INITIAL.values())})
    copy_df.to_csv(os.path.join(outdir, "copy_numbers.csv"), index=False)
    print("Data generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    main(args.out)
