import numpy as np
import pandas as pd
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

def generate_ground_truth():
    gold_model = {
        "parameters": GOLD_PARAMS,
        "initial": INITIAL,
        "feedback_enabled": True,
        "feedback_enzyme": "E_act",
        "feedback_targets": ["I_ub", "T_ub"]
    }
    wt_sim = simulate(gold_model, "WT", TIME_POINTS)
    wt_max = np.max(wt_sim, axis=0)
    wt_max[wt_max == 0] = 1.0
    data = {}
    for cond in CONDITIONS:
        y = simulate(gold_model, cond, TIME_POINTS)
        y_norm = y / wt_max
        for j, sp in enumerate(SPECIES):
            data[(cond, sp)] = y_norm[:, j]
    df = pd.DataFrame(data, index=TIME_POINTS)
    return df, wt_max
