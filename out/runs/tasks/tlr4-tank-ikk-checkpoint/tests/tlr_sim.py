import numpy as np
from scipy.integrate import odeint

SPECIES = ["M","M_ub","I","I_ub","T","T_ub","B","B_act","E","E_act","TANK","NFkB","MAPK","Cyt"]

def simulate(model, condition, timepoints):
    params = model["parameters"]
    y0 = np.array([model["initial"][s] for s in SPECIES])
    if condition == "IKKepsilon_KO":
        y0[SPECIES.index("E")] = 0.0
        y0[SPECIES.index("E_act")] = 0.0
    elif condition == "TANK_KO":
        y0[SPECIES.index("TANK")] = 0.0
    def ode(y, t):
        M, M_ub, I, I_ub, T, T_ub, B, B_act, E, E_act, TANK, NFkB, MAPK, Cyt = y
        L = 1.0 if t >= 0 else 0.0
        dM = -params["k1"]*L*M + params["k2"]*M_ub
        dM_ub = params["k1"]*L*M - params["k2"]*M_ub
        dI = -params["k3"]*M_ub*I + params["k4"]*I_ub
        dI_ub = params["k3"]*M_ub*I - params["k4"]*I_ub
        dT = -params["k5"]*I_ub*T + params["k6"]*T_ub
        dT_ub = params["k5"]*I_ub*T - params["k6"]*T_ub
        dB_act = params["k7"]*T_ub*(B - B_act) - params["k8"]*B_act
        dB = -dB_act
        dE_act = params["k9"]*T_ub*TANK*(E - E_act) - params["k10"]*E_act
        dE = -dE_act
        dTANK = 0.0
        dNFkB = params["k11"]*B_act*(1 - NFkB) - params["k12"]*NFkB
        dMAPK = params["k13"]*T_ub*(1 - MAPK) - params["k14"]*MAPK
        dCyt = params["k15"]*NFkB + params["k16"]*MAPK - params["k17"]*Cyt
        if model.get("feedback_enabled", False):
            k_fb = params["k_fb"]
            K_fb = params["K_fb"]
            n_fb = params.get("n_fb", 1.0)
            if "I_ub" in model.get("feedback_targets", []):
                fb_I = k_fb * (E_act**n_fb) * I_ub / (K_fb**n_fb + I_ub**n_fb)
                dI_ub -= fb_I
                dI += fb_I
            if "T_ub" in model.get("feedback_targets", []):
                fb_T = k_fb * (E_act**n_fb) * T_ub / (K_fb**n_fb + T_ub**n_fb)
                dT_ub -= fb_T
                dT += fb_T
        return [dM, dM_ub, dI, dI_ub, dT, dT_ub, dB, dB_act, dE, dE_act, dTANK, dNFkB, dMAPK, dCyt]
    sol = odeint(ode, y0, timepoints)
    return sol
