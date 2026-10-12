#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import json, numpy as np, pandas as pd
from scipy.optimize import minimize
from tlr_sim import simulate, SPECIES

data = pd.read_csv('/app/data/experimental_data.csv', header=[0,1], index_col=0)
timepoints = data.index.values.astype(float)

with open('/app/data/baseline_model.json') as f:
    model = json.load(f)

model['feedback_enabled'] = True
model['feedback_targets'] = ['I_ub', 'T_ub']
model['parameters']['k_fb'] = 1.0
model['parameters']['K_fb'] = 0.5
model['parameters']['n_fb'] = 1.0

def objective(x):
    model['parameters']['k_fb'] = x[0]
    model['parameters']['K_fb'] = x[1]
    y = simulate(model, 'WT', timepoints)
    wt_max = np.max(y, axis=0)
    wt_max[wt_max == 0] = 1.0
    y_norm = y / wt_max
    err = 0.0
    for i, t in enumerate(timepoints):
        for j, sp in enumerate(SPECIES):
            gt = data.loc[t, ('WT', sp)]
            pred = y_norm[i, j]
            err += (pred - gt)**2
    return err

res = minimize(objective, [2.0, 0.3], method='Nelder-Mead', bounds=[(0.1, 20.0), (0.01, 2.0)])
k_fb, K_fb = res.x
model['parameters']['k_fb'] = k_fb
model['parameters']['K_fb'] = K_fb
model['parameters']['n_fb'] = 1.0

with open('/app/output/model_fitted.json', 'w') as f:
    json.dump(model, f, indent=2)

result = {
    'missing_regulator': 'IKKepsilon',
    'scaffold': 'TANK',
    'affected_node': 'IRAK1-TRAF6 ubiquitin node',
    'feedback_parameters': {'k_fb': k_fb, 'K_fb': K_fb, 'n_fb': 1.0},
    'model_file': 'model_fitted.json'
}
with open('/app/output/result.json', 'w') as f:
    json.dump(result, f, indent=2)
print('Fitted k_fb=%.3f K_fb=%.3f' % (k_fb, K_fb))
PY
