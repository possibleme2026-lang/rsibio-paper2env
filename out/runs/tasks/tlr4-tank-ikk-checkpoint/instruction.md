# TLR4 signaling: fit missing TANK-IKKε negative feedback

You are a computational biologist studying Toll-like receptor 4 (TLR4) signaling in mouse macrophages. A rule-based model of the proximal MyD88-IRAK-TRAF6 module has been calibrated to early activation data, but it fails to capture deactivation of MyD88, TRAF6-associated species, and IKKβ at late time points. Your goal is to infer the missing negative feedback, implement it in the model, and calibrate it to reproduce the provided time-course data from wild-type (WT), IKKε knockout (IKKε KO), and TANK knockout (TANK KO) macrophages.

## Inputs
- `/app/data/experimental_data.csv`: time-course measurements (10 time points) for 14 signaling species across WT, IKKε KO, and TANK KO conditions. Values are normalized abundance (0-1) with experimental noise.
- `/app/data/baseline_model.json`: a rule-based model specification without negative feedback. It contains kinetic parameters and initial copy numbers.
- `/app/data/tlr_sim.py`: Python module with `simulate(model, condition, timepoints)` that integrates the ODEs. Read it to understand the model structure and how to add feedback.
- `/app/data/copy_numbers.csv`: measured protein copy numbers for normalization.

## Task
1. Explore the data and baseline model. Simulate the baseline model and compute residuals against WT data.
2. Identify which species show systematic model failure (sustained over-prediction at late times). The paper localized the missing regulation to the proximal MyD88-IRAK-TRAF6 module.
3. Hypothesize a negative feedback checkpoint: IKKε (activated downstream) restrains the IRAK1-TRAF6 ubiquitin node. The scaffold TANK is required for IKKε activation.
4. Extend the model by enabling the negative feedback from activated IKKε (`E_act`) to IRAK1 ubiquitinated (`I_ub`) and TRAF6 ubiquitinated (`T_ub`).
5. Fit the feedback parameters (`k_fb`, `K_fb`, `n_fb`) to the WT data using least squares.
6. Validate the fitted model against IKKε KO and TANK KO data. Ensure the model reproduces the observed phenotypes: enhanced NF-κB, MAPK, and cytokine responses in both KOs; loss of IKKε phosphorylation in TANK KO; increased IRAK1 and TRAF6 ubiquitination but not MyD88 ubiquitination in KOs.
7. Write the fitted model to `/app/output/model_fitted.json` and a summary to `/app/output/result.json`.

## Constraints
1. `result.json` must contain the keys: `missing_regulator`, `scaffold`, `affected_node`, `feedback_parameters`, `model_file`.
2. `missing_regulator` must be `"IKKepsilon"` (or `"IKKε"`), `scaffold` must be `"TANK"`, and `affected_node` must mention both `"IRAK1"` and `"TRAF6"`.
3. `model_fitted.json` must be valid JSON and set `feedback_enabled` to `true`, with `feedback_enzyme` = `"E_act"` and `feedback_targets` including `"I_ub"` and `"T_ub"`.
4. Fitted `k_fb` must be in [0.1, 20.0] and within 2-fold of the true value (2.5).
5. Fitted `K_fb` must be in [0.01, 2.0] and within 2-fold of the true value (0.3).
6. The fitted model must reproduce the provided WT, IKKε KO, and TANK KO time courses with RMSE < 0.15 across all species and time points.
7. In IKKε KO, simulated NF-κB, MAPK, and cytokine at t=240 min must be > 1.5-fold higher than WT.
8. In TANK KO, simulated activated IKKε (`E_act`) at t=60 min must be < 0.1.
9. In both KOs, simulated IRAK1_ub and TRAF6_ub at t=120 min must be > 1.2-fold higher than WT.
10. In both KOs, simulated MyD88_ub at t=120 min must be < 1.1-fold higher than WT.
11. The output directory `/app/output/` must contain `model_fitted.json` and `result.json`.

## Output
- `/app/output/model_fitted.json`: the fitted model JSON.
- `/app/output/result.json`: a JSON object with:
```json
{
  "missing_regulator": "IKKepsilon",
  "scaffold": "TANK",
  "affected_node": "IRAK1-TRAF6 ubiquitin node",
  "feedback_parameters": {"k_fb": 2.5, "K_fb": 0.3, "n_fb": 1.0},
  "model_file": "model_fitted.json"
}
```

## Scoring
The grader checks constraints 1-11 independently and awards partial credit (each constraint = 1/11 of score). The final score is the fraction of constraints passed. Score and reward are written to `/logs/verifier/reward.json`.
