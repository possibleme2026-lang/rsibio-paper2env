# Sensory Reweighting in Quiet Standing

Implement a faithful simplified fixed-point active-inference model of quiet standing. For each condition, compute the upright-goal belief bias, channel-specific state-update contributions, and ankle-torque postural shift.

Input: /app/environment/data/conditions.csv. Columns: condition_id, pi_prior, pi_visual, pi_proprio, pi_vestibular, bias_visual, bias_proprio, bias_vestibular, action_gain. All values are floats.

Model: prior_mean = 0.0. total_precision = pi_prior + pi_visual + pi_proprio + pi_vestibular. If total_precision <= 0, set all outputs to 0.0. Otherwise:
contribution_visual = pi_visual * bias_visual / total_precision
contribution_proprio = pi_proprio * bias_proprio / total_precision
contribution_vestibular = pi_vestibular * bias_vestibular / total_precision
belief_bias = contribution_visual + contribution_proprio + contribution_vestibular
postural_shift = action_gain * belief_bias

Output: /app/output/predictions.json. Create /app/output if needed. Write a JSON array. Each object must contain condition_id and numeric fields belief_bias, contribution_visual, contribution_proprio, contribution_vestibular, postural_shift with full floating-point precision.

Scoring: hidden gold is computed from the same model on a canonical input copy. RMSE is computed over all numeric fields across all condition rows. score = max(0, 1 - RMSE / 0.005).

Constraints: do not access /tests; modifying the environment input will not change the hidden gold. Deterministic execution only.
