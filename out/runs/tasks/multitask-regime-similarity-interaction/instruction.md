# Multi-task learning: training regime x structural similarity

You are re-analysing trial-level behavioural data from a multi-task learning experiment. Participants learned two tasks - a *conceptual* categorisation task and a *spatial* navigation task - whose underlying latent structure was either similar or distinct. Training followed either a *blocked* regime (all trials of one task, then all trials of the other) or an *interleaved* regime (short alternating sub-blocks), and the task order was counterbalanced across participants. A post-test measured performance on both tasks.

Your job is to rebuild the analysis pipeline from the raw file, quantify the training-regime x structural-similarity interaction on the post-test, and decide whether that interaction depends on the order in which the two tasks were learned.

## Inputs

`/app/data/behavior.csv` - one row per trial (~73,600 rows, ~6 MB).

| column | meaning |
|---|---|
| `participant_id` | integer id, unique per participant |
| `order` | `conceptual_first` or `spatial_first` - which task was learned first |
| `similarity` | `similar` or `distinct` - structural similarity of the two tasks |
| `regime` | `blocked` or `interleaved` - training regime |
| `block` | `practice`, `learn` or `posttest` |
| `task` | `conceptual` or `spatial` |
| `trial_in_block` | 1-based index of the trial inside that (participant, block, task) sequence |
| `stimulus` | `S1` ... `S4` |
| `response` | `left` or `right` |
| `correct` | 1 or 0 |
| `rt_ms` | response time in milliseconds |
| `feedback` | `correct` or `incorrect` |

Design: 2 (order) x 2 (similarity) x 2 (regime) between subjects, 36 participants per cell. Every participant runs a 21-trial practice block on the first task, 80 learning trials on each task, and a 40-trial post-test on each task. Some participants did not finish the session.

## Task

Build the pipeline and write the result file `/app/results.json`.

**Step 1 - exclusions, applied in exactly this order**

1. **E1**: drop every participant whose accuracy over *all* practice trials is below 0.60. (E1 is computed on the raw practice rows; no trial-level filter is applied before it.)
2. **E2**: drop every trial with `rt_ms < 150` or `rt_ms > 4000`.
3. **E3**: drop the first 5 trials of every (participant, block, task) sequence, i.e. keep rows with `trial_in_block > 5`.
4. **E4**: drop every participant who is left with fewer than 30 usable post-test trials on *either* task.

Call the surviving rows `D`. Report `n_participants_retained` (participants surviving E1 only), `n_participants_analyzed` (participants surviving E1-E4) and `n_trials_analyzed` (number of rows in `D`, across all blocks and both tasks).

**Step 2 - post-test accuracy**

For each participant in `D`, compute two numbers from the post-test rows:

* accuracy on the **first-learned task** (`conceptual` when `order == conceptual_first`, else `spatial`),
* accuracy on the **second-learned task** (the other one).

Both are proportions of correct trials for that participant. Then average those participant-level proportions inside each of the eight `order|similarity|regime` cells. All cell means are therefore *equal-weighted across participants*, not pooled over trials. Do this separately for the first-learned and the second-learned task.

**Step 3 - interaction contrasts on the second-learned task**

For each order define, from the second-task cell means,

`delta(order) = [mean(blocked, similar) - mean(blocked, distinct)] - [mean(interleaved, similar) - mean(interleaved, distinct)]`

Report `interaction_conceptual_first`, `interaction_spatial_first` and their difference `interaction_three_way_difference = interaction_conceptual_first - interaction_spatial_first`.

**Step 4 - inference**

Fit an ordinary-least-squares linear model of the participant-level second-task post-test accuracy on the full factorial design (order, similarity, regime and all interactions - the saturated 8-cell model), and obtain two-sided p-values for the two contrasts of step 3 using the corresponding linear contrast on the fitted model (Wald / t test). Report them as `p_interaction_conceptual_first` and `p_interaction_spatial_first`.

**Step 5 - decision**

Set `asymmetric_interaction_supported` to

`(abs(interaction_conceptual_first) >= 0.10) and (p_interaction_conceptual_first < 0.01) and (abs(interaction_spatial_first) < 0.5 * abs(interaction_conceptual_first))`

evaluated on your own reported numbers.

## Constraints

1. **Exclusion pipeline.** E1-E4 are applied once, in the stated order, to the raw file. `n_participants_retained`, `n_participants_analyzed`, `n_trials_analyzed` and all eight `n_participants_per_cell` values must match the reference exactly (integers, no tolerance).
2. **Cell means.** The 16 accuracy values (8 cells x 2 tasks) must be participant-level equal-weighted averages computed after all four exclusions, and must match the reference to within 0.005. Pooling trials instead of participants, or skipping any exclusion, will break this tolerance.
3. **Contrasts.** `interaction_conceptual_first`, `interaction_spatial_first` must equal the corresponding combinations of the reported second-task cell means to within 0.005, and must match the reference to within 0.01; `interaction_three_way_difference` must equal their difference and match the reference to within 0.02.
4. **Inference.** The two p-values must come from the OLS contrast of step 4 (not from a bootstrap, permutation test, trial-level logistic regression or bare cell-mean comparison), must lie in [0, 1], must satisfy `p_interaction_conceptual_first < 0.01`, and must satisfy `p_interaction_conceptual_first <= p_interaction_spatial_first`.
5. **Decision rule.** `asymmetric_interaction_supported` must be a JSON boolean and must be exactly the value produced by the step 5 formula applied to your own reported numbers.
6. **Output contract.** `/app/results.json` must contain a single JSON object with exactly the keys listed below; every scalar is a JSON number, `asymmetric_interaction_supported` is a JSON boolean, and the three maps contain exactly the eight cell keys in `order|similarity|regime` form.

## Output

Write `/app/results.json`:

```json
{
  "n_participants_retained": 0,
  "n_participants_analyzed": 0,
  "n_trials_analyzed": 0,
  "n_participants_per_cell": {
    "conceptual_first|similar|blocked": 0,
    "conceptual_first|similar|interleaved": 0,
    "conceptual_first|distinct|blocked": 0,
    "conceptual_first|distinct|interleaved": 0,
    "spatial_first|similar|blocked": 0,
    "spatial_first|similar|interleaved": 0,
    "spatial_first|distinct|blocked": 0,
    "spatial_first|distinct|interleaved": 0
  },
  "second_task_posttest_accuracy": {
    "conceptual_first|similar|blocked": 0.0,
    "conceptual_first|similar|interleaved": 0.0,
    "conceptual_first|distinct|blocked": 0.0,
    "conceptual_first|distinct|interleaved": 0.0,
    "spatial_first|similar|blocked": 0.0,
    "spatial_first|similar|interleaved": 0.0,
    "spatial_first|distinct|blocked": 0.0,
    "spatial_first|distinct|interleaved": 0.0
  },
  "first_task_posttest_accuracy": {
    "conceptual_first|similar|blocked": 0.0,
    "conceptual_first|similar|interleaved": 0.0,
    "conceptual_first|distinct|blocked": 0.0,
    "conceptual_first|distinct|interleaved": 0.0,
    "spatial_first|similar|blocked": 0.0,
    "spatial_first|similar|interleaved": 0.0,
    "spatial_first|distinct|blocked": 0.0,
    "spatial_first|distinct|interleaved": 0.0
  },
  "interaction_conceptual_first": 0.0,
  "interaction_spatial_first": 0.0,
  "interaction_three_way_difference": 0.0,
  "p_interaction_conceptual_first": 0.0,
  "p_interaction_spatial_first": 0.0,
  "asymmetric_interaction_supported": false
}
```

All numbers above are placeholders that only illustrate the schema - do not copy them. The cell key `conceptual_first|similar|blocked` means `order=conceptual_first`, `similarity=similar`, `regime=blocked`.

## Scoring

`tests/grade.py` re-derives the reference answer from a protected copy of the data generator and checks ten independent constraints, each worth one tenth of the score:

1. `data_integrity` - `/app/data/behavior.csv` is unmodified relative to the reference generator.
2. `results_schema` - file parses, all keys present, correct types, all eight cell keys present in each map.
3. `exclusion_counts` - the three participant/trial counts are exact.
4. `cell_counts` - all eight per-cell participant counts are exact.
5. `second_task_means` - all eight values within 0.005 of the reference.
6. `first_task_means` - all eight values within 0.005 of the reference.
7. `interactions` - the three interaction numbers within tolerance of the reference.
8. `p_values` - p-values are valid probabilities, `p_conceptual < 0.01` and `p_conceptual <= p_spatial` as in the reference.
9. `decision_flag` - the boolean follows the step 5 rule from your own numbers and agrees with the reference.
10. `internal_consistency` - the reported contrasts reproduce the reported cell means, and the difference-of-differences equals the difference of the two interactions.

Score = (number of checks passed) / 10. Partial credit is awarded per check.
