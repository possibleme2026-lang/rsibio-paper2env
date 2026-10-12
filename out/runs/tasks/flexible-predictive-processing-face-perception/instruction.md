# Task-dependent predictive processing in face perception

You are analyzing data from a face perception experiment inspired by a study on flexible predictive processing. Participants categorized either the emotional expression (happy vs. angry) or the sex (male vs. female) of faces. Expectations were manipulated via congruency between facial sex and emotional expression (congruent: angry male, happy female; incongruent: happy male, angry female). Mouse trajectories and EEG were recorded. Your goal is to reproduce the key analyses and quantify how the congruency effect depends on task demands (emotion vs. sex judgment).

## Inputs

- `/app/data/mouse_trajectories.csv`: long-format mouse trajectories. Columns: `subject_id`, `trial_id`, `task` (`emotion` or `sex`), `congruency` (`congruent` or `incongruent`), `time` (normalized 0-1, 101 points), `x`, `y`. 10 subjects × 2 tasks × 100 trials × 101 timepoints = 202,000 rows.
- `/app/data/events.csv`: trial-level metadata. Columns: `subject_id`, `trial_id`, `task`, `congruency`, `rt_ms`, `response`, `accuracy`.
- `/app/data/eeg/sub-XX.npz`: one file per subject. Arrays: `eeg` shape `(200, 16, 128)` (trials × channels × timepoints) in µV, `times` shape `(128,)` in ms (from -100 to 500 ms), `trial_ids` shape `(200,)`. Trials are ordered: first 100 emotion (50 congruent, 50 incongruent), then 100 sex (50 congruent, 50 incongruent). Channel names are `EEG01`..`EEG16`.
- `/app/data/subjects.txt`: list of subject IDs.

## Task

Write a reproducible analysis pipeline that:

1. **Behavioral analysis**: For each trial, compute the maximum absolute deviation (MAD) of the mouse trajectory from the straight line `y=0` (i.e., `max(|y|)` over the 101 timepoints). For each subject and task, compute the mean MAD and mean RT for congruent and incongruent trials, and the difference `incongruent - congruent`. Perform a one-sample t-test across subjects for each task. Also perform a paired t-test between tasks on the difference scores (emotion − sex).

2. **Time-resolved RSA**: For each subject, z-score each EEG channel across all 200 trials. For each task and timepoint `t`, compute the cross-validated Mahalanobis-like distance between congruent and incongruent condition means:
   `d(t) = sqrt( sum_c (mean_inc(c,t) - mean_con(c,t))^2 / var(c,t) )`,
   where `var(c,t)` is the pooled variance across trials (average of congruent and incongruent variances, with ddof=1) for channel `c` at timepoint `t`. Average `d(t)` across subjects. Find the peak amplitude and its latency (in ms) for each task.

3. **Multivariate decoding**: For each subject and task, train a logistic regression classifier (L2 penalty, `C=1.0`, `solver='lbfgs'`, `max_iter=1000`) to discriminate congruent vs. incongruent trials from the 16-channel EEG pattern at each timepoint. Use 5-fold stratified cross-validation with `shuffle=True, random_state=42`. Compute balanced accuracy. Average across subjects. Find the peak balanced accuracy and its latency for each task.

4. **EEG–mouse fusion**: For each task, use the group-level peak latency from decoding. At that timepoint, for each subject, run the same 5-fold cross-validated logistic regression and collect out-of-fold decision function values for each trial. Compute Pearson correlation between these decision values and the trial-level MAD. Convert correlations to Fisher z, average across subjects, and back-transform to mean r. Report mean r for each task and the difference (emotion − sex).

## Constraints

1. The output JSON must contain exactly the schema shown below. All numeric values must be finite and rounded to 4 decimal places (p-values to 6 decimal places).
2. Behavioral RT: the congruency effect (incongruent − congruent) must be larger for emotion than for sex; both one-sample t-tests must have p < 0.05; the task difference must have p < 0.05.
3. Behavioral MAD: the congruency effect must be larger for emotion than for sex; both one-sample t-tests must have p < 0.05; the task difference must have p < 0.05.
4. RSA: peak amplitude for emotion must be greater than for sex; both peak latencies must lie between 100 and 400 ms.
5. RSA: peak latency for emotion must be earlier than for sex.
6. Decoding: peak balanced accuracy for emotion must be greater than for sex; both peak accuracies must be > 0.6.
7. Decoding: peak latency for emotion must be earlier than for sex.
8. Fusion: mean r for emotion must be greater than for sex; emotion mean r must be > 0.15; the difference must be > 0.05.
9. All reported values must match the reference analysis within the tolerance used by the grader.

## Output

Write your results to `/app/output/results.json` with this exact structure (example values shown):

```json
{
  "behavior": {
    "rt": {
      "emotion": {"mean_diff": 78.1234, "t_stat": 12.3456, "p_value": 0.000001},
      "sex": {"mean_diff": 28.1234, "t_stat": 6.7890, "p_value": 0.000123},
      "task_difference": {"mean_diff": 50.0000, "t_stat": 8.1234, "p_value": 0.000045}
    },
    "mad": {
      "emotion": {"mean_diff": 0.1500, "t_stat": 15.1234, "p_value": 0.000001},
      "sex": {"mean_diff": 0.0500, "t_stat": 5.1234, "p_value": 0.000678},
      "task_difference": {"mean_diff": 0.1000, "t_stat": 10.1234, "p_value": 0.000012}
    }
  },
  "rsa": {
    "emotion": {"peak_amplitude": 4.0000, "peak_latency_ms": 200.0000},
    "sex": {"peak_amplitude": 1.6000, "peak_latency_ms": 300.0000}
  },
  "decoding": {
    "emotion": {"peak_accuracy": 0.9800, "peak_latency_ms": 200.0000},
    "sex": {"peak_accuracy": 0.7500, "peak_latency_ms": 300.0000}
  },
  "fusion": {
    "emotion": {"mean_r": 0.4500},
    "sex": {"mean_r": 0.2000},
    "difference": 0.2500
  }
}
```

## Scoring

The grader recomputes all reference values from `/app/data` and checks each constraint independently. The score is the fraction of passed checks (partial credit). Checks include: schema and finite numbers, behavioral RT and MAD effects, RSA amplitude and latency, decoding accuracy and latency, fusion correlation, p-value ranges, and numerical agreement with the reference within tolerance.
