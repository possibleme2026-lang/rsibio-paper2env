# Reproduce EEG correlates of prioritization in working memory

You are analyzing synthetic EEG data from a two-step retrocuing working memory task. In the prioritization condition, a cue indicates which of two items is prioritized; in the neutral condition, a cue is uninformative. The original study found that the prioritization cue triggers a phase reset and oscillatory power increase at ~15 Hz (low-beta), whereas the neutral cue triggers ~20 Hz (higher-beta). Burst analysis ruled out bursting as the underlying factor. Self-correlation analysis showed greater reorganization of the whole-topography EEG pattern for the prioritization cue than for the neutral cue.

Your task is to implement the analysis pipeline that reproduces these key metrics.

## Inputs

- `/app/data/eeg_data.npz`: NumPy archive with keys:
  - `data_pri`: array shape (120, 32, 626) — EEG trials for prioritization condition.
  - `data_neu`: array shape (120, 32, 626) — EEG trials for neutral condition.
  - `times`: array shape (626,) — time points in seconds, from -0.5 to 2.0 s relative to cue onset.
  - `ch_names`: array of 32 channel names.
  - `fs`: sampling rate (250 Hz).
- `/app/data/events.csv`: trial metadata (condition, cue_onset_sample).
- `/app/data/README.md`: dataset description.

## Task

Implement the following analyses and save the results to `/app/results.json`:

1. **Peak frequency**: For each condition, compute Morlet wavelet power for frequencies 10–25 Hz in 0.5 Hz steps with `n_cycles=7`. Average power across trials and channels. Baseline-normalize each frequency by the mean power in the pre-cue window (-0.2 to 0 s). For each condition, find the frequency with maximum normalized power in the 0.2–0.8 s post-cue window. Report as `peak_freq_prioritization` and `peak_freq_neutral`.

2. **ITPC at 15 Hz**: Compute complex Morlet wavelet coefficients at 15 Hz (`n_cycles=7`). Average across trials, take the absolute value to obtain inter-trial phase coherence (ITPC). Average over channels and the 0.2–0.8 s window. Report as `itpc_15hz_prioritization` and `itpc_15hz_neutral`.

3. **Beta burst rate**: Bandpass filter the data between 15 and 25 Hz (Butterworth order 4). Compute the Hilbert envelope. For each trial and channel, z-score the envelope using the mean and standard deviation from the baseline period (-0.5 to 0 s). Detect bursts as contiguous periods where the z-score exceeds 3 for at least 50 ms. Count bursts in the 0–1.5 s post-cue window. Compute burst rate (bursts per second) by dividing the total number of bursts by the total time (1.5 s × 120 trials × 32 channels). Report as `burst_rate_prioritization` and `burst_rate_neutral`.

4. **Topographic self-correlation**: Bandpass filter the data between 1 and 30 Hz (Butterworth order 4). For each trial, and for each time point in the 0.2–0.8 s window, extract the spatial pattern across the 32 channels and z-score it across channels. Compute the Pearson correlation between the pattern at time t and the pattern at time t + lag, for lags of 50, 100, 150, and 200 ms. Fisher z-transform each correlation. Average across all time points, lags, and trials. Report as `self_corr_prioritization` and `self_corr_neutral`.

5. **Output**: Save a JSON file at `/app/results.json` with exactly the eight keys above, all numeric (float).

## Constraints

Your results must satisfy all of the following:

1. `peak_freq_prioritization` must be between 14.0 and 16.0 Hz, and `peak_freq_neutral` must be between 19.0 and 21.0 Hz.
2. `itpc_15hz_prioritization` must exceed `itpc_15hz_neutral` by at least 0.05.
3. The absolute difference between `burst_rate_prioritization` and `burst_rate_neutral` must be less than 0.5 bursts/s.
4. `self_corr_neutral` must exceed `self_corr_prioritization` by at least 0.1 (Fisher z units).
5. The output file must be valid JSON at `/app/results.json` containing exactly the eight specified keys, each with a numeric value.

## Scoring

The grader recomputes the gold-standard metrics from the raw data using the same pipeline described above. It checks each of the five constraints independently and awards partial credit: `score = (number of constraints passed) / 5`. The score and reward are written to `/logs/verifier/reward.json` as `{"score": <value>, "reward": <value>}`.
