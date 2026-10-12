# Multiregional STD-RNN: MMN and P300-like responses

You are implementing a hierarchical, multiregional recurrent neural network (RNN) with short-term synaptic depression (STD) to test whether STD accounts for both mismatch negativity (MMN) and P300-like responses and their attentional amplification. The model has a sensory region (S) and a frontal region (F), each with N=100 units. Synaptic depression acts on recurrent, feedforward, and feedback connections.

## Inputs
All inputs are in /app/data:
- config.json: simulation parameters (N, dt, T_trial, stimulus timing, response window, time constants, STD parameters, input gains, noise levels, number of trials per condition, number of trials to discard).
- W_S.npy, W_F.npy, W_SF.npy, W_FS.npy: 100x100 weight matrices for recurrent S, recurrent F, feedforward S->F, feedback F->S.
- selectivity.npy: 100x2 array; column 0 is the sensory input pattern for standard stimuli, column 1 for deviant stimuli.
- sequences.json: list of 300 trials (100 per condition: baseline, noise, attention). Each trial has condition, label (0=standard, 1=deviant), and seed.
- README.md: brief description.

## Task
Implement the model exactly as specified, simulate all trials, compute six metrics, and write them to /app/output/metrics.json.

### Model specification
- N = 100, dt = 1.0 ms, T_trial = 200 ms (including inter-trial interval).
- Stimulus onset at 10 ms, offset at 30 ms. Response window: 30 <= t < 80 ms.
- For each condition, process trials sequentially. At the start of the condition, initialize all synaptic depression variables D to 1.0. For each trial, reset membrane potentials u_S and u_F to 0, but carry D over from the previous trial. Simulate T_trial/dt steps.
- At each time step t (0-indexed, time = t*dt):
  - x_S = max(u_S, 0); x_F = max(u_F, 0)
  - stim = 1.0 if stim_start <= t*dt < stim_end else 0.0
  - gain = input_gain * (attn_gain if condition == 'attention' else 1.0)
  - U_fs = U_fs_attn if condition == 'attention' else U
  - sigma = noise_sigma_high if condition == 'noise' else noise_sigma
  - Generate noise for this trial: rng = np.random.default_rng(seed); noise = rng.normal(0, sigma, size=(N, steps)). Then I_S = gain * stim * selectivity[:, label] + noise[:, t].
  - Update:
    rec_S = (W_S * D_S) @ x_S
    fb = (W_FS * D_FS) @ x_F
    u_S = u_S + dt/tau_m * (-u_S + I_S + rec_S + fb)
    rec_F = (W_F * D_F) @ x_F
    ff = (W_SF * D_SF) @ x_S
    u_F = u_F + dt/tau_m * (-u_F + rec_F + ff)
    D_S = D_S + dt * ((1 - D_S)/tau_d - U * x_S[None, :] * D_S)
    D_F = D_F + dt * ((1 - D_F)/tau_d - U * x_F[None, :] * D_F)
    D_SF = D_SF + dt * ((1 - D_SF)/tau_d - U * x_S[None, :] * D_SF)
    D_FS = D_FS + dt * ((1 - D_FS)/tau_d - U_fs * x_F[None, :] * D_FS)
    Clip all D to [0, 1].
  - Record x_S and x_F at each time step.

### Metrics
For each trial, compute:
- mean_S: average of x_S over all neurons and response window steps.
- mean_F: average of x_F over all neurons and response window steps.
- mean_S_vec: average of x_S over response window for each neuron (length N).
- mean_F_vec: average of x_F over response window for each neuron (length N).
- time_series_S: average of x_S over neurons at each time step (length steps).
- time_series_F: average of x_F over neurons at each time step (length steps).

Discard the first n_discard trials of each condition.

Compute:
1. MMN_sensory = mean(mean_S for baseline label=1) - mean(mean_S for baseline label=0).
2. P300_frontal = mean(mean_F for baseline label=1) - mean(mean_F for baseline label=0).
3. P300_latency_delay: average time_series_S and time_series_F across baseline label=1 trials. Find peak time (argmax * dt) for S and F. Delay = peak_F - peak_S (ms).
4. Noise_accuracy: threshold = (mean(mean_F for baseline label=0) + mean(mean_F for baseline label=1)) / 2. For noise condition trials, predict label=1 if mean_F > threshold else 0. Accuracy = fraction correct.
5. State_space_expansion: for baseline trials, compute participation ratio (PR) of mean_F_vec for label=0 and label=1. PR = (sum lambda)^2 / sum lambda^2 where lambda are eigenvalues of the covariance matrix (after subtracting the mean). Expansion = log(PR_dev / PR_std).
6. Attention_amplification: att_diff = mean(mean_F for attention label=1) - mean(mean_F for attention label=0). base_diff = P300_frontal. Attention_amplification = att_diff / base_diff - 1.

## Constraints
1. Output file /app/output/metrics.json must contain exactly the keys: MMN_sensory, P300_frontal, P300_latency_delay, Noise_accuracy, State_space_expansion, Attention_amplification, status. status must be 'ok'. All metric values must be finite JSON numbers.
2. MMN_sensory must be positive and within 5% relative or 0.01 absolute of the hidden gold value.
3. P300_frontal must be greater than MMN_sensory and within 5% relative or 0.01 absolute of the hidden gold value.
4. P300_latency_delay must be positive and within 5 ms of the hidden gold value.
5. Noise_accuracy must be >= 0.6 and within 0.05 of the hidden gold value.
6. State_space_expansion must be positive and within 0.05 of the hidden gold value.
7. Attention_amplification must be positive and within 0.05 of the hidden gold value.

## Output
Write /app/output/metrics.json with a single JSON object, e.g.:
{"MMN_sensory": 0.42, "P300_frontal": 0.91, "P300_latency_delay": 25.0, "Noise_accuracy": 0.84, "State_space_expansion": 0.18, "Attention_amplification": 0.31, "status": "ok"}

## Scoring
The grader checks each constraint independently. Score = (number of constraints passed) / 7. Partial credit is given. The gold metrics are recomputed from the provided inputs by the verifier; you must implement the model faithfully to match them.
