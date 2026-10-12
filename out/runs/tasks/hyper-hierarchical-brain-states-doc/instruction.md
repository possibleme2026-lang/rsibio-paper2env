# Hyper-hierarchical brain states in disorders of consciousness

You are a computational neuroscientist re-analysing a resting-state fMRI cohort of disorders of consciousness (DOC). The cohort contains parcellated BOLD time series (100 regions of interest x 300 volumes, TR = 2.0 s) for healthy controls (CNT), minimally conscious state (MCS) patients and unresponsive wakefulness syndrome (UWS) patients, together with Coma Recovery Scale-Revised (CRS-R) scores. Your job is to re-implement the trophic-hierarchy pipeline specified below, recover the pathological hyper-hierarchical dynamic brain state, and quantify its relation to diagnosis and behavioural responsiveness. The grader re-runs the same pipeline on the same inputs, so every numeric step below is normative.

## Inputs

- /app/data/subjects.csv - header subject_id,group,age,sex,crsr_score; 45 rows (sub-CNT01..sub-CNT15, sub-MCS01..sub-MCS15, sub-UWS01..sub-UWS15). crsr_score is an integer for patients and empty for controls.
- /app/data/roi_networks.csv - header roi,network; 100 rows in acquisition order. Network sizes: FRO 20, PAR 15, TEM 15, OCC 10, CIN 10, THA 10, INS 10, DMN 10.
- /app/data/bold/<subject_id>.csv - one file per subject: 300 rows (time, TR = 2.0 s) x 100 columns (ROIs in the order given by roi_networks.csv), raw BOLD signal in arbitrary units (SD about 2). Each file is roughly 250 kB.
- /app/data/gen_inputs.py - the deterministic generator (seed 20260730) used to create the cohort. Never modify anything inside /app/data.

## Analysis specification (follow exactly)

1. Standardisation. Z-score every ROI time series over the 300 time points (population SD, ddof = 0).
2. Directed effective connectivity. Lag-1 ridge regression on a z-scored block X (T x 100): Xl = X[0:T-1], Y = X[1:T], G = Xl.T @ Xl + 20.0 * I100, B = solve(G, Xl.T @ Y). B[i, j] is the directed weight ROI i at t-1 -> ROI j at t.
3. Edge thresholding. thr = 80th percentile (numpy default linear interpolation) of the 9900 off-diagonal absolute weights |B|. A[i, j] = |B[i, j]| if |B[i, j]| > thr and i != j, else 0; the diagonal is always 0.
4. Trophic levels. In-strength b_j = sum_i A[i, j]. h = minimum-norm least-squares solution of (diag(b) - A.T) h = b (numpy.linalg.lstsq(M, b, rcond=None), equivalently numpy.linalg.pinv(M) @ b). Any ROI with b_j = 0 gets h_j = 1.0.
5. Trophic incoherence. F0 = mean over all directed edges (i, j) with A[i, j] > 0 of (h_j - h_i - 1)^2.
6. Static per-subject metrics: h (100 values) and F0 from all 300 volumes.
7. Dynamic windows. 41 windows per subject: length 60 TR, step 6 TR, start indices 0, 6, 12, ..., 240. Recompute A_w (steps 2-3) and h_w (step 4) inside each window.
8. State identification. Stack the 1845 h_w vectors (45 subjects x 41 windows) into a (1845, 100) matrix, z-score each of the 100 ROI dimensions across the pooled windows (ddof = 0), then run KMeans(n_clusters=4, n_init=10, random_state=0). The hyper-hierarchical state is the cluster whose centroid maximises mean(centroid over FRO + THA + INS ROIs) - mean(centroid over the other 80 ROIs).
9. Per-subject dynamic metrics of the hyper state: occupancy = number of its windows / 41; dwell time = mean length of maximal consecutive runs (in windows) x 12.0 s; n_entries = number of runs.
10. Synchrony and metastability. Band-pass filter the z-scored series (4th-order Butterworth, 0.01-0.08 Hz, fs = 0.5 Hz, scipy.signal.filtfilt), take instantaneous phases with scipy.signal.hilbert, and form the order parameter R(t) = |mean_i exp(i phi_i(t))| over the 100 ROIs. synchrony = mean_t R(t) and metastability = std_t R(t) (ddof = 0), both over all 300 time points.
11. CRS-R associations (30 patients only). The composite trophic score of a network is the mean static h over that network's ROIs. Report the Spearman rank correlation (scipy.stats.spearmanr) between that composite and crsr_score for FRO, THA and INS, and between hyper-state occupancy and crsr_score.
12. Anatomical mapping. For every subject and ROI, elevation = mean h_w over that subject's hyper-state windows minus mean h_w over all 41 windows; average the elevation across all 45 subjects; rank ROIs by descending mean elevation and take the first 20 names.

## Output

Write two files. Everything must be produced offline and deterministically.

/app/outputs/results.json, with exactly these keys:

{
  "n_subjects": 45,
  "n_windows": 1845,
  "window_params": {"window_length_tr": 60, "step_tr": 6, "n_windows_per_subject": 41},
  "n_states": 4,
  "hyper_state_cluster_id": 2,
  "static_trophic": {"CNT": {"mean_f0": 0.41, "mean_trophic_level": 2.87, "sd_trophic_level": 1.02}, "MCS": {"mean_f0": 0.0, "mean_trophic_level": 0.0, "sd_trophic_level": 0.0}, "UWS": {"mean_f0": 0.0, "mean_trophic_level": 0.0, "sd_trophic_level": 0.0}},
  "dynamic_states": {"CNT": {"occupancy": 0.12, "dwell_time_s": 24.0, "n_entries": 5.0}, "MCS": {"occupancy": 0.0, "dwell_time_s": 0.0, "n_entries": 0.0}, "UWS": {"occupancy": 0.0, "dwell_time_s": 0.0, "n_entries": 0.0}},
  "state_occupancy_all_clusters": {"CNT": [0.12, 0.31, 0.28, 0.29], "MCS": [0.0, 0.0, 0.0, 0.0], "UWS": [0.0, 0.0, 0.0, 0.0]},
  "metastability": {"CNT": 0.21, "MCS": 0.0, "UWS": 0.0},
  "synchrony": {"CNT": 0.42, "MCS": 0.0, "UWS": 0.0},
  "crsr_association": {"FRO": {"rho": 0.52, "p": 0.003, "n": 30}, "THA": {"rho": 0.0, "p": 0.0, "n": 30}, "INS": {"rho": 0.0, "p": 0.0, "n": 30}, "hyper_occupancy": {"rho": 0.0, "p": 0.0, "n": 30}},
  "hyper_state_top_rois": ["FRO_11", "THA_04", "INS_02", "FRO_03", "THA_07", "FRO_18", "INS_09", "THA_01", "FRO_05", "FRO_14", "THA_10", "INS_05", "FRO_08", "THA_03", "INS_07", "FRO_01", "THA_06", "INS_04", "FRO_16", "THA_09"]
}

Meanings: mean_f0 / mean_trophic_level / sd_trophic_level are group means of F0, of the subject's mean ROI trophic level and of the subject's within-subject SD of h (ddof = 0); occupancy / dwell_time_s / n_entries are group means; metastability / synchrony are group means. Values shown above are illustrative placeholders, not the answers.

/app/outputs/subject_states.csv with this exact header:
subject_id,group,crsr_score,mean_trophic_level,f0,hyper_occupancy,hyper_dwell_time_s,hyper_n_entries,metastability,synchrony
and 45 data rows (one per subject, same order as subjects.csv), for example:
sub-CNT01,CNT,,2.91,0.40,0.098,24.0,4,0.212,0.423

## Constraints

1. Exactly 41 windows per subject (1845 in total), window length 60 TR, step 6 TR, ridge penalty 20.0, edge fraction 0.20, 4 k-means clusters - these structural numbers must be exactly right.
2. Every reported number must be finite and produced offline by deterministic code (no network access, no seeds other than random_state=0 for KMeans).
3. Your results must reproduce the cross-group ordering of the reference pipeline (hyper-state occupancy and dwell time lowest in CNT, highest in UWS; metastability and synchrony follow the same ordering as the reference implementation).
4. CRS-R associations must reproduce the reference pipeline to within 0.12 (rank correlation units) and must carry the reference signs.
5. Group statistics must agree with the reference pipeline within 10 percent relative error (occupancy and correlations within 0.06 and 0.12 absolute, dwell times within 15 percent or 6 s, whichever is larger).
6. The hyper_state_top_rois list must share at least 15 of its 20 ROIs with the reference list and must be ordered by decreasing mean elevation.
7. Per-subject values in subject_states.csv must agree with the reference for at least 40 of 45 subjects on mean_trophic_level and f0, and for at least 35 of 45 subjects on the dynamic metrics.
8. subject_states.csv must have 45 rows with the exact header, crsr_score present only for patients, and hyper_occupancy inside [0, 1].

## Scoring

The grader re-runs the reference pipeline on the same cohort (regenerated from seed 20260730) and awards partial credit: score = fraction of independently checked constraints that pass. Checks cover (a) JSON/CSV schema and structural parameters, (b) static F0 and trophic-level group statistics, (c) hyper-state occupancy, dwell time and n_entries, (d) metastability and synchrony, (e) CRS-R rank correlations for FRO, THA, INS and hyper-state occupancy, (f) top-20 anatomical ROI list overlap, and (g) per-subject agreement for all 45 subjects.
