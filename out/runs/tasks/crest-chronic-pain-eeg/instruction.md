# CREST chronic-pain inference from synthetic resting-state EEG spectrograms

Implement a simplified CREST spatial-transformer inference pipeline for chronic-pain status from resting-state EEG spectrogram features.

Inputs (generated in the container image):
- /app/data/crest_model.json: frozen simplified model weights.
- /app/data/spectrograms.csv: one row per subject; columns subject_id and r{region}_{band} for regions 0-55 and bands delta, theta, alpha, beta, gamma.

Output:
- /app/output/predictions.csv with header subject_id,pain_probability.
- Include all 125 subjects sub-000 through sub-124.
- pain_probability must be a finite float. Monotonic scores are accepted.

Required computation:
1. For each region r, compute region_logit = sum_b band_weights[b] * feature[r,b] + region_weights[r].
2. Use a cyclic ring graph over the 56 regions: graph_logit[r] = (region_logit[r] + region_logit[(r-1) mod 56] + region_logit[(r+1) mod 56]) / 3.
3. Compute final_logit = (1 - graph_weight) * mean(region_logit) + graph_weight * mean(graph_logit).
4. Write sigmoid(final_logit) as pain_probability, or any monotonic transform of final_logit.

Scoring:
The verifier computes AUROC between your subject-level scores and hidden chronic-pain labels. The hidden labels are not present in the environment.

Constraints:
- No internet access.
- Do not look for labels in /tests; they are unavailable during solving.
- Use the provided synthetic features; they replace non-redistributable real EEG data.
