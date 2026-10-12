# Structure-Constrained Intrinsic Timescales Across Tasks

You are given synthetic, anonymized persistence measurements for eight conditions and four brain regions. The goal is to estimate intrinsic neural timescales (INT), compute structure-constrained INT coupling, and identify the rest condition.

Inputs:
- /app/environment/data/autocorr.csv: columns condition, region, r1. r1 is the lag-1 autocorrelation of the fMRI-derived signal, in (0,1).
- /app/environment/data/sc_distance.csv: columns region, distance. distance is a standardized structural-connectome distance for that region.

Required computation:
1. For each region and condition, estimate INT tau using tau = -1 / log(r1).
2. For each condition, compute SC-INT coupling as the Pearson correlation between tau values and distance values across the four regions.
3. The rest condition is expected to have the strongest positive SC-INT coupling; task conditions should have weaker coupling.
4. Write /app/output.json with labels mapping every condition ID to either rest or task. You may also include rest_condition and coupling.

Output format:
{'labels': {'<condition_id>': '<rest|task>', ...}, 'rest_condition': '<condition_id>', 'coupling': {'<condition_id>': <float>, ...}}

Scoring:
The verifier computes F1 for the rest class using hidden gold labels. Missing or malformed output receives score 0. Exact correct labeling receives score 1.

Constraints:
- Use only files under /app/environment.
- Do not use network access.
- Do not modify /tests.
