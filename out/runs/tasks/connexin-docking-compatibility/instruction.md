# Predict connexin heterotypic docking compatibility

You are given a standardized synthetic FETCH assay table for human connexin heterotypic pairs. The training table contains pair_id, fetch_score, and label (1 compatible, 0 incompatible). The test table contains pair_id and fetch_score but no labels.

Implement a faithful simplified version of the paper's data-driven thresholding: choose a FETCH score threshold from the training data that maximizes F1 for the compatible class, then classify each test pair as compatible if fetch_score >= threshold.

Input paths:
- /app/environment/data/training_labels.csv
- /app/environment/data/test_pairs.csv

Output path:
- /app/predictions.csv

Output format:
CSV with header pair_id,prediction. One row per test pair. prediction must be integer 0 or 1. Include all test pair IDs.

Scoring:
F1 on positive class (compatible) against hidden gold labels. Missing or malformed predictions are scored as 0.

Constraints:
- Use only files in /app/environment and the Python standard library.
- Do not access the network.
- Do not modify /tests or attempt to read hidden gold labels.
