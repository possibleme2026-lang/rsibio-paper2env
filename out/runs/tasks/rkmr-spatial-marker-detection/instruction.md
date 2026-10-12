# RKMR-lite: Rapid Kernel Machine Regression for Spatial Omics Marker Detection

You are given a spatial transcriptomics dataset with spot-level gene expression, spatial coordinates, and a phenotype score. Your task is to implement a simplified version of the RKMR framework (rapid kernel machine regression) to rank genes by approximate posterior inclusion probability (PIP) and select a parsimonious marker panel. The RKMR-lite algorithm combines spatial kernel adjustment, nonlinear kernel ridge regression, and cross-validated relevance scoring.

## Inputs
- `/app/data/spots.csv`: 240 rows, columns `spot_id,x,y,phenotype`.
- `/app/data/expression.csv`: 240 rows, columns `spot_id,G001,...,G100`.
No other files are needed.

## Task
Implement the following RKMR-lite algorithm exactly. Write your code in `/app/solution.py` (or any script) and produce the outputs described below. The grader will compare your output to a reference implementation and a hidden true marker set.

### RKMR-lite algorithm specification
1. **Load and align data**: Read `spots.csv` and `expression.csv`. Align rows by `spot_id`. Let `X` be the n x G gene expression matrix, `y` the phenotype vector, and `S` the n x 2 spatial coordinates.
2. **Standardize**: For each gene column, subtract the mean and divide by the population standard deviation (ddof=0). If std < 1e-12, set the column to zeros. Standardize `y` the same way.
3. **Spatial kernel**: Compute pairwise squared Euclidean distances among `S`. Let `sigma_s = median(sqrt(D2))`. Set spatial RBF lengthscale `l_s = sigma_s`. Compute the full spatial RBF kernel `K_s = exp(-D2 / (2 * l_s^2))`.
4. **Nyström approximation**: Use k-means on `S` with `n_clusters=80`, `random_state=0`, `n_init=10` to get landmark coordinates `L`. Compute `K_mm` among landmarks, add `1e-6` to its diagonal, and compute its Cholesky factor `L_chol`. Compute `K_nm` between spots and landmarks. Form `Z_s = K_nm @ inv(L_chol.T)`. (This approximates `K_s ≈ Z_s @ Z_s.T`.)
5. **Spatial adjustment**: For any vector `v` (either `y` or a gene column), regress it on `Z_s` with ridge `lambda_s = 1.0`: `alpha = solve(Z_s.T @ Z_s + lambda_s * I, Z_s.T @ v)`, then `residual = v - Z_s @ alpha`. Compute `r_y` as the residual of standardized `y`. For each gene `g`, compute `r_g` as the residual of the standardized `X[:, g]`.
6. **Nonlinear gene relevance**: For each gene `g`, compute an RBF kernel on `r_g`: `K_g[i,j] = exp(-(r_g[i]-r_g[j])^2 / (2 * l_g^2))`, where `l_g = 1.4826 * median(|r_g - median(r_g)|)`. If `l_g < 1e-6`, set `l_g = 1e-6`. Use kernel ridge regression to predict `r_y` from `K_g` with ridge `lambda_g = 1.0` under 5-fold cross-validation (`KFold(n_splits=5, shuffle=True, random_state=0)`). Compute the cross-validated R^2: `1 - sum((r_y - pred)^2) / sum((r_y - mean(r_y))^2)`. If the denominator is 0, set R^2 to 0. Set `score_g = max(0, R^2)`.
7. **PIP calibration**: `PIP_g = score_g / (score_g + 0.05)`. Clip to [0, 0.999].
8. **Ranking and panel**: Sort genes by `PIP_g` descending, with ties broken by gene name ascending (lexicographic). The panel consists of all genes with `PIP_g >= 0.5`. If the panel has fewer than 3 genes, take the top 3. If it has more than 15, take the top 15. The panel order follows the same sorting.

## Output
Write exactly two files:
1. `/app/output/ranked_pips.csv`: header `gene,pip`, followed by 100 rows (one per gene in `expression.csv`), sorted by PIP descending, ties by gene name ascending.
2. `/app/output/panel.txt`: selected panel genes, one per line, sorted by PIP descending (ties by gene name ascending).

Example (format only):
```
gene,pip
G007,0.912
G023,0.887
...
```

## Constraints
1. Both output files must exist at the specified paths.
2. `ranked_pips.csv` must contain exactly the header `gene,pip` and 100 data rows, with each gene from `expression.csv` appearing exactly once.
3. All `pip` values must be numeric and within [0, 1].
4. Rows in `ranked_pips.csv` must be sorted by `pip` descending, with ties broken by gene name ascending.
5. `panel.txt` must contain between 3 and 15 genes, all present in `expression.csv`, with no duplicates, and sorted by PIP descending (ties by gene name ascending).
6. Your PIP ranking must have Spearman correlation >= 0.90 with the reference RKMR-lite PIPs (partial credit if >= 0.75).
7. Your selected panel must achieve F1 >= 0.60 against the hidden true marker set (partial credit if >= 0.40).

## Scoring
The grader checks each constraint independently. Score = (sum of points) / 7, where constraints 6 and 7 can earn 0.5 points for partial fulfilment. The score is written to `/logs/verifier/reward.json`.
