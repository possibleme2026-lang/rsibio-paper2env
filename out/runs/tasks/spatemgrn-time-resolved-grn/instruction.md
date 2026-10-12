# Time-resolved spatial-unit-specific gene regulatory network inference

You are given multi-timepoint spatial transcriptomics data from a simulated tissue. The data contains gene expression measurements for spatial spots across 3 timepoints (0, 1, 2). Each spot is assigned to one of 4 spatial quadrants (Q1-Q4). Your task is to implement a simplified version of the SpaTemGRN pipeline to infer time-resolved, spatial-unit-specific gene regulatory networks (GRNs) among 20 genes.

## Inputs
- `/app/data/expression.csv`: long-format table with columns `timepoint, spot_id, gene_0, gene_1, ..., gene_19`. Values are log1p-transformed expression levels.
- `/app/data/coordinates.csv`: columns `timepoint, spot_id, x, y, spatial_unit`. `spatial_unit` is one of `Q1`, `Q2`, `Q3`, `Q4`.
- `/app/data/genes.txt`: one gene name per line (20 genes).

The number of spots per spatial unit differs across timepoints: timepoint 0 has 50 spots per unit, timepoint 1 has 55, timepoint 2 has 60.

## Task

Implement a Python script that, for each timepoint `t` in {0,1,2} and each spatial unit `u` in {Q1,Q2,Q3,Q4}, infers an undirected GRN among the 20 genes using the following steps:

1. **Data preparation**: Extract spots for `(t,u)`. Build expression matrix `X` (rows = spots, columns = genes). Z-score each gene column (subtract mean, divide by standard deviation across spots in this unit; if std=0, set column to 0).

2. **Spatial smoothing**: For each spot, find its 5 nearest neighbors (Euclidean distance in `(x,y)`) among spots in the same `(t,u)`. Include the spot itself. Compute a smoothed expression vector for each gene as the mean over the spot and its 5 neighbors. This yields matrix `X_smooth` of shape `(N_t, 20)`.

3. **Temporal borrowing**: If `t > 0`, load the smoothed matrix `X_prev` for the same unit at timepoint `t-1`. If the number of rows differs from `N_t`, randomly sample `N_t` rows with replacement from `X_prev` using a fresh `numpy.random.default_rng(42)` for each `(t,u)` pair. Concatenate vertically: `A = vstack([X_smooth, X_prev_sampled])`. For `t = 0`, set `A = X_smooth`.

4. **Network inference**: Compute the sample covariance matrix of `A`. Fit a graphical lasso model using `sklearn.covariance.GraphicalLasso(alpha=0.1, max_iter=100, tol=1e-4)`. Let `Theta` be the estimated precision matrix. For each gene pair `(i,j)` with `i < j`, compute the absolute partial correlation:
   `rho_ij = abs(-Theta[i,j] / sqrt(Theta[i,i] * Theta[j,j]))`.
   Let `scores` be the vector of `rho_ij` for all `190` undirected pairs. Normalize: `scores = scores / max(scores)`; if `max(scores) == 0`, set all scores to 0.

5. **Output**: For each `(t,u)`, write all gene pairs with normalized score > 0.05. Sort rows by `timepoint`, then `spatial_unit`, then `score` descending. Write to `/app/output/edges.csv` with columns:
   `timepoint, spatial_unit, source, target, score`
   - `source` and `target` are gene names in canonical order: `source < target` lexicographically.
   - `score` is rounded to 6 decimal places.

## Constraints
1. Output file must exist at `/app/output/edges.csv` with exactly those five columns and no extra columns.
2. All 12 `(timepoint, spatial_unit)` combinations must be present (3 timepoints × 4 units).
3. Each `(timepoint, spatial_unit)` network must contain at least 10 and at most 100 edges.
4. Scores must be in `[0, 1]`. No self-loops (`source != target`). No duplicate undirected edges within a network (i.e., `frozenset({source, target})` must be unique per `(t,u)`).
5. Rows must be sorted by `timepoint` ascending, then `spatial_unit` (`Q1` < `Q2` < `Q3` < `Q4`), then `score` descending.

## Scoring
The grader recomputes the reference networks using the exact method above (hidden implementation) and compares your edge scores to the reference via AUROC per network. Partial credit:
- 40% for constraints 1–5 (each worth 8%).
- 60% for mean AUROC across the 12 networks.
Final score = 0.4 * (constraints_passed / 5) + 0.6 * mean_AUROC. Reward equals this score.
