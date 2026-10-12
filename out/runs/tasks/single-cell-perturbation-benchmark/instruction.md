# Single-Cell Perturbation Response: Benchmark Metric Reproduction

You are a computational biologist evaluating perturbation-response prediction models, as in Cheng et al. (2026). The benchmark compares methods on expression-level accuracy, delta correlation, differential-expression (DE) recovery, and distributional similarity. Your job is to implement the evaluation pipeline that reproduces the paper's metric suite and run it on two provided tasks.

## Inputs
- `/app/data/task1/` and `/app/data/task2/` each contain:
  - `control.h5ad`: control cells (500 cells x 300 genes), log1p-normalized expression.
  - `perturbations.json`: list of perturbation names and gene names.
  - `true/<pert>.h5ad`: true perturbed cells (200 cells x 300 genes) for each perturbation.
  - `methods/<method>/pred_<pert>.h5ad`: predicted cells (200 cells x 300 genes) for each method and perturbation.
- Methods: `mean`, `linear`, `mlp`, `perturbnet`.
- Task 1: unseen single-gene perturbations (6 perturbations).
- Task 2: combinatorial perturbations (5 perturbation pairs).

## Task
Write a reusable Python script `/app/solution/evaluate.py` that takes `--input <dir>` (a directory containing `task1/` and `task2/` subdirs) and `--output <file>`. For each task, compute the following per method per perturbation and then aggregate.

For each perturbation:
1. Mean expression vectors: mu_true (true perturbed cells), mu_pred (predicted cells), mu_ctrl (control cells).
2. MSE = mean((mu_pred - mu_true)^2) over genes.
3. Delta Pearson = Pearson correlation between (mu_true - mu_ctrl) and (mu_pred - mu_ctrl) across genes.
4. DE recovery F1:
   - True DE genes: Welch's t-test (equal_var=False) between true perturbed cells and control cells on log1p expression. Compute log2 fold change as log2((mean(expm1(perturbed)) + 1) / (mean(expm1(control)) + 1)). Apply Benjamini-Hochberg FDR across genes. Call DE if FDR < 0.05 and |log2FC| > 0.5.
   - Predicted DE genes: same procedure using predicted cells vs control.
   - F1 = 2*precision*recall/(precision+recall). If true DE set is empty, F1 = 1 if predicted DE set is also empty, else 0.
5. Distributional similarity:
   - Fit PCA on control cells with n_components=min(20, n_genes, n_control_cells), random_state=0.
   - Project true perturbed cells and predicted cells.
   - Energy distance = 2*mean(pairwise Euclidean distances between true and pred) - mean(within true) - mean(within pred).
   - dist_sim = 1 / (1 + energy_distance).

Aggregate across perturbations: take the arithmetic mean of each metric per method.

Then compute for each task:
- `ranking` for each metric: sort methods by MSE ascending (lower is better), and by delta_pearson, de_f1, dist_sim descending (higher is better). Ties broken alphabetically.
- `overall_score` per method: min-max normalize each metric across methods. For MSE, use 1 - (mse - min_mse)/(max_mse - min_mse + 1e-12). For others, (value - min)/(max - min + 1e-12). Average the four normalized values.
- `best_method`: method with highest overall_score; tie broken alphabetically.

Write the results to `/app/output/metrics.json` with this exact structure (using double quotes in the actual JSON):
{
  "task1": {
    "methods": {
      "mean": {"mse": float, "delta_pearson": float, "de_f1": float, "dist_sim": float},
      ...
    },
    "ranking": {
      "mse": ["method1", ...],
      "delta_pearson": [...],
      "de_f1": [...],
      "dist_sim": [...]
    },
    "overall_score": {"mean": float, ...},
    "best_method": "method"
  },
  "task2": { ... }
}

## Constraints
1. Output file must be at `/app/output/metrics.json` and be valid JSON.
2. Must include both `task1` and `task2` with exactly the methods `mean`, `linear`, `mlp`, `perturbnet`.
3. MSE values must be >= 0; delta_pearson in [-1, 1]; de_f1 in [0, 1]; dist_sim in (0, 1].
4. DE rule must use Welch's t-test, BH FDR < 0.05, |log2FC| > 0.5.
5. PCA must use control cells, n_components=min(20, n_genes, n_control_cells), random_state=0.
6. Energy distance must use Euclidean distances on PCA projections.
7. Ranking lists must contain each method exactly once, sorted correctly.
8. `overall_score` must use min-max normalization as specified.
9. `/app/solution/evaluate.py` must be a reusable script accepting `--input` and `--output`.

## Scoring
The grader runs your `/app/solution/evaluate.py` on a hidden dataset and compares the output to gold values computed independently. There are 15 constraints: file validity, presence of task1/task2 methods, each of 4 metrics for task1, each of 4 metrics for task2, rankings for task1, rankings for task2, overall_score+best_method for task1, overall_score+best_method for task2. Score = fraction passed. Partial credit is given.
