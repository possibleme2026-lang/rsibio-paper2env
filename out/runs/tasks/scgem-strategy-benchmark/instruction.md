# Benchmark scGEM construction strategies from single-cell expression

## Context

Single-cell genome-scale metabolic models (scGEMs) are built by projecting a single-cell expression
profile onto a genome-scale metabolic model (GEM). Gene expression is first preprocessed, then
aggregated to reaction-level scores through the GEM gene-protein-reaction (GPR) rules, an expression
threshold decides which reactions stay active, and the resulting per-cell reaction set is the
context-specific model. The construction choices - data preprocessing, model extraction method (MEM)
and expression threshold - strongly change model structure and downstream predictions, which is why
they have to be benchmarked explicitly.

You are rebuilding such a benchmark on a curated 300-reaction GEM and an 80-cell x 600-gene scRNA-seq
count matrix. You must run the complete strategy grid, emit every per-cell model, score each strategy
on three axes (accuracy, robustness, specificity) and report the winning strategy.

## Inputs

All files live under `/app/data` and are tab-separated with a header row:

- `expression_counts.tsv` - raw UMI counts, 80 rows (cells `CELL001`...`CELL080` in exactly this order)
  by 600 columns (genes `G0001`...`G0600`).
- `expression_counts_perturbed.tsv` - the same matrix after a 10% dropout perturbation (10% of the
  non-zero entries were set to zero). Same row/column order.
- `cell_metadata.tsv` - `cell_id`, `cell_type` (hepatocyte / T_cell / macrophage / neuron, 20 cells
  each), `dataset`.
- `model_reactions.tsv` - 300 reactions in the canonical model order, columns `reaction_id`,
  `reaction_name`, `gpr`, `subsystem`, `metabolites`, `essential`. `gpr` holds expressions such as
  `(G0134 and G0401) or G0288`; exactly 20 reactions have an empty `gpr`; exactly 8 reactions have
  `essential = 1`.
- `reference_models.tsv` - long format `cell_type`, `reaction_id`: the bulk-derived reference active
  reaction set of each cell type (used only for the accuracy axis).
- `strategy_grid.tsv` - `strategy_id`, `preprocessing`, `mem`, `param`: the 18 strategies to evaluate.

## Pipeline specification

Follow this specification exactly; your numbers are compared against an independent reference
implementation of the very same specification.

### 1. Preprocessing
Apply to the raw integer count matrix `X` (cells x genes); `V` is the resulting float matrix.
- `cpm`      : per cell, `V = X / sum(cell) * 1e6`.
- `cpm_log2` : per cell, `V = log2(X / sum(cell) * 1e6 + 1)`.
- `zscore`   : per gene, `V = (X - mean_over_cells) / sd_over_cells` using the population standard
  deviation (`ddof = 0`); if `sd == 0` the whole gene column becomes 0.
If a cell total is 0, use 1 instead.

### 2. GPR parsing and evaluation
GPR strings are built from gene IDs, `and`, `or` and parentheses. `and` binds tighter than `or`, so
`a and b or c` means `(a and b) or c`. A reaction with an empty `gpr` is gene-independent and is active
in every cell for every strategy.
- continuous (fuzzy) MEM: the value of a gene in a cell is `V[cell, gene]` (0 if that gene is not a
  column of the expression matrix); an `and` node returns the minimum of its operands, an `or` node
  the maximum.
- boolean MEM: first binarize every gene: gene `g` is ON in cell `c` when `V[c, g]` is strictly greater
  than the `param`-th percentile of that gene's 80 values (linear interpolation, i.e. numpy's default
  `np.percentile`); then evaluate the GPR with ordinary logical `and` / `or`.

### 3. Reaction activity and model assembly
- fuzzy MEM: for cell `c`, compute the continuous score of every reaction with a non-empty GPR;
  `tau_c` is the `param`-th percentile (linear interpolation) of those scores; such a reaction is
  active if and only if its score is >= `tau_c`.
- boolean MEM: such a reaction is active if and only if its boolean GPR evaluates to true.
- in both cases reactions with an empty GPR and reactions with `essential = 1` are always active.
A cell's model is its 300-bit activity vector in `model_reactions.tsv` row order.

### 4. Evaluation per strategy
Let `M(s, c)` be the active reaction set of cell `c` under strategy `s` built from
`expression_counts.tsv`, and `M'(s, c)` the same built from `expression_counts_perturbed.tsv`.
- `accuracy`  = mean over the 80 cells of the Jaccard index between `M(s, c)` and the reference set of
  `cell_type(c)` from `reference_models.tsv`.
- `robustness` = mean over the 80 cells of the Jaccard index between `M(s, c)` and `M'(s, c)`.
- `specificity` = (mean Jaccard over all pairs of distinct cells with the same cell type) minus
  (mean Jaccard over all pairs of distinct cells with different cell types), computed from the 80x80
  pairwise Jaccard matrix of `M(s, .)`.
- `mean_model_size` = mean number of active reactions per cell.
- `composite = 0.5 * accuracy + 0.3 * robustness + 0.2 * specificity`.
Jaccard index of two empty sets is defined as 1.

## Task
1. Read the grid and load the model, the metadata, the reference sets and both count matrices.
2. Implement the preprocessing, the GPR parser/evaluator and the two model-extraction methods.
3. Build the 300-bit per-cell models for all 18 strategies, for both the original and the perturbed
   matrix.
4. Compute the four summary numbers and the composite for every strategy.
5. Report the strategy with the largest composite (ties broken by the smaller `strategy_id`).

## Constraints
1. Exactly one model file per strategy must exist at `/app/output/models/<strategy_id>.tsv` for all 18
   strategies of the grid.
2. Each model file must have a header row whose first field is `cell_id` followed by all 300 reaction
   IDs of `model_reactions.tsv`, then 80 data rows in exactly the `expression_counts.tsv` cell order;
   every value must be exactly `0` or `1`.
3. Every per-cell model must contain all 20 empty-GPR reactions and all 8 essential reactions.
4. Every per-cell model must contain at least 5 active reactions.
5. `/app/output/strategy_metrics.tsv` must contain all 18 strategies and the columns `strategy_id`,
   `preprocessing`, `mem`, `param`, `mean_model_size`, `accuracy`, `robustness`, `specificity`,
   `composite`, sorted by `strategy_id`.
6. `accuracy`, `robustness` and `specificity` must agree with the reference implementation to within
   2e-2 absolute, and `mean_model_size` to within 0.5.
7. Within each strategy, the mean per-cell Jaccard index between your models and the reference models
   must be at least 0.98.
8. `/app/output/selection.json` must report the argmax of `composite`, ties broken by the smaller
   `strategy_id`, with its composite value.

## Output
- `/app/output/models/<strategy_id>.tsv` - 18 binary matrices, header `cell_id` + 300 reaction IDs,
  80 data rows (`CELL001` ... `CELL080`), tab-separated `0`/`1`.
- `/app/output/strategy_metrics.tsv` - 18 rows sorted by `strategy_id`. Worked example row:
  `S01	cpm	fuzzy	50	168.425000	0.612300	0.745100	0.118800	0.553440`
- `/app/output/selection.json` - e.g. `{"best_strategy": "S11", "composite": 0.553440}`.

## Scoring
The grader independently recomputes the whole pipeline from `/app/data` and awards partial credit over
seven constraints: C1 output format (0.10), C2 per-cell model agreement with the reference models
(0.28, fraction of the 18 strategies reaching mean Jaccard >= 0.98), C3 accuracy + mean_model_size
(0.10), C4 robustness (0.10), C5 specificity (0.08), C6 selected strategy and composite (0.22), C7
mandatory-reaction and non-degenerate models (0.12). `score` and `reward` are the weighted sum.
