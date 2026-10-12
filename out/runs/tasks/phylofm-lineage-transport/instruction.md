# Lineage-constrained reconstruction of gene-program dynamics

You are given a synthetic-but-realistic single-cell embryogenesis dataset with lineage barcodes, a cell-fate-anchor topology, and a precomputed latent embedding. Your job is to reconstruct lineage-constrained population transport and downstream fate dynamics, as described in the PhyloFM framework.

## Inputs
- `/app/data/counts.npy`: 1500 x 24 integer count matrix (cells x genes). Cells are ordered by timepoint (300 cells per timepoint, 5 timepoints).
- `/app/data/cell_metadata.csv`: columns `cell_id`, `timepoint` (0-4), `clone_id` (0-179).
- `/app/data/latent_pca.npy`: 1500 x 10 precomputed PCA embedding of the cells. Use this as the latent geometry for transport.
- `/app/data/anchors.npy`: 3 x 10 cell-fate-anchor vectors in the same latent space.
- `/app/data/lineage_tree.json`: lineage tree of clones with three main branches (branch IDs 0,1,2). Each branch contains 60 clones.
- `/app/data/gene_modules.json`: mapping from module name to list of gene indices.

## Task
Implement a lineage-constrained flow-matching reconstruction pipeline:

1. **Transport**: For each consecutive timepoint pair (t=0..3), solve an entropy-regularized optimal transport problem with uniform marginals. The cost between cell i at time t and cell j at time t+1 is:
   `C[i,j] = ||z_i - z_j||^2 + lambda * (1 - same_clone(i,j))`
   where `z` is the 10D latent embedding, `same_clone(i,j)=1` if `clone_id_i == clone_id_j` else 0, `lambda=5.0`. Use entropy regularization `epsilon=0.05`. Save the coupling matrices `T_0..T_3`.

2. **Fate probabilities**: For each cell i at time t, propagate its mass forward through the transport matrices to the final timepoint t=4. At t=4, compute distances to the 3 anchors and convert to probabilities via `softmax(-d / tau)` with `tau=1.0`. This yields `P[i,k]` for each cell and fate `k`.

3. **Commitment times**: For each branch k (0,1,2), consider the cells at t=4 whose maximum fate probability is k. For each timepoint t, compute the mean fate probability for branch k across those cells. The commitment time is the smallest t where this mean >= 0.7. If never, use 4. Report confidence as the mean probability at that t.

4. **Module dynamics**: For each branch k and timepoint t, compute the fate-weighted average expression of each gene module: `sum_i P[i,k](t) * expression[i, genes] / sum_i P[i,k](t)`. Save the mean across genes in the module.

## Constraints
1. Transport matrices must be valid couplings: nonnegative, rows sum to 1, columns sum to 1 (tolerance 1e-6), and cover all cells.
2. Lineage consistency: For each transition, the average transport mass between same-clone cells must be at least 3x the global average mass for at least 90% of clones.
3. Fate probabilities must be nonnegative and sum to 1 for each cell. Macro AUROC against hidden true fate labels must be >= 0.85.
4. Commitment times must be within ±1 timepoint of the hidden gold for at least 2 of 3 branches, and the ordering of branches must match the lineage tree.
5. Module dynamics: Pearson correlation with hidden gold trajectories must be >= 0.7 on average across the 4 modules, with at least 3/4 modules having r >= 0.6.
6. Transport accuracy: normalized Frobenius error vs. hidden gold transport must be <= 1e-4 on average.

## Output
Write the following files to `/app/output/`:
- `transport.npz`: keys `T_0`, `T_1`, `T_2`, `T_3`. Each is a 2D numpy array of shape (n_t, n_{t+1}).
- `fate_probabilities.csv`: columns `cell_id`, `timepoint`, `fate_0`, `fate_1`, `fate_2`.
- `commitment_times.csv`: columns `branch`, `commitment_time`, `confidence`.
- `module_dynamics.csv`: columns `branch`, `timepoint`, `module`, `mean_expression`.

## Scoring
The grader checks each of the 6 constraints independently. Score is the fraction of constraints passed (partial credit). All checks use a deterministic reference implementation against hidden gold.
