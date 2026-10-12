# Variational EM for a Mixture of Coupled Amino-Acid Models

## Context
Protein coevolution studies how substitutions at different sites are correlated. Pooling all site pairs into a single model can average out opposing signals (Simpson's paradox). A mixture of coupled Potts models can separate distinct coevolutionary environments. In this task you will implement a variational EM algorithm to fit a 2-component mixture of coupled Potts models to paired-site observations from a structural alignment, and report the fitted parameters, per-pair responsibilities, and ELBO history.

## Inputs
- `/app/data/alignment.fasta`: 200 protein sequences, each 80 residues.
- `/app/data/contacts.tsv`: 40 candidate coevolving pairs. Columns: `pair_id`, `site_i`, `site_j` (1-indexed).
- `/app/data/alphabet_map.tsv`: maps the 20 amino acids to 4 coevolution groups (1-4).

## Task
1. Parse the FASTA and map each amino acid to its group (1-4) using `alphabet_map.tsv`. Treat `X` as missing.
2. For each contact pair, extract the two columns and build a list of observed `(a,b)` group pairs, dropping any observation where either residue is missing. Convert group numbers to 0-based indices (1→0, 2→1, 3→2, 4→3) for the model.
3. Implement the following EM algorithm for a 2-component mixture of coupled Potts models:
   - Model: For component `c ∈ {1,2}`, parameters `h_c ∈ R^4` with `sum(h_c)=0`, and `J_c ∈ R^{4×4}` symmetric with zero diagonal. The probability of an observed pair `(a,b)` is
     `P_c(a,b) = exp(h_c(a)+h_c(b)+J_c(a,b)) / Z_c`, where `Z_c = Σ_{x,y} exp(h_c(x)+h_c(y)+J_c(x,y))`.
   - Mixture: `P(a,b) = Σ_c π_c P_c(a,b)`.
   - For a pair `p` with observations `{(a_s,b_s)}`, the log-likelihood under component `c` is `L_{p,c} = Σ_s log P_c(a_s,b_s)`.
   - Initialization: `π = [0.5, 0.5]`, `h_c = [0,0,0,0]`, `J_c = zeros(4,4)` for both components.
   - Iterate for up to 100 iterations or until `|ELBO - prev_ELBO| < 1e-6`:
     - **E-step**: Compute responsibilities `r_{p,c} = π_c exp(L_{p,c}) / Σ_{c'} π_{c'} exp(L_{p,c'})`. Use log-sum-exp for numerical stability.
     - **M-step**: Update `π_c = (1/M) Σ_p r_{p,c}`. For each `c`, minimize the negative expected complete log-likelihood `Q_c = - Σ_p r_{p,c} L_{p,c}` with respect to `h_c` and `J_c`. Use `scipy.optimize.minimize` with `method='L-BFGS-B'`, `jac=True`, `tol=1e-8`, and the previous `h_c, J_c` as the initial guess. Parameterize `h_c` by 3 free parameters `u1,u2,u3` with `h_c = [u1,u2,u3,-u1-u2-u3]`, and `J_c` by 6 free parameters for the upper triangle. Provide analytic gradients.
     - Compute `ELBO = Σ_p Σ_c r_{p,c} ( log π_c + L_{p,c} - log r_{p,c} )`. Record it.
   - Stop when convergence criterion is met.

4. Write the outputs to `/app/results/`:
   - `params.json`: JSON with keys `pi` (list of 2 floats), `h` (2x4 list of lists), `J` (2x4x4 list of lists), `elbo` (float, final ELBO), `iterations` (int, number of EM iterations run).
   - `responsibilities.tsv`: tab-separated with header `pair_id\tr1\tr2`. One row per pair, sorted by `pair_id`, with `r1+r2=1`.
   - `elbo_history.tsv`: tab-separated with header `iteration\telbo`. One row per EM iteration (1-indexed).

## Constraints
1. All output files must exist and be parseable.
2. `params.json` must contain keys `pi`, `h`, `J`, `elbo`, `iterations` with correct shapes: `pi` length 2, `h` shape (2,4), `J` shape (2,4,4).
3. Mixture weights `pi` must sum to 1 (tolerance 1e-6) and each must be >= 0.01.
4. Each `h` vector must sum to 0 (tolerance 1e-6).
5. Each `J` matrix must be symmetric (abs diff < 1e-6) and have zero diagonal (abs < 1e-6).
6. `elbo_history.tsv` must have columns `iteration` and `elbo`, and ELBO must be non-decreasing across iterations (differences >= -1e-6).
7. The final `elbo` in `params.json` must match the last ELBO in `elbo_history.tsv` (abs diff < 1e-6).
8. Parameter recovery: after optimal matching of components (permutation of the two components), the RMSE between your `h` and the reference `h` must be < 0.1, and the RMSE between your `J` and the reference `J` must be < 0.1.
9. Responsibilities: `r1` and `r2` must be between 0 and 1, sum to 1 per pair (tolerance 1e-6), and the mean absolute error compared to the reference responsibilities must be < 0.05.
10. Assignment accuracy: after optimal component matching, the argmax component assignment must match the reference for at least 90% of pairs.
11. Final ELBO must be within 1e-3 relative error of the reference ELBO.

## Scoring
The grader checks the 11 constraints above. Each constraint is worth 1/11 of the total score. The final score is the fraction of constraints passed. Partial credit is given per constraint.
