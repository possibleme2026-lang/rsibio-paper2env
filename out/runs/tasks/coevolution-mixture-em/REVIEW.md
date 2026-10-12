# Review notes

- Data is synthetic but realistic: 200 sequences x 80 sites, 40 disjoint contact pairs, 3% missing data. Generated deterministically by environment/data/gen_inputs.py.
- The task requires implementing a full variational EM algorithm with analytic gradients and scipy optimization. The gold standard is computed by the grader's reference implementation on the same data.
- Shortcut probe: the agent cannot read the reference implementation or gold outputs; they must implement the algorithm from the specification. The data generator does not contain the EM answer.
- Tool chain: numpy and scipy are installed. The grader uses scipy.optimize.minimize with L-BFGS-B and analytic gradients.
- Known limitations: The model is a simplification of the paper's CTBN; it uses a pairwise Potts model with shared component parameters across pairs, which is a tractable approximation suitable for a benchmark task.
