# Review notes

- Data is synthetic but realistic: 40 background proteins (80-160 residues) and 9 target proteins (100-150 residues) generated deterministically by environment/data/gen_inputs.py with seed 42.
- Shortcut probe: the agent cannot read the grader or hidden labels; the gold scores are recomputed from input by tests/gold_gen.py. The sink AUC uses hidden labels derived from the gold sink channel.
- Tool chain: Python 3.11 with numpy, scipy, pandas, scikit-learn, biopython. The grader uses the same stack to recompute gold.
- Known limitations: The transfer entropy formula is a simplified GNM-based surrogate for the paper's net TE; the alphabet word definition is a simplification of the paper's contact cliques.
