# Review notes

- Data is synthetic but realistic: 4 species, 2 chromosomes each of 300 kb, 100 genes per species, 120 ortholog pairs across 6 species pairs.
- Shortcut probe: the agent cannot read the grader or gold generation code (tests/) during solve time; the gold is regenerated deterministically from a protected seed in tests/gold_gen.py.
- Tool chain: Python 3.11 with numpy, scipy, pandas, scikit-learn, statsmodels, biopython. No external domain binary required because the G4 motif definition is explicitly specified.
- Known limitation: the synthetic dataset is smaller than the paper’s 198 genomes; the task focuses on the computational method (G4 motif scanning, compartment definitions, CAI-controlled logistic regression) rather than exact paper numbers.
