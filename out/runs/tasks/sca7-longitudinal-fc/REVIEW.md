# Review notes

- Data is synthetically generated with a deterministic seed (20260215) to mimic a longitudinal SCA7 fMRI cohort (16 SCA7, 16 controls, 3 visits, 8 networks, 28 edges). The generator is included at environment/data/gen_inputs.py and is run during image build.
- Shortcut probe: The gold standard is not present in the agent-readable environment. The grader recomputes expected results from a protected seed in tests/gold_gen.py. The agent cannot read tests/.
- Tool chain: statsmodels for linear mixed-effects models, scikit-learn for classification, pandas/numpy/scipy for data handling.
- Known limitations: The synthetic data has stronger effects than the original paper to ensure deterministic and significant results; the exact numerical values are not intended to replicate the paper's statistics but to benchmark the analysis workflow.
