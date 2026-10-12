# Review notes

- Synthetic multi-omics data generated programmatically with a fixed seed in environment/data/gen_inputs.py; the generator is removed from the image after build so the agent cannot read the latent effect assignments.
- Shortcut probe: hardcoding public-data metrics or species lists will fail because the grader runs the agent script on a held-out dataset generated with a different seed (2027) and checks internal consistency of CSV outputs.
- Tool chain: Python 3.11 with numpy, pandas, scipy, scikit-learn, and scikit-bio (for PERMANOVA and Bray–Curtis distances).
- Known limitation: the synthetic signal sizes were calibrated to reproduce the paper’s qualitative findings (facility > treatment) rather than exact numeric values from the original study.
