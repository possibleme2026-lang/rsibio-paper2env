# Review notes

- Data are synthetic but realistic: 6000 genes x 2000 reference cells and 1200 cultured cells across three conditions, generated deterministically by environment/data/gen_inputs.py with fixed seed. The generator is visible to the agent but does not contain the expected pipeline outputs.
- Shortcut probe: an agent might try to hardcode assignments or read latent labels from the generator. The grader recomputes expected assignments from raw counts using the specified Pearson-correlation-to-centroid pipeline, so shortcut answers fail.
- Tool chain: Python 3.11 with numpy, scipy, pandas, scikit-learn, and scanpy installed. The expected pipeline uses standard scientific Python.
- Known limitations: simulated expression profiles are simplified and lack real batch effects, doublets, or full biological complexity; the task tests computational mapping and DE logic rather than wet-lab biology.
