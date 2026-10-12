# Review notes

- Data are synthetic but realistic (108 samples x 30 genes, log2 expression) generated deterministically from seed 20260729 using environment/data/gen_inputs.py; the same generator is used by the grader to recreate gold data.
- Shortcut probe: The agent cannot read gold answers because the grader recomputes all values from a protected copy of the data and the generator script only produces expression matrices, not analysis results.
- Tool chain: Python 3.11, numpy, pandas, scipy, scikit-learn. All analyses (PCA, hierarchical clustering, correlation networks) use canonical scientific Python packages.
- Known limitation: The synthetic data approximate the paper's qualitative findings (UFPM acute/adaptation, BC chronic/dysfunction, PS-NP chronic/biogenesis) but are not the original experimental values.
