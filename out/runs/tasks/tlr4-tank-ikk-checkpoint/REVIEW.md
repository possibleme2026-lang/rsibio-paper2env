# Review notes

- Data substitution: synthetic time-course data generated from a simplified ODE model of TLR4 signaling that captures the IRAK1-TRAF6-IKKε checkpoint. The generator is run at Docker build time and removed from the final image.
- Shortcut probe: the gold parameters and generator are deleted from the image; only noisy data and the baseline model remain. The grader regenerates noise-free ground truth from a protected seed in tests/gold_gen.py.
- Tool chain: numpy, scipy, pandas, scikit-learn. No external domain binaries are required because the task focuses on parameter inference from provided ODEs.
- Known limitations: the model is a simplified representation of the full rule-based TLR4 model; it abstracts molecular details while preserving the core checkpoint topology and quantitative phenotypes.
