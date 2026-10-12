# Review notes

- Synthetic data was generated to mimic CHO epigenetic profiles with realistic sizes (1000 train, 400 test, 350 features) and missing values.
- The label-generating function is only in the verifier's tests/gold_gen.py; the environment generator is deleted after image build, so the agent cannot read the label rule.
- Tool chain: scikit-learn, pandas, numpy, shap installed. Grader uses scikit-learn metrics and recomputes gold from a protected seed.
- Thresholds are calibrated to be achievable but non-trivial, matching the paper's reported F1 of ~0.59.
