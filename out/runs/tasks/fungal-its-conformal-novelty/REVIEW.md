# Review notes

- Data is synthetic but generated programmatically with a deterministic seed to mimic realistic ITS metabarcoding score distributions.
- Shortcut probe: the gold labels and thresholds are never written to the agent-visible files; the grader regenerates them from the protected seed in tests/gold_gen.py.
- Tool chain: Python 3.11 with numpy, pandas, scipy, scikit-learn. No external domain binaries are required because the task focuses on the statistical calibration of precomputed scores.
- Known limitation: the precomputed scores abstract away raw sequence alignment; this keeps the task focused on the conformal novelty flag.
