# Review notes

- Data are synthetic-but-realistic, generated deterministically from seed 20260728; 12 participants x 8 days x 8640 10-s samples (829,440 rows) across 8 sites.
- Shortcut probe: the gold summary is not present in the image; the grader regenerates it from a hidden copy of the generator in tests/gold_gen.py. The agent can read environment/data/gen_inputs.py, but must still implement the full filtering, metric, and bias pipeline.
- Tool chain: Python 3.11 with numpy/pandas; no domain binary is ground truth because the paper's analysis is a deterministic reanalysis of time-series data.
- Limitations: only 7 of the paper's 54 daily metrics are implemented; mixed-effects/GAMM modelling and hierarchical bootstrap precision analysis are omitted for deterministic scoring.
