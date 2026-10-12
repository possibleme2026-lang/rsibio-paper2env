# Review notes

- Synthetic-but-realistic data generated at container build time from a fixed seed (20260802) via environment/data/gen_inputs.py; the generator is deleted before the agent starts. tests/gold_gen.py mirrors the generator and derives reference outputs deterministically.
- Shortcut probe: the gold answer is never present in /app; the grader recomputes HSA scores, 4PL fits, and Western-blot fold changes from raw data. Hard-coding is defeated by exact row/key checks and independent recomputation.
- Tool chain: Python 3.11 with numpy, pandas, scipy (curve_fit for 4PL). No proprietary binaries are required; the domain ground truth is the 4PL and HSA definitions specified in the instruction.
- Limitations: Western-blot data are simulated densitometry values, not raw images; synergy model is HSA on viability, which is one of several possible synergy metrics. The task checks procedural correctness and data integration rather than biological discovery.
