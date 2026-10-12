# Review notes

- Data is synthetic but realistic, generated deterministically from seed 42. The generator is provided in environment/data/gen_data.py; the verifier regenerates the same data from a protected copy in tests/gold_gen.py.
- Shortcut probe: The agent cannot read the expected metrics. The raw data is provided, and the agent must implement the metrics correctly. The verifier recomputes the expected metrics independently.
- Tool chain: Python 3.11 with numpy, pandas, scipy. No specialized domain binaries are required; the task is a data-analysis and statistical-reasoning benchmark.
- Known limitation: The dataset is synthetic and the method names are fictional proxies for real algorithms (AlphaFold2, ESMFold, Rosetta, FoldX, etc.). The statistical structure mimics the paper's benchmark but does not use real experimental measurements.
