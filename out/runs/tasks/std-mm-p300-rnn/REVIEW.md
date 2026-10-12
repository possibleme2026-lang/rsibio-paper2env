# Review notes

- Data is synthetic, generated deterministically by environment/data/gen_inputs.py with fixed seeds. The model equations are specified in the instruction; the agent must implement them exactly.
- Shortcut probe: the agent cannot read the gold metrics; they are recomputed by tests/gold_gen.py from the same inputs. Hardcoding paper values is not possible because the paper does not provide these exact metrics for this synthetic configuration.
- Tool chain: Python 3.11 with numpy and scipy. No external domain tools are required; the ground truth is the specified simulation.
- Known limitations: the model is a simplified rate-based RNN, not a spiking network. The metrics are computed on synthetic sequences with 100 trials per condition to keep runtime tractable. The grader uses tolerances to allow small numerical differences.
