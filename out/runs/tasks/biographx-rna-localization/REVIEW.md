# Review notes

- Synthetic data generated deterministically; no real patient data.
- Shortcut probe: gold outputs are computed by the grader from the same inputs; the agent must implement the full pipeline correctly rather than read a hidden answer file.
- Tool chain: ViennaRNA RNAfold installed via apt; used as ground truth for secondary structure.
- Limitations: the task tests inference reproduction, not training; model weights are provided.
