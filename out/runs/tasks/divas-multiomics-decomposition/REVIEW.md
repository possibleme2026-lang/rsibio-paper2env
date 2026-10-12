# Review notes

- Data is synthetic multiomics generated with a fixed seed; true latent components are hidden in tests/gold_gen.py and not readable by the agent.
- Shortcut probe: the generator script is deleted from the final image after data generation; the agent only sees the CSV matrices.
- Tool chain: container installs R and the DIVAS package from GitHub; the agent may use R or implement the method in Python.
- Known limitation: The DIVAS package API may evolve; the grader matches components via score correlation and exact modality sets, which is robust to sign flips and orthogonal rotations.
