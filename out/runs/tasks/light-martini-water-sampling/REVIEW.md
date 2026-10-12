# Review notes

- Data substitution: synthetic trajectories generated deterministically from seeds; format mimics GROMACS .gro/.xtc.
- Shortcut probe: generator script removed after build; gold metrics recomputed by grader from trajectories.
- Tool chain: MDAnalysis for trajectory parsing, numpy/scipy for analysis.
- Known limitations: Simplified chain and lipid models; metrics are illustrative of the paper's method.
