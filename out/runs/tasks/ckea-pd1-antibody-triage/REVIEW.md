# Review notes

- Data substitution: synthetic CA-only MD trajectories and antibody sequences based on nivolumab/pembrolizumab are generated programmatically. The generator script is deleted after build so the agent cannot inspect exact coordinate generation parameters.
- Shortcut probe: The agent might try to read the generator script to infer contacts without parsing trajectories; the script is removed from the final image, forcing actual MDAnalysis parsing.
- Tool chain: conda installs MDAnalysis, Biopython, abnumber, ANARCI; the grader uses these same tools to recompute gold.
- Known limitations: CA-only coarse-grained model; synthetic motifs; the binding score is a simplified surrogate for MD free-energy estimates.
