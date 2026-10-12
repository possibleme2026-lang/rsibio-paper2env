# Review notes

- Synthetic data generated programmatically: 8 antibody-antigen complexes (heavy chain 120 res, light 100 res, antigen 80 res) and 200 loop fragments (6-12 residues). No hidden gold answer; the task is a constraint-satisfaction/optimization problem where any loop satisfying the physical and docking constraints is acceptable.
- Shortcut probe: agent could read gen_inputs.py, but it only generates the masked framework and fragment library, not the missing loop. The loop must be built de novo or by fragment grafting with closure optimization.
- Tool chain: numpy, scipy, biopython installed for PDB parsing and numerical optimization. No external domain binaries required.
- Known limitations: synthetic coordinates are generated with a simple random-walk CA trace and fixed atom offsets, so they are not crystallographic. The grader checks loop geometry and docking contacts, not overall fold accuracy.
