# Review notes

- Synthetic data generated programmatically with a fixed seed; realistic PDB backbone geometry and amino acid potentials.
- Shortcut probe: the gold sequence is not present in the environment; the grader recomputes it from a protected seed using the exact algorithm.
- Tool chain: numpy and biopython installed for PDB parsing and numerical computation.
- Limitations: the actual TriFlow model is not available; this task implements a deterministic mean-field surrogate inspired by the paper's discrete flow-matching idea.
