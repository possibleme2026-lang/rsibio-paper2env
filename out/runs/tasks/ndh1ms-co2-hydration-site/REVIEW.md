# Review notes

- Synthetic PDB data generated deterministically with seed 20260803; realistic sizes (thousands of atoms) but simplified sidechain representation for random residues.
- Shortcut probe: the generator script is deleted after building the image; the agent cannot read the exact planted features without running geometric analysis on the PDBs.
- Tool chain: Python 3.11 with Biopython, NumPy, SciPy available; grader recomputes all ground-truth values from the PDB files using the same algorithms.
- Known limitations: the synthetic structure is not a physically valid protein fold, but it provides a realistic computational challenge for distance-based structural analysis.
