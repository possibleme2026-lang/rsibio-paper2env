# Review notes

- Data substitution: Synthetic library of 1,000 cyclic peptides (length 18) generated deterministically with a hidden 8-residue motif at 90% per-position conservation; mimics the MDM2-targeting library described in the paper.
- Shortcut probe: The gold motif and true rotations are not present in the environment; the input generator script is removed after build. A naive linear MSA or k-mer counting cannot recover the positional motif because the motif is mutated and circularly permuted.
- Tool chain: Docker image includes Python 3.11, numpy, scipy, pandas, biopython. No standard MCSA tool is installed; the agent must implement the circular alignment.
- Known limitations: The synthetic data is a simplified model of the paper's real peptide library; it does not model secondary structure or binding affinity. The MCSA problem is reduced to rotation-only alignment because all peptides are length 18 with no indels.
