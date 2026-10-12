# Review notes

- Data is synthetic and generated programmatically with a fixed seed for reproducibility. The task down-samples the paper's 452 patients to 200 samples and 50 SNPs to keep the blueprint compact while preserving realistic structure.
- Shortcut probe: The generator script is placed in the environment, but it only produces raw inputs (genotypes, mapping, network, severity). The propagated TF activities and correlations are not stored anywhere in the environment; the agent must implement the propagation algorithm. The grader recomputes the expected outputs from the input files.
- Tool chain: numpy, pandas, scipy. The network propagation uses a closed-form solution with row-normalized network and alpha=0.7, analogous to the iSNP approach in the paper.
- Known limitations: This is a simplified version of the full iSNP platform. It does not include the full epithelial expression filtering or clustering steps, focusing instead on the core SNP-to-TF propagation and correlation with clinical severity.
