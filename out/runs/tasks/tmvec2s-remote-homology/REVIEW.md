# Review notes

- Data substitution: synthetic protein domains with generated 3D structures and sequences; realistic sizes (200 domains, 128 aa, 2000 training pairs).
- Shortcut probe: agent cannot access query structures; the generator script is deleted after Docker build, and hidden query structures are only in the grader.
- Tool chain: numpy, scipy, scikit-learn, biopython. No domain-specific binaries needed because ground truth is synthetic.
- Known limitations: synthetic fold families may be easier than real CATH/SCOPe; thresholds calibrated for this synthetic benchmark.
