# Review notes

- Synthetic SARS-CoV-2-like reference (29,903 bp) and 8 longitudinal samples generated deterministically from seed 20260506; no real patient data.
- Generator script is deleted from the image after build so the agent cannot read the simulation code; the grader recomputes the gold standard from raw inputs.
- Shortcut probe: no answer file exists in the agent-readable filesystem; the gold is derived by parsing VCF, coverage, GFF, and reference files during verification.
- Tool chain: bcftools/samtools/tabix installed for inspection/normalization; Python with Biopython/pandas/numpy for translation and tabulation.
- Known limitations: synthetic SNV-only dataset; all genes are + strand; small 8-sample cohort; no indels or structural variants.
