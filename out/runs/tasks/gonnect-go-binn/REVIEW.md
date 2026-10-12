# Review notes

- TCGA RNA-seq data cannot be redistributed directly, so a small synthetic expression matrix with the same sample-by-gene schema and a simplified GO term table is shipped instead.
- The task is a faithful simplified version of GONNECT: GO terms define latent nodes and the encoder is restricted to member genes; a linear GO-constrained encoder is accepted rather than a full deep BINN.
- Shortcut probe: shipping a visible data generator could expose latent activities; instead the observable expression matrix is precomputed and the hidden gold activations live only in tests/gold.csv during verifier phase.
- Known limitations: GO terms are non-overlapping, reconstruction error and published BINN baselines are not scored, and the hidden gold represents true latent pathway activities rather than a GSEA-derived reference computed from external gene sets.
