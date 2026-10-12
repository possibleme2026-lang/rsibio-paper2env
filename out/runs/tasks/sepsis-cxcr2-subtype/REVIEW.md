# Review notes

- Data substitution: synthetic-but-realistic bulk RNA-seq counts (200 genes x 36 mice per tissue) generated deterministically from a fixed seed. The generator is removed from the image after build so the agent cannot inspect the latent subtype assignments.
- Shortcut probe: the grader lives in tests/ and recomputes the reference z-score classification, subtype summaries, and treatment effects from the input files; the agent cannot read the reference or the true latent labels during solve time.
- Tool chain: numpy and pandas are sufficient for the specified CPM/z-score workflow; no external domain binary is required because the method is a custom signature-scoring procedure.
- Known limitations: the synthetic expression matrix simplifies the true immune cell composition and uses anonymized signature gene IDs; the task focuses on the computational workflow rather than exact biological discovery.
