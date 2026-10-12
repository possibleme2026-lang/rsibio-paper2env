# Review notes

- Data substitution: synthetic-but-realistic scRNA-seq and bulk RNA-seq are generated deterministically by environment/data/gen_inputs.py with seed 42. Hallmark gene sets are curated subsets of real MSigDB Hallmark pathways. The generator is visible to the agent, but the grader runs the agent's pipeline on a held-out dataset with a secret seed to prevent hardcoding.
- Shortcut probe: The agent could read gen_inputs.py and hardcode the provided data's cluster IDs or signature. This fails because the held-out dataset uses a different seed and shuffled cluster labels; the pipeline must generalize.
- Tool chain: gseapy (pinned 1.1.3) for GSEA, scipy/statsmodels for differential expression, pandas/numpy for data handling, scanpy installed for optional single-cell tooling.
- Known limitations: Synthetic data may not capture full biological complexity; GSEA results depend on gseapy version (pinned). The task uses Welch t-tests rather than bulk RNA-seq-specific tools like DESeq2, as specified in the instruction.
