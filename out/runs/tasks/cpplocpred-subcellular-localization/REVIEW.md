# Review notes

- Data substitution: The original CPP sequences and labels are not publicly available in the paper; we generated a synthetic dataset of 2021 peptides (1216 train, 305 val, 500 test) using class-specific amino acid motifs and background composition, ensuring realistic lengths (5-30 aa) and class-specific patterns. The generator is deterministic (seed 20260728) and lives in tests/gold_gen.py; the environment generator is deleted after build.
- Shortcut probe: The hidden test labels are never written to the agent-readable filesystem. The build-time generator is removed from the final image. The grader regenerates the dataset from a protected seed in tests/gold_gen.py, so the agent cannot read or infer the gold labels.
- Tool chain: CatBoost 1.2.1 is installed via pip. The grader uses scikit-learn's roc_auc_score. DDR features are computed from first principles as specified in the instruction.
- Known limitations: The synthetic data is designed to make DDR features highly discriminative; real CPP data may be noisier. The mitochondria test set has only 19 sequences, so AUC for that class may have higher variance.
