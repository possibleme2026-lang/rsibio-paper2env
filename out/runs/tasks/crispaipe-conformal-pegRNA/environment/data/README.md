# Inputs for crispAIPE-style prime-editing task

At image build time the following files are generated into /app/data/ from a
deterministic synthetic model of PRIDICT-style prime-editing libraries:

- `train_library1.csv`: 1,500 pegRNAs with sequences, annotations, and noisy
  three-outcome sequencing counts.
- `eval_library2_features.csv`: 500 feature-only pegRNAs from a shifted
  cell-type context.

The raw sequence columns are present so that GC fractions must be computed by
the agent. The hidden Library-2 outcome fractions are never placed in the image.
