# Review notes

- Data substitution: the original metaproteomics dataset is not fully public; we generate a synthetic-but-realistic DDA dataset with 400 proteins, ~2200 unique peptides, 100 shared peptides, 6 runs, and known spike-in fold changes. The generator runs at Docker build time and is removed from the final image.
- Shortcut probe: we considered the shortcut of reading the generator or hidden truth. The generator is removed from /app/data after build; hidden truth is only in tests/. The grader regenerates expected outputs from a protected seed.
- Tool chain: Python 3.11 with numpy, pandas, scipy. The task is a data-analysis pipeline that abstracts raw spectra into feature tables; no external mass-spectrometry binary is required.
- Known limitations: synthetic data cannot capture all real MS1 feature complexity; AUC is simulated as a noisy power of true abundance. The MBR false-positive definition is run-specific and based on absence of non-MBR evidence.
