# Review notes

- Synthetic lipidomics data generated programmatically with seed 20251211; real domain structure (1200 features, 5 replicates/condition, QC and blanks) but not real patient data.
- Shortcut probe: generator script is deleted from image after data generation; gold analysis and seed live only in tests/.
- Tool chain: Python 3.11 with numpy, pandas, scipy, statsmodels for Welch t-test and BH FDR.
- Known limitation: lipid classes are simplified; real lipidomics would require MS/MS confirmation and more complex normalization.
