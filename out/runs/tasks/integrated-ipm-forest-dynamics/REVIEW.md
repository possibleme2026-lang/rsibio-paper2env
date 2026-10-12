# Review notes

- Synthetic data generated deterministically with seed 42 using a size-structured IPM. Input files are realistic: 400 population count rows, 2000 growth records, 21 weather rows.
- Shortcut probe: the data-generating script is removed from the container after build; gold parameters and 2020 distribution are only in tests/gold_gen.py.
- Tool chain: standard scientific Python (numpy, pandas, scipy, scikit-learn, statsmodels) installed. No domain-specific binary needed; the IPM is the ground truth.
- Known limitations: the true model is a simplified linear IPM; real forests have more complex growth and survival functions. The tolerance for growth_vpd (0.08) is calibrated to the simulation noise.
