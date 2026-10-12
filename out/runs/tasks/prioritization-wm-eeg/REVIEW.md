# Review notes

- Data are synthetic but structured to mimic the paper's Experiment 2 EEG effects (15 Hz vs 20 Hz, phase reset, burst rate equality, topographic reorganization).
- Shortcut probe: the generator script is deleted from the image after data generation, so the agent cannot read the true parameters. The grader recomputes gold from the raw data.
- Tool chain: MNE for Morlet wavelets, SciPy for filtering and Hilbert transform.
- Known limitations: the synthetic data simplify the original task; no behavioral data are included.
