# Review notes

- Data are synthetic, generated deterministically by environment/data/gen_inputs.py with seed 20260729. They emulate the paper's design (10 subjects, 100 trials per task per subject, 16-channel EEG at 128 timepoints, 101-point mouse trajectories) but are not the original data.
- Shortcut probe: the generator is visible to the agent, but the requested outputs are sample statistics that require running the full analysis pipeline; knowing the generation parameters does not directly provide the grader's reference values.
- Tool chain: Python 3.11 with numpy, pandas, scipy, scikit-learn, mne installed. The reference implementation uses scikit-learn logistic regression and scipy t-tests.
- Known limitations: the EEG signal is a simplified evoked pattern; real EEG preprocessing (filtering, artifact rejection, ICA) is not modeled. The fusion analysis uses a single timepoint and Pearson correlation, whereas the paper used more complex representational dynamics.
