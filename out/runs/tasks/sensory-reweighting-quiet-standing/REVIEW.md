# Review notes

- Synthetic data: no public perturbation-response dataset is redistributed; environment/data/conditions.csv contains generated sensory precision and bias inputs matching the simplified model schema.
- Shortcut probe: considered tampering with environment input to change the grader; the verifier uses a hidden canonical copy of the input rows and the environment directory is made read-only.
- Known limitations: the blueprint implements a fixed-point precision-weighting simplification of the paper's variational free-energy model, not a full temporal inverted-pendulum simulation; scoring uses RMSE with a fixed tolerance.
