# Review notes

- Real CA3-CA1 optogenetic pattern trains and electrophysiology cannot be redistributed; environment contains a synthetic unlabeled dataset with the same schema (binary E/I bouton activations, trial-grouped events).
- Shortcut probe: gold labels are not present in environment; verifier regenerates hidden labels from the specified STP/EI model using a private copy of the input hex string. Raw E/I-count novelty was considered and is not sufficient for faithful scoring.
- Known limitations: simplified single-neuron STP/EI model, fixed parameters, deterministic synthetic data, and event-level AUROC against model-generated labels rather than experimental recordings.
