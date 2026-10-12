# Neutron vs Photon Transcriptional Discrimination

The agent must train a binary classifier on synthetic log2 gene-expression profiles from irradiated human blood samples and predict neutron versus photon exposure for an unlabeled test set.

Inputs:
- /app/data/train.csv: labeled samples. label=0 means photon, label=1 means neutron. Feature columns are sample_id and eight gene-expression columns.
- /app/data/test.csv: unlabeled samples with the same feature columns.

Output:
- /app/predictions.csv with columns sample_id and prob_neutron. prob_neutron must be finite numeric values.

Scoring:
- The verifier compares prob_neutron to hidden gold labels for the test samples and computes AUROC. A missing, malformed, or incomplete prediction file scores 0.

Constraints:
- Use only the provided data under /app/data. No network access is available. Do not attempt to read or modify /tests; gold labels are hidden.
