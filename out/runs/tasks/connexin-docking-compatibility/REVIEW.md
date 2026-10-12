# Review notes

- Data substitution: real FETCH assay data are not redistributable; synthetic heterotypic pair scores and labels were generated with the same schema (pair_id, fetch_score, label) and include noise to reflect imperfect motif-class prediction.
- Shortcut probe: direct lookup of pair IDs in training is prevented by disjoint test pairs; compatibility-group rules would miss hidden high-score negatives and low-score positives, so threshold learning is required.
- Known limitations: simplified binary classification on synthetic scores, not full 21-isoform/210-pair experimental atlas; threshold selection may be sensitive to training noise.
