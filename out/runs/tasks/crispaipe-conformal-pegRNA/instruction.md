# Prime editing outcome prediction with conformal simplex regions

You are implementing the core of crispAIPE: given a pegRNA design, predict the three competing prime-editing outcome fractions (edited, unedited, indel) and report a calibrated 90% highest-density region on the 2-simplex. Prime-editing outcomes are compositional: the three fractions sum to one, and the region must quantify uncertainty rather than giving only a point estimate.

## Inputs

- `/app/data/train_library1.csv`: 1,500 pegRNAs from PRIDICT-style Library-1 (HEK293T). Columns: `peg_id`, `cell_type`, `spacer_seq` (20 nt), `pbs_seq` (12 nt), `rtt_seq` (10-24 nt), `target_seq` (30 nt), `edit_type` (`sub`/`ins`/`del`), `edit_size`, `pam_type` (`NGG`/`NAG`/`NGA`), `nick_offset`, `rtt_len`, `total_reads`, `n_edited`, `n_unedited`, `n_indel`. GC fractions are not given; compute them from the sequences. The count columns are noisy sequencing observations of the underlying outcome composition.
- `/app/data/eval_library2_features.csv`: 500 feature-only pegRNAs from Library-2 (K562, cross-cell-type). Same feature columns but no count columns. No labels for Library-2 are provided.

## Task

1. Engineer features from the raw sequences and design annotations (at minimum: GC fraction of RTT, PBS, and spacer; RTT length; edit type and size; PAM type; nick offset; edit-size-by-deletion interaction).
2. Fit a probabilistic three-class model over the 2-simplex. Use a Dirichlet-style likelihood (e.g. Dirichlet regression or Dirichlet-multinomial regression) so each pegRNA gets a full predictive distribution, not just independent point estimates.
3. Perform split-conformal calibration on a held-out fold of Library-1 only. Use a nonconformity score based on the Dirichlet predictive density of the observed outcome fractions and compute the finite-sample quantile `q_hat` for nominal 90% coverage. Do not use Library-2 labels.
4. Apply the same `q_hat` unchanged to every Library-2 pegRNA to construct a 90% highest-density region. Because the outcome lives on the 2-simplex, represent each region as a convex polygon: a list of vertices in the same `(edited, unedited, indel)` coordinate system.
5. Write predictions, regions, and calibration metadata under `/app/output/`.

## Constraints

1. `/app/output/predictions.csv` exists with header `peg_id,p_edited,p_unedited,p_indel` and contains exactly the 500 Library-2 peg_ids, one row each, no duplicates.
2. Every prediction row has nonnegative probabilities that sum to 1.0 within 1e-3.
3. Spearman rho between predicted `p_edited` and the hidden Library-2 edited fraction is at least 0.50.
4. Spearman rho is at least 0.40 for the unedited fraction and at least 0.15 for the indel fraction.
5. Mean absolute error is at most 0.15 for the edited fraction and at most 0.12 for the indel fraction.
6. `/app/output/regions.jsonl` contains exactly one JSON object per Library-2 peg_id. Each `region` is a convex polygon on the 2-simplex with at least 3 vertices; every vertex is nonnegative and sums to 1.0 within 1e-6.
7. The nominal-90% conformal regions achieve empirical coverage between 0.80 and 0.98 on the hidden Library-2 outcomes.
8. Regions are informative: median simplex area across the 500 regions is at most 0.25, and at least 80% of regions have area at most 0.40 (the full simplex has area 0.5).
9. `/app/output/calibration.json` records `nominal_coverage` equal to 0.90 within 0.01, `calib_size` at least 250, and a finite conformal quantile `q_hat`.

## Output

Write all files under `/app/output/`:

- `predictions.csv` with header `peg_id,p_edited,p_unedited,p_indel` and 500 rows, e.g. `L2_00000,0.42,0.50,0.08`.
- `regions.jsonl`, one JSON object per line: `{"peg_id": "L2_00000", "region": [[0.30,0.60,0.10],[0.45,0.45,0.10],[0.40,0.52,0.08]]}`.
- `calibration.json`, e.g. `{"nominal_coverage": 0.90, "calib_size": 300, "q_hat": 2.17}`.

## Scoring

The grader independently checks each of the 9 numbered constraints (format, probability validity, three accuracy checks, region geometry, coverage, region efficiency, calibration metadata). The score is the fraction of constraints passed, giving partial credit across the pipeline.
