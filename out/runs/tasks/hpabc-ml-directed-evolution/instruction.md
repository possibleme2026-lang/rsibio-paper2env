# HpaBC monooxygenase design: fuse zero-shot ML with directed-evolution data

HpaBC is a flavin-dependent monooxygenase that converts L-tyrosine to L-Dopa, the rate-limiting
step of microbial L-Dopa production. The alpha subunit (606 aa) has weak native activity on
L-tyrosine, so your job is to rank combinatorial variants that improve catalytic activity
relative to wild type (WT = 1.0).

You have two information sources, both shipped in `/app/data`:

* a directed-evolution (DE) campaign: 220 assayed single mutants and 56 assayed multi-mutant
  variants, each with a measured `relative_activity` (WT = 1.0, typical assay noise ~10%);
* an EVOLVEpro-style **zero-shot** model that scores every possible single substitution of the
  alpha subunit (606 positions x 19 substitutions).

A partner laboratory has built 64 combinatorial designs (`test_variants.csv`) and will run the
activity assay on them. Those assay values are held by the verifier; you must predict them.

## Inputs (all under /app/data)

* `hpabc_alpha.fasta` - wild-type alpha subunit, 606 aa, single record.
* `single_mutants.csv` - 220 rows: `variant_id,mutation,position,wt_aa,mut_aa,relative_activity,assay_batch`.
  `mutation` is written `<wt><position><mut>`, e.g. `A45T`.
* `de_variants.csv` - 56 rows: `variant_id,mutations,n_mutations,relative_activity`.
  `mutations` is a semicolon-separated list of single-mutation tokens (2-4 mutations each).
* `zeroshot_scores.csv` - 11514 rows: `position,wt_aa,mut_aa,zeroshot_score`. The score is the
  model's predicted change in log activity for that substitution; scores at positions
  101, 104, 106, 109, 205 and 397 are strongly negative.
* `test_variants.csv` - 64 rows: `variant_id,mutations,n_mutations` (2-5 mutations each, no labels).
  The order of these rows defines the required order of your prediction file.
* `gen_inputs.py` - the deterministic generator (public seed) that produced every file above.
  Re-running it reproduces the shipped measurements exactly. It does **not** reproduce the
  held-out assay values, which come from a separate protected generator.

## Task

1. Parse the FASTA, the DE measurements and the zero-shot table; check that every mutation token
   in the DE and test files is consistent with the wild-type sequence.
2. Quantify how informative the zero-shot scores are: fit the relationship between zero-shot
   score and measured log activity on the experimental data, so that substitutions with **no**
   experimental measurement can still be handled.
3. Build a quantitative activity model that (a) uses the measured single and multi-mutant data,
   (b) uses the zero-shot scores for substitutions never measured, and (c) generalises to
   3-5 mutation designs. Choose model complexity with 5-fold cross-validation and report the
   configurations you compared.
4. Apply the model to the 64 held-out designs, rank them, and select a 5-design panel for
   experimental testing under the design rules below.

## Constraints

1. `/app/output/predictions.csv` must have the header `variant_id,predicted_activity` and exactly
   64 data rows, with `variant_id` values **in the same order** as `data/test_variants.csv`.
2. Every predicted activity must be a finite float in `[0.005, 10.0]` (relative to WT = 1.0).
3. Positions 101, 104, 106, 109, 205 and 397 are essential (FAD binding / catalysis): every
   design that carries a mutation at one of these positions must be predicted at `<= 0.4`.
4. `/app/output/top10.csv` (header `rank,variant_id,predicted_activity`) must list exactly the ten
   designs with the highest predicted activity, ranks 1-10 in descending order of your prediction.
5. `/app/output/panel.csv` (header `variant_id,mutations,predicted_activity,rationale`) must contain
   exactly the five best **eligible** designs, where eligible means `n_mutations <= 4` **and** no
   mutation at an essential position. `mutations` must reproduce the token list from
   `test_variants.csv`, `predicted_activity` must equal the value in `predictions.csv`, and all five
   panel members must have predicted activity `>= 1.2`.
6. `/app/output/cv_results.json` must benchmark at least three model configurations. Each entry in
   `configs` has `name`, `folds` (exactly five Spearman values from a 5-fold cross-validation) and
   `mean` (the arithmetic mean of `folds`). The file also carries `best_config` and
   `best_cv_spearman` naming the best configuration.
7. Accuracy: the Spearman rank correlation between your predictions and the held-out assay values
   must exceed the zero-shot-only baseline (a model that uses only `zeroshot_scores.csv` with its
   scale fitted on the DE data) by at least 0.06.
8. Log-scale accuracy: the RMSE between your predicted activities and the held-out assay values,
   computed on natural-log activity, must be at most 0.85 times the baseline RMSE.
9. Design recovery: at least four of your top-10 designs must be among the twenty most active
   designs in the held-out assay.

## Output

All files go to `/app/output/`.

`predictions.csv` (row order = `test_variants.csv` order, one row per design):

```
variant_id,predicted_activity
TV001,2.4180
TV002,0.0460
```

`panel.csv`:

```
variant_id,mutations,predicted_activity,rationale
TV007,A45T;L112M;E208K;S334A,3.1204,stacked four beneficial measured single mutants
```

`top10.csv` starts `rank,variant_id,predicted_activity` and lists 10 ranked designs.

`cv_results.json` looks like:

```json
{"method": "ridge on mutation indicators plus zero-shot fallback",
 "configs": [{"name": "single_plus_de_ridge", "folds": [0.71, 0.64, 0.69, 0.58, 0.66], "mean": 0.656}],
 "best_config": "single_plus_de_ridge", "best_cv_spearman": 0.656}
```

## Scoring

The verifier recomputes the held-out assay values with a protected generator, recomputes the
zero-shot-only baseline from the shipped data, and scores nine constraint groups independently:
prediction file format (0.08), prediction domain (0.05), top-10 ranking consistency (0.08),
panel correctness (0.17), cross-validation report (0.10), Spearman improvement over the baseline
(0.22), log-RMSE ratio versus the baseline (0.12), top-10 recovery (0.10), and essential-position
decoy handling (0.08). Accuracy groups use graded partial credit for near-threshold performance;
the final reward is the weighted mean of the nine group scores.
