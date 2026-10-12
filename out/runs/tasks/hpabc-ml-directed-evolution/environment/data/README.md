# HpaBC alpha-subunit engineering dataset

All numbers were produced by `gen_inputs.py` (public seed 20260731). Re-run
`python gen_inputs.py --out .` to reproduce them exactly.

* `hpabc_alpha.fasta` - wild-type alpha subunit, 606 aa.
* `single_mutants.csv` - 220 single mutants from the directed-evolution library.
  `relative_activity` is the L-Dopa titre relative to wild type (WT = 1.0), measured in four
  assay batches (B1-B4) with about 10% log-scale assay noise.
* `de_variants.csv` - 56 multi-mutant variants (2-4 substitutions) isolated during the
  directed-evolution campaign.
* `zeroshot_scores.csv` - EVOLVEpro-style zero-shot prediction of the log-activity change for
  every possible single substitution of the alpha subunit. These predictions are noisy
  (correlation with measured effects is moderate) and are strongly negative at the essential
  positions 101, 104, 106, 109, 205 and 397.
* `test_variants.csv` - 64 combinatorial designs whose activities are withheld by the verifier.

The essential positions (101, 104, 106, 109, 205, 397) form the FAD-binding and catalytic
motifs; substitutions there abolish activity.
