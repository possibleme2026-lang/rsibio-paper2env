# DBD-class annotation of candidate regulatory sequences

## Context

Transcription factors (TFs) are grouped into families by their DNA-binding domain (DBD), and members of the same family share highly similar binding preferences. Assigning a candidate regulatory region to a **DBD class** instead of to an individual TF is therefore the statistically well-posed version of the TFBS-assignment problem, and it is what makes genome-wide regulatory annotation possible at all. You are given DBD-class position probability matrices (PPMs), a genomic background model, and a set of candidate 120 bp regulatory sequences. Your job is to implement a complete, statistically calibrated, competition-aware DBD-class annotation pipeline and to run it over the whole candidate set.

## Inputs (all under `/app/data`)

| file | content |
|---|---|
| `dbd_classes.txt` | the 23 canonical DBD-class names, one per line |
| `dbd_class_models.tsv` | header `class_name<TAB>position<TAB>A<TAB>C<TAB>G<TAB>T`; one row per motif position; each row sums to 1 (6 decimals) |
| `background.tsv` | header `base<TAB>frequency`; i.i.d. genomic background over A, C, G, T |
| `test_sequences.fa` | 520 records, ids `seq_000001` ... `seq_000520`, each exactly 120 bp, 60 bp per line |
| `calibration_sequences.fa` | 200 further sequences drawn from the same simulation process |
| `calibration_labels.tsv` | `seq_id<TAB>dbd_class` — true DBD class of each calibration sequence (`None` = background only) |

The test sequences are simulated: most of them carry a single functional site of one DBD class (sampled from that class's PPM and possibly mutated), some carry additional weaker competing sites, and the rest are background only. No annotation of class, position or strength exists for `test_sequences.fa` anywhere you can read; the calibration set is the only labelled data.

## Task

Implement the following pipeline exactly and run it over all 520 test sequences (do not use the calibration labels as a shortcut — they only cover a different set of sequences).

1. Parse `dbd_class_models.tsv` and `background.tsv`.
2. Convert every class PPM into log-odds: `L[c][j][b] = log2(P[c][j][b] / bg[b])`.
3. Discretise in units of 0.01 bit: `k[c][j][b] = floor(100 * L[c][j][b] + 0.5)` (`floor` on the offset value, so negative values are also rounded half-up). The integer score of a word `w` of length `W_c` is `sum_j k[c][j][w_j]`; the **reported score is that integer divided by 100**, written with 2 decimals.
4. Compute, for each class, the **exact** null distribution of that integer score when the word is drawn i.i.d. from `bg` — use dynamic programming over the integer score range (equivalent to enumerating all `4^W_c` words; Monte-Carlo sampling is not acceptable). The p-value of an observed word with integer score `s` is `#{words with integer score >= s} / 4^W_c`.
5. Scan every test sequence on both strands and every window start; for each class keep its single best window (largest integer score). `start`/`end` are 0-based, half-open, forward-strand coordinates; for a `-` match the reverse complement of `seq[start:end]` is the matching word.
6. Decide. With `N_tests = 2 * sum_c (120 - W_c + 1)` and `p_cut = 0.05 / N_tests`, class `c` is *detected* in a sequence iff its best p-value is `<= p_cut`. The reported class is the detected class with the smallest p-value (ties: larger score, then alphabetically first class name). If no class is detected, report `None` — but still report the globally best window (lowest p-value over all classes, all windows) in `score`, `p_value`, `strand`, `start`, `end`. Set `ambiguous = 1` iff at least two classes are detected and the second smallest detected p-value is `<= 5 *` the smallest detected p-value; otherwise `0`.
7. Aggregate the per-class statistics of your own calls into a genome-wide annotation table.

## Constraints

1. `/app/output/predictions.tsv` must be tab separated with the header exactly `seq_id`, `predicted_class`, `score`, `p_value`, `strand`, `start`, `end`, `ambiguous` and exactly 520 data rows, one per test sequence, in the same order as `test_sequences.fa`.
2. `predicted_class` must be one of the 23 names in `dbd_classes.txt` or `None`; `strand` is `+` or `-`; `0 <= start < end <= 120`; for non-`None` rows `end - start` equals the motif width of the predicted class; `ambiguous` is `0` or `1`.
3. `score` must be the discretised log-odds score (2 decimals) of the window actually pointed at by `strand`/`start`/`end`, under the model of the reported class (or, for `None` rows, of the globally best window) — recomputed independently, it must agree within 0.01 bit.
4. `p_value` must be the exact dynamic-programming null tail probability of that same word, in scientific notation, accurate to within 0.05 log10 units.
5. The reported class must follow the detection rule of step 6 exactly, and macro-F1 over the 24 labels (23 DBD classes + `None`) computed against the hidden truth must be as high as possible: `>= 0.55` for full credit.
6. Specificity on the negative control: at least 70% of the background-only test sequences must be labelled `None`.
7. Consistency across classes: at least 20 of the 23 DBD classes must reach per-class recall `>= 0.5`.
8. `/app/output/annotation.tsv` must be consistent with your own `predictions.tsv`: header `class_name`, `n_sites`, `total_bp`, `bp_fraction`, `mean_score`; exactly 23 rows, one per DBD class in the order of `dbd_classes.txt`; `n_sites` and `total_bp` (sum of `end-start` of that class's winning calls) must match your predictions exactly, `bp_fraction = total_bp / (520 * 120)` and `mean_score` the mean of that class's reported scores (0.0 when `n_sites = 0`).
9. The `ambiguous` column must agree with the rule of step 6 (the grader recomputes it) on at least 85% of the sequences.

## Output

- `/app/output/predictions.tsv` (520 rows + header), tab separated, rows in FASTA order.
- `/app/output/annotation.tsv` (23 rows + header), tab separated, classes in the order of `dbd_classes.txt`.

Worked example rows for `predictions.tsv` (values illustrative):

```
seq_id	predicted_class	score	p_value	strand	start	end	ambiguous
seq_000001	ETS	19.42	3.71e-08	+	42	57	0
seq_000002	None	11.07	2.44e-04	-	83	98	0
seq_000003	Homeodomain	21.85	7.90e-09	+	5	21	1
```

## Scoring

The grader (`/tests/grade.py`) recomputes everything from a protected generator seed and scores nine independent checks with partial credit: output format/integrity (0.10), field and coordinate validity (0.05), score accuracy on a sampled set of rows (0.10), p-value accuracy on the same rows (0.10), macro-F1 against the hidden truth (0.30), specificity on background-only sequences (0.07), per-class recall coverage (0.08), annotation-table consistency (0.12) and ambiguity-flag agreement (0.08). `score = reward =` weighted sum, written as numbers to `/logs/verifier/reward.json`.
