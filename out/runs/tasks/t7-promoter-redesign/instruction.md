# Redesign a T7 expression cassette with the T7 Promoter Calculator

T7 RNA polymerase gives very high in-vitro transcription yields, but it also initiates at low-affinity
("cryptic") T7-like sites scattered through the template. Cryptic initiation consumes NTPs, lowers
full-length mRNA purity and produces immunogenic byproducts that must be removed downstream. A
sequence-to-function "T7 Promoter Calculator" has been published that predicts the log10 relative
transcription rate of any DNA sequence from its core motif (-17..+6 around the transcription start
site, TSS = +1) and its two flanking regions (-40..-18 and +7..+30).

You are given the published calculator parameters and one therapeutic T7 expression cassette. Your
job is to (i) implement the calculator, (ii) map every T7 site in the cassette on both strands, and
(iii) re-design the cassette so that cryptic initiation is strongly reduced while the encoded protein
and the intended on-target promoter are preserved.

## Inputs

All inputs are in `/app/data/`:

- `model_spec.json` - the calculator: basal offset, a position-specific weight matrix (`core_pwm`)
  over the 24 core positions -17..+6, a `dinucleotide_weights` table applied to the upstream flank
  (-40..-18) and downstream flank (+7..+30), the detection threshold (0.0), the strong-site threshold
  (2.5) and the non-maximum-suppression window (12 nt).
- `cassette.fa` - the wild-type cassette, one record `t7_cassette_wildtype`, 3000 nt, uppercase A/C/G/T.
- `cassette_annotation.json` - `{length, canonical_tss_1based, cds_start_0based, cds_end_0based}`.
  The CDS is the half-open interval [cds_start_0based, cds_end_0based) and begins with ATG, ends with a
  stop codon. `canonical_tss_1based` is the 1-based position of the +1 nucleotide of the intended
  (canonical) T7 promoter.
- `heldout_promoters.csv` - 1500 sequences of 90 nt (`seq_id,seq`). For every row the TSS is at 0-based
  index 50 of the sequence (the core therefore occupies indices 33..56).
- `benchmark_promoters.csv` - 500 sequences of 90 nt with noisy measured log10 rates
  (`seq_id,seq,measured_log10_rate`). A correct implementation of the calculator reaches R^2 ~ 0.8
  against these measurements; use it to sanity-check your scorer.
- `gen_inputs.py` - the deterministic script that produced all of the above.

## Task

1. Implement the calculator exactly as specified:
   `score(seq, tss) = basal_offset + sum_{r=-17..6} core_pwm[r][seq[tss+r]] + sum of dinucleotide weights`
   where a dinucleotide weight is added for every adjacent pair (i, i+1) wholly inside a flank window.
   The 24-nt core must fit inside the sequence, otherwise the position is not a valid TSS; flank
   positions that fall outside the sequence simply contribute nothing.
2. Predict the log10 rate of every held-out promoter.
3. Scan the wild-type cassette on both strands for every valid TSS. Keep positions with score >= 0.0 and
   apply non-maximum suppression within +/-12 nt on the same strand: a position is reported only if no
   other position on the same strand within the window has a higher score (ties are won by the smaller
   forward coordinate).
4. Re-design the cassette. You may change nucleotides anywhere except inside the canonical promoter
   window [-40, +30] around the annotated canonical TSS. Every change that falls inside the CDS must be
   synonymous. Destroy (or push below the detection threshold) as many cryptic sites as possible.
5. Report the wild-type and re-designed site inventories and design statistics.

## Constraints

1. `predictions.csv` must contain one row for every id in `heldout_promoters.csv` with a finite numeric
   `predicted_log10_rate`; RMSE against the true calculator score must be <= 0.05 log10 units.
2. `sites_wildtype.json` must list the local-maximum sites of the wild-type cassette (position tolerance
   +/-2 nt, score tolerance +/-0.15); F1 against the reference list must be >= 0.90.
3. The re-designed cassette must be exactly 3000 nt long.
4. The re-designed cassette must contain only uppercase A/C/G/T.
5. The CDS of the re-designed cassette must translate to the identical protein (same amino-acid string,
   including the terminal stop) as the wild-type CDS.
6. The 24-nt canonical core at `canonical_tss_1based` must be byte-identical to the wild type.
7. Summing the scores of all cryptic sites (0.0 <= score < 2.5) must give a total that is at least 50%
   lower than the wild-type total.
8. No new strong site (score >= 2.5) may appear anywhere except within +/-40 nt of the canonical TSS.
9. The GC content of the re-designed CDS must be within 0.02 of the wild-type CDS GC content.
10. At most 60 nucleotides may change in the whole cassette, and at most 82 inside the CDS.
11. The canonical site score of the re-designed cassette must be at least 0.97 times the wild-type
    canonical score.
12. `design_report.json` must be present and its fields must agree with the grader's recomputation
    (numeric tolerance 0.05 for scores, 0.02 for fractions, exact for counts and positions).

## Output

Write everything into `/app/output/`.

- `/app/output/predictions.csv`

  ```
  seq_id,predicted_log10_rate
  ho_0001,1.2374
  ```

- `/app/output/sites_wildtype.json`

  ```json
  {"sites": [{"position": 401, "strand": "+", "score": 4.1321, "class": "strong"},
              {"position": 712, "strand": "-", "score": 0.8410, "class": "cryptic"}]}
  ```

  `position` is the 1-based forward coordinate of the +1 nucleotide.

- `/app/output/redesigned_cassette.fa` - one record, header `>t7_cassette_redesigned`, then the sequence.

- `/app/output/design_report.json`

  ```json
  {
    "canonical_site_position": 401,
    "wildtype":  {"n_sites": 9, "n_cryptic": 8, "total_cryptic_score": 6.4210, "canonical_score": 4.1321},
    "redesigned":{"n_sites": 2, "n_cryptic": 1, "total_cryptic_score": 0.7510, "canonical_score": 4.1321},
    "nucleotide_changes": 17,
    "cds_nucleotide_changes": 9,
    "gc_content_cds_wildtype": 0.5123,
    "gc_content_cds_redesigned": 0.5088,
    "cryptic_score_reduction_fraction": 0.8830
  }
  ```

  (the numbers above are illustrative only).

## Scoring

The grader recomputes the calculator from the published parameters and re-derives the wild-type and
re-designed site inventories itself. It evaluates 13 independent checks (the 12 constraints above, with
length and alphabet counted separately) and reports `score = reward = passed / 13`; partial credit is
impossible to fake because every check is recomputed from the protected reference implementation.
