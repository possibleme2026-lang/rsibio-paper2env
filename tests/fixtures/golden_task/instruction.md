# ChIP-seq peak-calling benchmark

You are given a synthetic ChIP-seq experiment: an immunoprecipitation (IP)
coverage track and a matching input/control track, both as bedGraph files
over a 1 Mb region of chr1. Twelve enrichment peaks are embedded in the IP
track; the control track carries only background.

Write a peak caller that reads /app/data/signal.bedgraph and
/app/data/control.bedgraph and outputs predicted peak intervals to
/app/predictions.bed (BED6 format, one peak per line).

## Inputs
- /app/data/signal.bedgraph: per-10bp-bin coverage of the IP track
  (1.6 MB, ~79k lines; synthetic and deterministic, numpy seed 42).
- /app/data/control.bedgraph: matching input/control coverage track
  (same background law, no embedded peaks; seed 43).
- /app/data/gen_inputs.py: the deterministic generator that re-creates
  both tracks (for inspection and provenance only).

## Task
1. Load both bedGraph tracks and build per-bin coverage arrays.
2. Estimate the background coverage rate from the control track
   (a robust statistic such as the median is preferred over the mean).
3. Score each candidate interval in the IP track against the background
   using a Poisson-style significance statistic.
4. Merge adjacent significant bins (and any peaks closer than 50 bp)
   into peak intervals.
5. Discard intervals whose mean coverage is below 2x the background rate.
6. Write the final peak list to /app/predictions.bed in BED6 format
   (chrom, start, end, name, score, strand), sorted by start position.
7. Then run the environment's evaluation entry and record your result in
   /logs/verifier/reward.json as: {"score": <float>, "n_peaks": <int>}

## Constraints
- Do not read any gold-standard peak list; none is shipped in the agent
  image and the grader checks for access attempts.
- Peaks closer than 50 bp must be merged into one interval.
- Only intervals with at least 2x background coverage may be reported.
- The output BED must be sorted by start position (ascending) within
  each chromosome.
- Every output line must be valid BED6 with a numeric score column.
- Runtime budget: 10 minutes, 2 CPUs, 4 GB RAM. No network access.

## Output
/app/predictions.bed (BED6: chrom, start, end, name, score, strand) and
/logs/verifier/reward.json with the fields above.

## Scoring
The grader re-derives the hidden gold peaks from an independent oracle
seed at verify time (the agent container never sees them), then scores
predicted-vs-gold overlap with interval F1 (reciprocal overlap >= 0.5,
ENCODE-style) and applies per-constraint checks (merge distance, minimum
fold enrichment, coordinate sorting, gold-file access probe) as partial
credit. Total score = 0.7 * F1 + 0.3 * constraint pass rate. A shortcut
that copies any shipped input file into predictions.bed scores ~0.
