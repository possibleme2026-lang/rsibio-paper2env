# Flex Probe Assignment and UMI Counting

You are implementing a critical module of the STAR Suite pipeline for 10x Flex data. This module processes sequencing reads from a multiplexed Flex experiment: it corrects cell barcodes, assigns each read to a probe by exact Hamming distance (allowing up to 2 mismatches and considering both read orientations), and counts unique UMIs per probe per sample.

## Inputs

- `/app/data/probes.tsv`: 600 probes (200 per sample). Columns: `probe_id`, `sample_id`, `probe_seq` (40 bp).
- `/app/data/cell_barcodes.txt`: 2000 valid 16 bp cell barcodes, one per line.
- `/app/data/reads.fastq`: 30,000 reads, each 100 bp. Read structure (1-indexed): positions 1-16 = cell barcode, 17-28 = UMI (12 bp), 29-68 = probe sequence (40 bp), 69-100 = transcript sequence (32 bp). Quality scores are all 'I'.
- `/app/data/gen_inputs.py`: deterministic generator (seed 42) used to create the input files. You may read it to understand the data, but it does not produce the expected output.

## Task

Write a program that processes the reads and produces two output files:

1. `/app/output/probe_counts.tsv`: a TSV with columns `probe_id`, `sample_id`, `umi_count`. Include all 600 probes crossed with all 3 samples (1800 rows). Sort by `probe_id` then `sample_id`. `umi_count` is the number of unique (`corrected_cell_barcode`, `umi`) pairs assigned to that probe and sample.
2. `/app/output/summary.json`: a JSON object with these integer keys:
   - `total_reads`: total number of reads in the FASTQ.
   - `assigned_reads`: number of reads successfully assigned to a probe.
   - `ambiguous_probe_reads`: number of reads where the minimum Hamming distance was <= 2 but multiple probes tied for the minimum.
   - `unassigned_probe_reads`: number of reads where the minimum Hamming distance in both orientations was > 2.
   - `invalid_cell_barcode_reads`: number of reads whose cell barcode was not within 1 mismatch of any whitelist barcode, or matched multiple whitelist barcodes at distance 1.
   - `corrected_cell_barcodes`: number of reads where the cell barcode was corrected to a whitelist barcode (exactly 1 mismatch to exactly one whitelist barcode).
   - `unique_umis_total`: total number of unique (`corrected_cell_barcode`, `umi`, `probe_id`, `sample_id`) tuples across all assigned reads (sum of all `umi_count` values).

## Constraints

1. **Read parsing**: Each read is 100 bp. Cell barcode = positions 1-16, UMI = positions 17-28, probe sequence = positions 29-68. Ignore the rest.
2. **Cell barcode correction**: For each read, compare its 16 bp cell barcode to the whitelist. If it matches exactly, use it. If it is within Hamming distance 1 of exactly one whitelist barcode, correct it to that barcode and count it in `corrected_cell_barcodes`. If it is within distance 1 of multiple whitelist barcodes, or distance > 1 from all, the read is discarded as invalid.
3. **Probe assignment**: For each read, consider both the probe sequence as-is and its reverse complement. For each orientation, compute the Hamming distance to all 600 probes. Let `d_min` be the minimum distance across both orientations. If `d_min <= 2`, the read is assigned to the probe(s) achieving `d_min` only if exactly one probe achieves it. If multiple probes achieve `d_min` (in either orientation), the read is ambiguous. If `d_min > 2` in both orientations, the read is unassigned.
4. **UMI deduplication**: For an assigned read, record a molecule as the tuple (`corrected_cell_barcode`, `umi`, `probe_id`). For each (`sample_id`, `probe_id`), count the number of unique (`corrected_cell_barcode`, `umi`) pairs. Duplicate pairs contribute only once.
5. **Output format**: `probe_counts.tsv` must have the header `probe_id\tsample_id\tumi_count`, all 1800 rows, sorted lexicographically by `probe_id` then `sample_id`. `summary.json` must be valid JSON with exactly the keys listed.
6. **Determinism**: Your output must match the deterministic reference implementation exactly. No randomness or external data allowed.

## Output

- `/app/output/probe_counts.tsv`
- `/app/output/summary.json`

Example row for `probe_counts.tsv`:
```
PRB_0001	SAMPLE_A	5
```

## Scoring

The grader checks 7 constraints independently and computes `score = passed / 7`:
1. `probe_counts.tsv` has correct header and 1800 rows.
2. `summary.json` is valid and contains all required integer keys.
3. `total_reads` matches gold.
4. `invalid_cell_barcode_reads` and `corrected_cell_barcodes` match gold.
5. `assigned_reads`, `ambiguous_probe_reads`, `unassigned_probe_reads` match gold.
6. `unique_umis_total` matches gold.
7. All 1800 `umi_count` values exactly match gold.
