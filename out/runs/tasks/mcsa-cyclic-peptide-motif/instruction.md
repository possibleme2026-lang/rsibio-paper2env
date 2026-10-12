# Recover the conserved binding motif from a library of head-to-tail cyclic peptides

You are a computational biologist working on a library of 1,000 head-to-tail (H2T) cyclized peptides targeting the oncoprotein MDM2. The peptides are length 18 and are represented as linear amino-acid strings, but because they are cyclized, the linear representation is only defined up to a circular rotation. To discover the conserved positional motif responsible for MDM2 binding, you must solve the multiple circular sequence alignment (MCSA) problem: find for each peptide the rotation that aligns the conserved motif across all sequences.

## Inputs
- `/app/data/cyclic_peptides.csv`: CSV with header `peptide_id,sequence,cyclic`. Contains 1,000 rows. `sequence` is a length-18 string of one-letter amino-acid codes (valid alphabet: ACDEFGHIKLMNPQRSTVWY). `cyclic` is `true`. The sequences are already circularly permuted by an unknown offset.
- The conserved motif is exactly 8 residues long and is embedded once per sequence. It may be mutated at individual positions (roughly 10% of positions deviate from the consensus).

## Task
1. Load the peptide library.
2. Implement an MCSA algorithm to determine the circular rotation offset for each peptide that aligns the conserved 8-residue motif across the library.
3. Derive the consensus motif (the most frequent residue at each of the 8 aligned positions).
4. Canonicalise the rotations so that the recovered motif starts at position 0 (i.e., after applying the rotation, the first residue of the motif is at index 0 in every sequence).
5. Write the results to `/app/output/mcsa_results.json`.

## Constraints
1. Output JSON must contain exactly two top-level keys: `consensus_motif` (string of length 8) and `rotations` (object mapping every `peptide_id` to an integer rotation offset).
2. `rotations` must contain exactly 1,000 entries, one for each peptide ID in the input, and each value must be an integer in [0, 17].
3. For each peptide, applying the reported rotation (left rotation by `rotations[peptide_id]`) must place the conserved motif at index 0. The grader checks this against a hidden ground truth; partial credit is given for the fraction of peptides correctly rotated.
4. `consensus_motif` must be exactly 8 characters, using only the standard 20 amino-acid one-letter codes, and must match the hidden consensus motif (partial credit per position).
5. Self-consistency: after applying your rotations, at each of the 8 motif positions, the residue frequency of your reported consensus character must be at least 0.70 across the 1,000 aligned sequences.
6. The output file must be valid JSON and be written to the specified path.

## Output
`/app/output/mcsa_results.json`:
```json
{
  "consensus_motif": "ACDEFGHI",
  "rotations": {
    "PEP0001": 5,
    "PEP0002": 12,
    ...
    "PEP1000": 0
  }
}
```
Note: The example consensus motif `ACDEFGHI` is a placeholder; do not use it.

## Scoring
The grader checks constraints 1–6 independently and computes a weighted score:
- 0.10 for valid JSON and correct schema
- 0.10 for complete and valid rotations map
- 0.45 for rotation accuracy (fraction of peptides whose rotation places the hidden motif at index 0)
- 0.25 for consensus motif accuracy (fraction of 8 positions matching hidden consensus)
- 0.10 for self-consistency of the reported consensus with the aligned sequences
