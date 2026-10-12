# PPO2 target-site resistance detection from Nanopore amplicon sequencing

You are a computational biologist analyzing herbicide resistance in common ragweed (*Ambrosia artemisiifolia*). Farmers in Michigan reported failures of PPO-inhibiting herbicides. Nanopore amplicon sequencing was performed on the PPO2 gene from three populations: a susceptible reference (S) and two suspected resistant populations (R1, R2). Your job is to identify the target-site mutations at codon 98 of PPO2, quantify their allele frequencies, classify resistance, and design a diagnostic allele-specific primer pair for the novel R98Q mutation.

## Inputs
- `/app/data/reference.fasta`: PPO2 coding sequence (CDS) reference (~1.7 kb). Codon 98 is at nucleotide positions 292-294 (1-based). The wild-type codon is CGA (Arg).
- `/app/data/reads/S.fastq.gz`, `/app/data/reads/R1.fastq.gz`, `/app/data/reads/R2.fastq.gz`: Nanopore amplicon reads (~500 bp) covering the region around codon 98. Reads contain sequencing errors typical of Nanopore (substitutions, insertions, deletions). Each file has ~2000-2500 reads.
- The amplicon covers positions 1-600 of the reference.

## Task
1. Align the reads from each population to the reference CDS. Use `minimap2` (or another aligner) and `samtools`/`bcftools` if desired.
2. For each population, call the variants at codon 98. Consider the three possible alleles: WT (CGA, Arg), R98L (CTA, Leu), and R98Q (CAA, Gln). Note: the mutation site is the second base of codon 98 (position 293, 1-based).
3. Estimate the allele frequency of each of the three alleles in each population. Frequencies must sum to 1.0.
4. Determine whether each population carries R98L and/or R98Q (presence defined as frequency > 0.05).
5. Classify each population as resistant or susceptible. Resistance is conferred by the presence of R98L or R98Q.
6. Design a diagnostic allele-specific primer pair for the R98Q allele. One primer must have its 3' terminal base at the mutation site (position 293) and match the R98Q allele (i.e., the 3' base must be complementary to the R98Q sense base A: forward primer ends in A, reverse primer ends in T). The product size must be 150-300 bp. Both primers must have Tm 58-62°C, GC content 40-60%, and no four consecutive identical bases. Use `primer3` if helpful.

## Constraints
1. Output must be a valid JSON file at `/app/output/result.json` with the exact schema described below.
2. For each population, the three allele frequencies (WT_R98, R98L, R98Q) must sum to 1.0 ± 0.01.
3. Allele frequency estimates must be within ±0.05 of the true underlying frequencies.
4. Presence/absence calls (frequency > 0.05) for R98L and R98Q must be correct for each population.
5. Resistance classification (true/false) must be correct for each population.
6. The diagnostic primer pair must target R98Q: one primer's 3' end must be at position 293 and match the R98Q allele.
7. The primer pair must amplify a product of 150-300 bp. Both primers must satisfy Tm 58-62°C, GC 40-60%, and no four consecutive identical bases.
8. Report the product size in the JSON and ensure it matches the primer binding positions on the reference.

## Output
Write a JSON file to `/app/output/result.json` with this structure:
```json
{
  "populations": {
    "S": {
      "variant_frequencies": {"WT_R98": 0.0, "R98L": 0.0, "R98Q": 0.0},
      "carries_R98L": false,
      "carries_R98Q": false,
      "resistant": false
    },
    "R1": {
      "variant_frequencies": {"WT_R98": 0.0, "R98L": 0.0, "R98Q": 0.0},
      "carries_R98L": false,
      "carries_R98Q": false,
      "resistant": false
    },
    "R2": {
      "variant_frequencies": {"WT_R98": 0.0, "R98L": 0.0, "R98Q": 0.0},
      "carries_R98L": false,
      "carries_R98Q": false,
      "resistant": false
    }
  },
  "diagnostic_primers": {
    "forward": "ACGTACGTACGTACGTACGT",
    "reverse": "TGCATGCATGCATGCATGCA",
    "product_size": 212,
    "target_allele": "R98Q",
    "primer_with_3p_mutation": "forward"
  }
}
```
(Note: the values above are placeholders only; you must compute the actual values from the provided reads.)

## Scoring
The grader checks each constraint independently. Partial credit is given: score = (number of passed checks) / (total checks). Checks include schema validity, frequency sums, frequency accuracy, presence calls, resistance calls, primer mutation-site positioning, product size, and primer thermodynamic properties.
