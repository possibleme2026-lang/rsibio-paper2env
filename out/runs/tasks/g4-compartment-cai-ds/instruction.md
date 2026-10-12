# Compartment-resolved G-quadruplex selection audit

You are auditing the G4 paradox on a synthetic eukaryotic pangenome. The dataset contains four species (one per kingdom) with chromosome sequences, gene annotations, orthologous gene pairs, and codon usage tables. Your task is to compute G4 densities per compartment (CDS, intron, promoter) and to test whether CAI adjustment removes the apparent dS suppression at G4 codons.

## Inputs
- `/app/data/genomes/{species}.fa`: 4 FASTA files, each 2 chromosomes x 300,000 bp.
- `/app/data/annotation.gff3`: combined GFF3 with gene, CDS, intron features for all species. SeqIDs are `{species}_chr{1,2}`.
- `/app/data/ortholog_pairs.tsv`: table with columns pair_id, species1, species2, file.
- `/app/data/orthologs/{pair_id}.fa`: aligned CDS sequences for each ortholog pair.
- `/app/data/codon_usage.tsv`: RSCU table for each species and codon.

## Task
1. Parse the GFF3 and FASTA files. Define compartments:
   - CDS: union of all CDS features.
   - Intron: union of all intron features.
   - Promoter: 1000 bp upstream of the TSS. For a + strand gene, TSS is the start coordinate (1-based); promoter = [start-1000, start-1]. For a - strand gene, TSS is the end coordinate; promoter = [end+1, end+1000]. Truncate to chromosome bounds. Merge overlapping intervals per chromosome before computing lengths and assigning motifs.
2. Identify G4 motifs using the exact rule: regex `G{3,5}(?:[ACGT]{1,7}G{3,5}){3}` with total length <= 45. Scan both strands, merge overlapping intervals into single loci.
3. For each species, compute for each compartment:
   - `g4_counts`: number of G4 loci whose midpoint falls in the compartment.
   - `g4_density_per_kb`: g4_counts / (compartment length in kb).
   - `log2_IRR`: log2( (g4_count / length_kb) / (total_g4 / total_length_kb) ) where total is the sum over the three compartments for that species.
4. For each species pair in `ortholog_pairs.tsv`:
   - For each codon in the first sequence of each pair, determine if it is a G4 codon (any nucleotide overlaps a G4 motif in the first sequence).
   - Determine if the codon pair is synonymous (same amino acid).
   - Compute naive OR = (S_G4 / N_G4) / (S_nonG4 / N_nonG4) across all genes in the species pair.
   - Compute CAI for each gene from the first sequence using the species1 codon usage table (see CAI definition below).
   - Fit a logistic regression: `synonymous ~ is_G4 + CAI` across all codons in the species pair. Report `adjusted_OR = exp(beta_G4)`.
5. Compute summary counts: number of species pairs with naive_OR < 1, and with adjusted_OR < 1.
6. Compute CAI for the first three pairs (`pair_000`, `pair_001`, `pair_002`).

## CAI definition
For a sequence, split into codons (ignore incomplete final codon). For each codon, w = RSCU(codon) / max RSCU among synonymous codons for the same amino acid. If w == 0, set w = 0.5. CAI = exp( (1/L) * sum(ln w) ), where L is the number of codons.

## Constraints
1. G4 motif definition must be exactly as specified (regex, length <= 45, both strands, merge overlapping).
2. Compartment definitions must follow the exact TSS/promoter rules and merging of intervals.
3. Density and log2 IRR calculations must use the total over the three compartments for that species.
4. Naive OR must use the exact counts S_G4, N_G4, S_nonG4, N_nonG4 summed across all genes in the species pair.
5. CAI must be computed with the RSCU table and the specified w formula.
6. Logistic regression must include both `is_G4` and `CAI` as covariates; adjusted_OR is exp(beta_G4).
7. Output JSON must have the exact keys and structure.

## Output
Write `/app/output/result.json` with the schema shown below. Example:
```json
{
  "species": {
    "Homo_sapiens": {
      "g4_counts": {"CDS": 10, "intron": 50, "promoter": 30},
      "g4_density_per_kb": {"CDS": 0.12, "intron": 0.45, "promoter": 0.38},
      "log2_IRR": {"CDS": -1.52, "intron": 0.58, "promoter": 0.33}
    },
    ...
  },
  "species_pairs": {
    "Homo_sapiens-Arabidopsis_thaliana": {
      "naive_OR": 0.65,
      "adjusted_OR": 1.12,
      "n_genes": 20
    },
    ...
  },
  "summary": {
    "n_pairs_naive_OR_lt_1": 6,
    "n_pairs_adjusted_OR_lt_1": 0
  },
  "cai_examples": {
    "pair_000": 0.72,
    "pair_001": 0.65,
    "pair_002": 0.80
  }
}
```

## Scoring
The grader checks six independent constraints. Score = (number of fully passed constraints) / 6. Partial credit is only across constraints, not within a constraint.
