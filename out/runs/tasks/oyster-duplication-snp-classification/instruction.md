# Oyster duplication-associated SNP classification

The European flat oyster Ostrea edulis has many SNPs with allelic coverage fractions that deviate from the expected heterozygous 0.5 because of polymorphic duplications. Simplified copy-number classes are: CN2 (normal heterozygous SNP), CN3 (het-mer pattern AAB, minor allele fraction about 1/3), and CN4 (het-mer patterns AAAB or AABB, minor allele fraction about 1/4 or 1/2 with elevated local depth).

Input: /app/environment/data/variants.tsv. Columns: variant_id, chrom, pos, ref, alt, total_depth, minor_allele_depth, baseline_depth. The minor allele depth is always <= half the total depth in this synthetic table.

Task: classify every variant into CN2, CN3, or CN4 using allelic coverage fraction and depth ratio. Write /app/output/predictions.tsv with exactly two tab-separated columns and a header: variant_id and predicted_class. Allowed predicted_class values are CN2, CN3, CN4.

Scoring: macro-averaged F1 over the three classes. Missing or malformed output receives score 0.

Constraints: use only the provided input, no internet access, do not read verifier files. Standard Python libraries are sufficient.
