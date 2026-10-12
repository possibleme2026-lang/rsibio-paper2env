# Evaluate GONNECT-style GO-constrained latent activations

You are given a synthetic RNA-seq expression matrix and a simplified Gene Ontology (GO) gene set table. Implement a faithful simplified GONNECT encoder: a GO-constrained linear autoencoder with one latent node per GO term. For each GO term, infer a sample-level latent activation from the expression of its member genes. A mean/z-score/ridge linear GO-constrained encoder is acceptable; you do not need to train a deep network.

Inputs:
- /app/data/expression.csv: rows are samples (sample_id), columns are genes (G1...G12).
- /app/data/go_terms.csv: columns term_id,gene_id. Latent nodes correspond to term_id values.

Output:
Write /app/output.json with this exact schema:
{
  "samples": ["sample_id", ...],
  "terms": ["term_id", ...],
  "activations": [[float, ...], ...]
}
Rows in activations must align with samples; columns must align with terms.

Scoring:
For each term, compute Pearson correlation between your predicted activations and hidden gold activations over all samples. The final score is the mean Pearson correlation, clipped to [0, 1].

Constraints:
- Use only files under /app/data.
- Do not read /tests.
- No internet access.
- Do not assume the hidden gold is available.
