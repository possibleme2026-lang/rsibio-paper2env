# Hyperspherical latent geometry for microbiome multi-omics clustering

## Context

Microbiome cohorts span heterogeneous, high-dimensional measurements (taxonomic abundances, host metabolites, host transcripts), and the *geometry* of a shared latent space decides whether the underlying community structure is recovered. DBayesCM compares two variants that differ **only** in latent geometry: `DBayesCM-vMF` constrains the shared latent to the unit hypersphere and fits a von Mises-Fisher (vMF) mixture, while `DBayesCM-GMM` keeps a Euclidean latent and fits a Gaussian mixture on the very same embedding.

You must rebuild that comparison on an unlabelled cohort of 240 subjects and produce a complete, uncertainty-aware clustering report. **There is no ground truth inside the container.** The evaluator holds hidden community labels, hidden discriminative-feature annotations and hidden taxon-host association pairs that were generated from a protected seed.

## Inputs

All files live in `/app/data/` and share the same row order (`S001` ... `S240`):

* `microbiome_abundance.csv` - 240 x 55 matrix of relative abundances in (0, 1); every row sums to 1. Columns `taxon_01` ... `taxon_55`.
* `host_metabolomics.csv` - 240 x 45 matrix of log-scale metabolite intensities. Columns `metab_01` ... `metab_45`.
* `host_transcriptomics.csv` - 240 x 35 matrix of host gene expression values. Columns `tx_01` ... `tx_35`.
* `sample_metadata.csv` - `sample_id, cohort, sex, age, bmi` (nuisance covariates; not required).
* `README.md` - data dictionary.

About one in ten subjects is expected to sit between two community states (transitional microbiomes). Your uncertainty estimates should flag them.

## Task

1. **Shared latent.** Add a pseudocount of 1e-6 to the microbiome table, renormalise each row, and apply the centred log-ratio (CLR) transform per sample. z-score every host feature. Concatenate the 135 transformed features, standardise the columns, and project to a shared 8-dimensional latent with PCA (`n_components=8`, `svd_solver="full"`, `random_state=0`).
2. **Hyperspherical variant (DBayesCM-vMF).** L2-normalise every latent row onto the unit sphere and fit a von Mises-Fisher mixture by Expectation-Maximisation for K = 1..10. Choose K with a penalised-likelihood criterion (record which criterion you used). Output cluster labels and the full posterior responsibility matrix.
3. **Euclidean variant (DBayesCM-GMM).** Fit full-covariance Gaussian mixtures to the *un-normalised* latent Z for K = 1..10, select K by BIC, and output the resulting labels and K.
4. **Uncertainty.** For every sample report the entropy (nats) of the vMF posterior.
5. **Feature selection (spike-and-slab style).** For each of the 135 input features test whether it separates the vMF clusters and report a posterior inclusion probability (PIP) in [0, 1]; a PIP of at least 0.5 counts as selected. Control the false discovery rate across the 135 features.
6. **Conditional co-occurrence.** For each of the 55 x 80 taxon-host pairs report a probabilistic co-occurrence score in [0, 1] (larger = stronger conditional co-occurrence between that taxon and that host feature).

Work in `/app`, write your scripts and intermediate files there, and place the three deliverables in `/app/output/`.

## Constraints

1. **Files and schema.** `/app/output/results.json`, `/app/output/selected_features.csv` and `/app/output/cooccurrence.csv` must exist with exactly the schemas in *Output*. `posterior_probs` must be 240 x `k_vmf`, every row must sum to 1 (tolerance 1e-6) with all entries in [0, 1], and `posterior_entropy` must equal -sum(p log p) of those rows (tolerance 1e-3).
2. **Community recovery.** `cluster_labels_vmf` must be a partition of the 240 samples into exactly `k_vmf` groups whose adjusted Rand index against the hidden community labels is >= 0.40 (full credit at >= 0.85). The evaluator excludes the transitional subjects from this comparison.
3. **Number of communities.** `k_vmf` must equal the true number of community states in the cohort: exact match earns full credit, +/- 1 earns half credit.
4. **Geometry contrast.** `cluster_labels_gmm` and `k_gmm` must come from the Euclidean variant on the same latent (`k_gmm` must equal the number of distinct GMM labels), and the hyperspherical variant must not be worse than it: ARI(vMF) >= ARI(GMM) - 0.01.
5. **Feature selection.** `selected_features.csv` must contain all 135 features with a PIP in [0, 1]. The set with PIP >= 0.5 is scored against the hidden discriminative-feature annotation by F1 (>= 0.30 partial, >= 0.75 full).
6. **Co-occurrence.** `cooccurrence.csv` must contain all 55 x 80 = 4400 ordered pairs with a score in [0, 1]. The ranking is scored against the hidden association pairs by AUROC (>= 0.55 partial, >= 0.80 full).
7. **Uncertainty calibration.** The 72 samples (30%) with the highest `posterior_entropy` must capture the transitional subjects: recall >= 0.70 for full credit.

## Output

**`/app/output/results.json`**

```json
{
  "k_vmf": 5,
  "k_gmm": 6,
  "k_selection_criterion": "BIC on the vMF mixture log-likelihood",
  "cluster_labels_vmf": [0, 2, 1, 1, 4, 3, "... 240 integers in [0, k_vmf)"],
  "cluster_labels_gmm": [0, 2, 1, 3, 5, 4, "... 240 integers in [0, k_gmm)"],
  "posterior_probs": [[0.99, 0.01, 0.0, 0.0, 0.0], "... 240 rows of k_vmf probabilities"],
  "posterior_entropy": [0.02, 0.05, "... 240 floats in nats"]
}
```

**`/app/output/selected_features.csv`** (header + 135 rows, one per input feature)

```
feature_id,pip
taxon_01,0.998
taxon_02,0.031
metab_01,0.874
...
tx_35,0.412
```

**`/app/output/cooccurrence.csv`** (header + 4400 rows)

```
taxon_id,host_feature,probability
taxon_01,metab_01,0.13
taxon_01,metab_02,0.71
...
taxon_55,tx_35,0.05
```

## Scoring

The grader recomputes everything it needs from the protected seed and awards partial credit per constraint. Weights: schema 0.10, ARI(vMF) 0.25, number of communities 0.10, geometry contrast 0.10, feature-selection F1 0.15, co-occurrence AUROC 0.15, uncertainty recall 0.15. `tests/test.sh` writes `{"score": <value>, "reward": <value>}` to `/logs/verifier/reward.json`.
