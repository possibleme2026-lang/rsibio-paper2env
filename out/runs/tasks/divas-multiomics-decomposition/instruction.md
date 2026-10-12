# Multiomics Decomposition with DIVAS

Multiomics data integration aims to identify biological patterns shared across molecular modalities. Most existing methods detect either jointly shared variation (across all modalities) or individual variation (unique to a single modality), but overlook partially shared variation (shared by a subset of modalities). The DIVAS framework combines angle-based subspace analysis with rotational bootstrap to hierarchically search all combinations of modalities and decompose multiomics data into interpretable components with scores and loadings.

You are given three omics matrices from the same set of 60 samples: transcriptomics (200 genes), proteomics (150 proteins), and metabolomics (120 metabolites). Your task is to apply a DIVAS-style decomposition to identify jointly shared, partially shared, and individual components, and report them in a standardized format.

## Inputs
- `/app/data/transcriptomics.csv`: 60 samples × 200 genes (log2 TPM). First column `sample_id`, remaining columns gene IDs.
- `/app/data/proteomics.csv`: 60 samples × 150 proteins (log2 abundance). First column `sample_id`, remaining columns protein IDs.
- `/app/data/metabolomics.csv`: 60 samples × 120 metabolites (log2 intensity). First column `sample_id`, remaining columns metabolite IDs.
- `/app/data/samples.csv`: sample IDs.

## Task
1. Load and preprocess the three matrices. Standardize features (z-score) and ensure samples are aligned.
2. Apply the DIVAS method (or an equivalent angle-based subspace analysis with rotational bootstrap) to decompose the multiomics data into components. You may use the R package `DIVAS` installed in the container, or implement the method in Python.
3. Determine the number of components using a data-driven criterion (e.g., variance explained, bootstrap stability). The true data contain several joint, partial, and individual components.
4. For each detected component, record:
   - `component_id`: a unique string.
   - `sharing_type`: one of `joint` (all three modalities), `partial` (exactly two modalities), `individual` (one modality).
   - `modalities`: semicolon-separated list of modalities involved, using the names `transcriptomics`, `proteomics`, `metabolomics`.
   - `variance_explained`: the proportion of total variance in the entire dataset explained by this component (sum of squared loadings times score variance, divided by total variance of all features).
5. For each component, provide the sample scores in `scores.csv` (columns: `sample_id`, `component_id`, `score`). Scores must be standardized (mean 0, sd 1 across samples for each component).
6. For each modality, provide the feature loadings in `loadings_<modality>.csv` (columns: `feature_id`, `component_id`, `loading`). Only include loadings for components that involve that modality; loadings for other components should be omitted or zero.
7. Write all output files to `/app/results/`.

## Constraints
1. Output files `components.csv`, `scores.csv`, `loadings_transcriptomics.csv`, `loadings_proteomics.csv`, `loadings_metabolomics.csv` must exist in `/app/results/` with the exact columns specified.
2. The number of detected components must be between 10 and 16 (the true structure has 13 components). `sharing_type` must be consistent with the number of modalities in `modalities` (joint=3, partial=2, individual=1).
3. For each component, the scores must have mean absolute value < 0.1 and standard deviation between 0.9 and 1.1.
4. The absolute Pearson correlation between scores of any two distinct components must be < 0.3 (orthogonality).
5. For each component, the reported `variance_explained` must match the value recomputed from the provided loadings and scores within 30% relative tolerance.
6. The sum of `variance_explained` over all components must be ≤ 1.0 and ≥ 0.5.
7. For each component, the modalities listed in `modalities` must have non-zero loadings (at least one loading with absolute value > 0.02) and modalities not listed must have no loadings in the corresponding files.

## Output
- `/app/results/components.csv`: component_id, sharing_type, modalities, variance_explained
- `/app/results/scores.csv`: sample_id, component_id, score
- `/app/results/loadings_transcriptomics.csv`: feature_id, component_id, loading
- `/app/results/loadings_proteomics.csv`: feature_id, component_id, loading
- `/app/results/loadings_metabolomics.csv`: feature_id, component_id, loading

Example `components.csv`:
```
component_id,sharing_type,modalities,variance_explained
comp_01,joint,transcriptomics;proteomics;metabolomics,0.12
comp_02,partial,transcriptomics;proteomics,0.08
comp_03,individual,transcriptomics,0.05
```

## Scoring
The grader evaluates:
- Format and constraint compliance (30%): files exist, columns correct, sharing types consistent, scores standardized, orthogonality, variance explained consistency, sum of variance explained.
- Component recovery (50%): matches detected components to true latent components using score correlations. A true component is recovered if matched with |r| ≥ 0.7. The F1 score over components is computed.
- Sharing structure accuracy (20%): among recovered components, the fraction with exactly correct `modalities` set.
Final score = 0.3 * format_score + 0.5 * F1 + 0.2 * sharing_accuracy. Partial credit is given for each constraint.
