# Predict fish life-history strategy axes from lateral-view morphology

## Background

Life-history strategies emerge from trade-offs between growth, survival and reproduction, and are key predictors of how populations respond to disturbance - but the demographic data needed to estimate them are missing for most fish species. Because morphological traits govern whole-organism performance, they are candidate low-cost proxies for life-history strategy. In the source study, dynamic energy budget integral projection models (DEB-IPMs) were parameterised for 290 marine and freshwater fish species, eight morphological traits were measured from lateral-view photographs, and phylogenetically corrected principal component analysis was used to test whether morphology predicts life-history strategy. The two leading life-history axes were generation turnover (years) and reproductive output (index). Lateral-size morphology - body elongation, relative eye size and oral gape position - predicted generation turnover depending on water column position, and predicted reproductive output depending on clade.

You are given morphology, ecology and a molecular phylogeny for these species plus the two life-history axes for 238 of them. Demographic data are absent for the remaining 50 species, whose life-history axes you must predict.

## Inputs (all in `/app/data`)

* `morphology.csv` - 290 species, columns: `species` (names contain spaces), `clade`, `water_column` (pelagic | demersal | benthic | reef) and eight morphological traits: `body_elongation`, `relative_eye_size`, `oral_gape_position`, `caudal_fin_ratio`, `pectoral_fin_ratio`, `mouth_width_ratio`, `head_length_ratio`, `body_depth_ratio`. A small number of trait cells are blank (missing at random).
* `life_history_train.csv` - 238 species with `generation_turnover_years` and `reproductive_output_index` (the two leading axes of a phylogenetically corrected PCA of the DEB-IPM life-history traits).
* `phylogeny.nwk` - ultrametric Newick phylogeny of 288 species; tip labels use underscores instead of spaces, branch lengths in relative time units.
* `test_species.txt` - the 50 species for which predictions are required, one name per line, spelled as in `morphology.csv`.

## Task

1. Reconcile identifiers (tree tips use underscores, the tables use spaces) and keep only species that have both measurements and a phylogenetic placement. Two species in `morphology.csv` have neither life-history data nor a tip in the tree and must be dropped.
2. Impute the blank morphological values with the median of that trait within the species' clade, computed over the species kept in step 1.
3. Run a phylogenetic PCA of the eight morphological traits, exactly as follows:
   a. z-score every trait across the kept species (mean 0, sample sd 1);
   b. build the Brownian-motion covariance matrix `C` from the phylogeny, with `C_ij` = root-to-MRCA distance of species i and j and `C_ii` = total root-to-tip height;
   c. whiten the z-scored matrix: `X_w = C^{-1/2} X`, where `C^{-1/2}` is the symmetric inverse square root obtained from the eigendecomposition of `C`;
   d. centre the columns of `X_w` and run a PCA (eigendecomposition of the sample covariance matrix of `X_w`);
   e. report the fraction of total variance explained by PC1 and the eight PC1 loadings in the order of the trait list above, with the sign of PC1 fixed so that the loading of `body_elongation` is positive.
4. Fit a phylogenetic generalised least squares (PGLS) model of `log(generation_turnover_years)` on the three lateral-size traits. Standardise `body_elongation`, `relative_eye_size` and `oral_gape_position` with the mean and sample sd of the 238 training species; the design matrix is `[1, z(body_elongation), z(relative_eye_size), z(oral_gape_position)]` and the GLS estimate is `beta = (X' C^-1 X)^-1 X' C^-1 y`, using the `C` sub-matrix of those 238 species. Report the three slopes (excluding the intercept).
5. Fit models that predict `log(generation_turnover_years)` and `log(reproductive_output_index)` from the eight morphological traits for the 238 training species, letting the effect of the lateral-size traits vary with `water_column` for generation turnover and with `clade` for reproductive output. Validate with cross-validation and report the cross-validated RMSE of both log-scale models.
6. Apply the fitted models to the 50 species listed in `test_species.txt` and write the predictions on the original (untransformed) scale.

## Constraints

1. `/app/output/predictions.csv` must exist with header `species,generation_turnover,reproductive_output` and exactly 50 data rows, one per species in `test_species.txt`, using the same names; no duplicates, no extra or missing species, all values finite numbers.
2. Predicted `generation_turnover` must lie in (0.1, 100) years and predicted `reproductive_output` in (0.005, 500); values outside these ranges are rejected.
3. `/app/output/diagnostics.json` must exist and contain: `n_species_analysed`, `n_training_species`, `phylo_pca_morphology.pc1_variance_explained`, `phylo_pca_morphology.pc1_loadings` (a dict with exactly the eight trait names as keys and finite floats as values), `pgls_log_generation_turnover` (dict keyed by `body_elongation`, `relative_eye_size`, `oral_gape_position`), `cv_rmse_log_generation_turnover` and `cv_rmse_log_reproductive_output` (strictly positive finite floats). No NaN or Infinity anywhere in the file.
4. `n_species_analysed` must equal the number of species passing step 1 and `n_training_species` the number of those that have life-history data; both are checked against the reference reconstruction.
5. The reported PC1 loadings and variance-explained fraction must correspond to the phylogenetic PCA defined in step 3 (they must not come from an ordinary PCA of the untransformed traits); each loading is compared with the reference value.
6. The three reported PGLS slopes must correspond to the model defined in step 4 (phylogenetic GLS, z-scored traits); each slope is compared with the reference value.
7. Predictions are scored against the withheld life-history values by the Pearson correlation between log10(predicted) and log10(true) and by the Spearman rank correlation, whichever is larger. Full credit at r >= 0.72 for generation turnover and r >= 0.65 for reproductive output.
8. Both output files must be written under `/app/output/` and every number in `diagnostics.json` must be reproducible from the provided inputs alone (no unseeded random components).

## Output

`/app/output/predictions.csv` (any row order):

```
species,generation_turnover,reproductive_output
Gadus morhua,12.4500,1.8300
Salmo trutta,3.1200,0.9100
```

`/app/output/diagnostics.json` - schema as in constraint 3, for example:

```json
{
  "n_species_analysed": 288,
  "n_training_species": 238,
  "phylo_pca_morphology": {
    "pc1_variance_explained": 0.0,
    "pc1_loadings": {"body_elongation": 0.0, "relative_eye_size": 0.0,
                     "oral_gape_position": 0.0, "caudal_fin_ratio": 0.0,
                     "pectoral_fin_ratio": 0.0, "mouth_width_ratio": 0.0,
                     "head_length_ratio": 0.0, "body_depth_ratio": 0.0}
  },
  "pgls_log_generation_turnover": {"body_elongation": 0.0, "relative_eye_size": 0.0,
                                    "oral_gape_position": 0.0},
  "cv_rmse_log_generation_turnover": 0.0,
  "cv_rmse_log_reproductive_output": 0.0
}
```

## Scoring

The grader regenerates the held-out life histories from a protected seed, recomputes the reference diagnostics itself and scores ten independently checked constraints with partial credit: prediction file format (0.08), test-species coverage (0.05), biological range validity (0.05), diagnostics schema (0.04), sample counts (0.08), phylogenetic PC1 variance explained (0.08), the eight PC1 loadings (0.12, graded per loading), the three PGLS slopes (0.15, graded per slope), generation turnover accuracy (0.20, linear ramp between r = 0.30 and 0.72), reproductive output accuracy (0.15, linear ramp between r = 0.30 and 0.65). The final score is the weighted sum and is written to `/logs/verifier/reward.json`.
