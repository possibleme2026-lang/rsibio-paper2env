# Review notes

- Data are synthetic but realistic: 80 children, 24 cerebellar regions, behavioral scores generated with embedded true effects in Crus II (ToM) and VIIB/vermis (action prediction). The generator is executed at Docker build time and then deleted, so the agent cannot inspect the true effect parameters.
- Shortcut probe considered: the agent cannot read the gold answer because the generator is removed after build. The grader recomputes the reference analysis from the same data and checks exact region sets, FDR tables, CV R², and network correlations.
- Tool chain: numpy, pandas, scipy, statsmodels, scikit-learn installed via pip. These are canonical Python packages for OLS, FDR, PLS regression, and Pearson correlation.
- Known limitation: the task uses CSV summaries rather than raw NIfTI images, but preserves the statistical multi-step structure (covariate adjustment, multiple comparison correction, cross-validated prediction, network covariance).
