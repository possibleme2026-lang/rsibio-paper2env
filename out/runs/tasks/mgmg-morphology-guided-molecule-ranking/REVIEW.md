# Review notes

- Data is synthetic but realistic: 500 reference compounds with 256 morphological features, 20000 candidate molecules with RDKit-computable SMILES and 32-dim text embeddings.
- The generator script is visible but contains only raw data generation; the gold top-20 is computed by the grader using the specified deterministic algorithm, so reading the generator does not reveal the answer.
- Real domain tools: RDKit for chemistry (Morgan fingerprints, descriptors, PAINS, SA score) and scikit-learn for Ridge regression.
- Partial credit: 10 independently checkable constraints, score = fraction passed.
- Known limitation: the pipeline is a simplified linear projection baseline, not the full MGMG transformer; it still requires multi-step feature engineering, model training, filtering, and diversity selection.
