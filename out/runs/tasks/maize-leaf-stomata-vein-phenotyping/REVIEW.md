# Review notes

- Data is synthetic and generated programmatically to mimic maize leaf epidermis microscopy images with realistic stomatal and vein structures.
- The generator script is run during Docker build and deleted, so the agent cannot read the true trait values or QTL effects.
- Shortcut probe: agent might try to reverse-engineer the generator; mitigated by deleting it at build time.
- Tool chain: Python with OpenCV, scikit-image, scipy, pandas, numpy, statsmodels. The reference solution uses these for image analysis and QTL mapping.
- The grader recomputes true traits from a hidden gold generator that replicates the build-time generation with the same seed.
- Known limitation: the synthetic images are simpler than real microscopy; performance on real data may differ.
