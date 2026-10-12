# Review notes

- Real DDM image sequences from Xenopus egg extracts and HeLa cells are not redistributable; synthetic preprocessed effective decay curves E(q,t) were generated with the same schema and fBm scaling behavior.
- Shortcut probe: checked that gold parameters are not exposed in filenames, comments, or instruction examples; hidden gold lives only in tests/grade.py, and the input includes an offset so raw first values are not the answer.
- Known limitations: this is a simplified single-mode fBm fit, not the full inverse-Gaussian diffusivity distribution, non-Gaussian displacement analysis, or two-state fBm model described in the paper.
