# Review notes

- Data substitution: HCP fMRI and structural connectomes cannot be redistributed directly; a small synthetic anonymized dataset with lag-1 autocorrelation and standardized SC distances is provided.
- Shortcut probe: considered exposing condition names or true tau labels; condition IDs are anonymized, rest is not first by ID, and gold labels are only in tests/.
- Known limitations: the task uses a simplified four-region, lag-1 INT approximation rather than full whole-brain fMRI autocorrelation fitting.
