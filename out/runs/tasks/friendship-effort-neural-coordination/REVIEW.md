# Review notes

- Real triad EEG, force, and classroom network data are not redistributable; synthetic participant-level features with the same schema were generated.
- Shortcut probe: test labels are hidden only in tests/grade.py, test IDs do not appear in train.csv, and features are noisy so constant or train-label leakage does not solve the task.
- Known limitations: full time-series force and hyperscanning neural dynamics are simplified to participant-level summary features; friendship status is inferred, not causally estimated.
