# Review notes

- Real EEG data and pretrained CREST weights cannot be redistributed; the environment ships a deterministic synthetic 125-subject spectrogram feature matrix with the same 56-region x 5-band schema.
- Shortcut probe: considered that model weights and the feature generator are visible; hidden gold labels are regenerated only in tests/grade.py from a noisy top-74 rule and are not stored in /app/environment.
- Known limitations: the task uses a simplified ring graph and linear band weights instead of the paper's frozen CNN/graph transformer, and synthetic labels are not clinical ground truth.
