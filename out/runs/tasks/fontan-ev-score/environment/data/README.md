# Cohort data (synthetic, deterministic)

`clinical.csv` and `mirna_counts.csv` are generated at image build time by
`gen_inputs.py` (seed 20260731). They emulate the translational ovine Fontan
model of the Fontan EV Score study: 19 sheep (13 Fontan, 6 Sham), longitudinal
serum sampling between 3 and 12 months post-surgery, 58 serum small-EV
small-RNA libraries over 180 miRNAs, clinical chemistry, elastography and an
ordinal fibrosis grade (0-3), plus a pre-specified train/test split.
