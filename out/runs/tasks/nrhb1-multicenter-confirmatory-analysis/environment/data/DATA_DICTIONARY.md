# Data dictionary — rhNRGβ1 multicenter confirmatory study (synthetic)

All files in this directory are produced by `gen_inputs.py` with a fixed seed
(`SEED = 20260731`); re-running the script reproduces them byte for byte.

| file | rows | columns | notes |
| --- | --- | --- | --- |
| `nerve_thickness.csv` | 216 | `mouse_id, site, sex, arm, baseline_um, week13_um` | one row per randomised mouse; `week13_um` is empty for mice lost to follow-up |
| `western_blot.csv` | 144 | `mouse_id, site, sex, arm, actin, MBP, ERBB2, pERBB2, ERK1_2, pERK1_2, YAP1` | raw densitometry intensities (arbitrary units); `actin` is the loading control |
| `proteomics_panel.tsv` | 12960 | `sample_id, site, sex, arm, protein, intensity` | long-format DIA quantification, 216 samples x 60 panel proteins; `intensity` is linear and empty when not quantified |
| `protein_panel_pathways.tsv` | 60 | `protein, pathway` | membership of the 60 panel proteins in `Hippo_YAP`, `mTORC1`, `RTK_Ras_MAPK`, `Differentiation`, `Metabolic` |

The cohort is balanced: 3 sites x 2 arms x 2 sexes x 18 mice = 216.
