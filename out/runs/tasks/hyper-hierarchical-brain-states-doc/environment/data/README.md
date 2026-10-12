Synthetic resting-state fMRI cohort for the hyper-hierarchical brain state task.

Files are produced by gen_inputs.py with the fixed seed 20260730:
  subjects.csv      45 subjects (CNT/MCS/UWS), ages, sexes, CRS-R scores
  roi_networks.csv  100 ROIs with their functional network label
  bold/*.csv        300 time points x 100 ROIs, TR = 2.0 s (one file per subject)

Run `python gen_inputs.py --out .` to regenerate the whole cohort deterministically.
CRS-R: Coma Recovery Scale-Revised (0-23); patients only.
