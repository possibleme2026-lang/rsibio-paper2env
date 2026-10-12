# CPPLocPred: Subcellular Localization of Cell-Penetrating Peptides

You are working on the second stage of CPPLocPred, a machine-learning framework for predicting the subcellular localization of cell-penetrating peptides (CPPs). The goal is to reproduce the paper's approach: compute Distance Distribution of Residues (DDR) features from peptide sequences and train a CatBoost multiclass classifier to predict one of five localizations: Cytoplasm, Nucleus, Mitochondria, Endo_lysosome, Others.

## Inputs
- `/app/data/train.csv`: 1216 CPP sequences with labels (columns: seq_id, sequence, localization).
- `/app/data/val.csv`: 305 CPP sequences with labels (columns: seq_id, sequence, localization).
- `/app/data/test.csv`: 500 CPP sequences without labels (columns: seq_id, sequence).
- `/app/data/README.md`: additional context.

## Task
1. Implement DDR feature extraction exactly as specified below.
2. Train a CatBoost multiclass classifier on the combined train and validation sets using the following fixed hyperparameters:
   - iterations=500
   - learning_rate=0.1
   - depth=6
   - random_seed=42
   - loss_function='MultiClass'
   You may use the validation set for early stopping if you wish, but the final model must be trained on all labeled data (train+val) before predicting the test set.
3. Predict class probabilities for the test sequences.
4. Output three files:
   - `/app/predictions.csv`: columns `seq_id, Cytoplasm, Nucleus, Mitochondria, Endo_lysosome, Others`. Probabilities must be in [0,1] and sum to 1 per row (tolerance 0.01).
   - `/app/test_ddr_features.csv`: DDR features for the test sequences. First column `seq_id`, followed by 200 columns named exactly `DDR_<AA>_<bin>` where `<AA>` iterates over the 20 standard amino acids in alphabetical order (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y) and `<bin>` is 0..9.
   - `/app/val_metrics.json` (optional): per-class AUC on the validation set.

### DDR specification
For a peptide sequence of length L:
- For each amino acid type `t` in the 20-letter alphabet:
  - Find all positions where the residue is `t`.
  - For every pair of such positions (i, j) with i < j, compute the distance `d = j - i`.
  - Bin `d` into 10 bins using the following edges:
    - bin 0: d = 1
    - bin 1: d = 2
    - bin 2: 3 <= d <= 4
    - bin 3: 5 <= d <= 6
    - bin 4: 7 <= d <= 9
    - bin 5: 10 <= d <= 13
    - bin 6: 14 <= d <= 18
    - bin 7: 19 <= d <= 24
    - bin 8: 25 <= d <= 30
    - bin 9: d >= 31
  - Count the number of pairs in each bin, then normalize the 10-bin histogram by the total number of pairs for residue type `t`. If there are zero pairs, set all bins to 0.
- Concatenate the normalized histograms for all 20 amino acids in alphabetical order to obtain 200 features per sequence.

## Constraints
1. `predictions.csv` must contain exactly the test seq_ids (no missing, no extras, no duplicates).
2. `predictions.csv` must have exactly the columns `seq_id, Cytoplasm, Nucleus, Mitochondria, Endo_lysosome, Others`.
3. All probability values must be in [0,1] and each row must sum to 1 within ±0.01.
4. `test_ddr_features.csv` must have exactly 201 columns: `seq_id` followed by 200 DDR columns named as specified.
5. The DDR features for any test sequence must match the reference implementation within 1e-6.
6. The per-class AUC on the hidden test set must be at least the reference CatBoost model's AUC minus 0.05 for each of the five classes.

## Output
- `/app/predictions.csv` (example row):
```
seq_id,Cytoplasm,Nucleus,Mitochondria,Endo_lysosome,Others
test_Cyt_0001,0.85,0.05,0.02,0.03,0.05
```
- `/app/test_ddr_features.csv` (example header):
```
seq_id,DDR_A_0,DDR_A_1,...,DDR_Y_9
```

## Scoring
The grader checks each of the 10 constraints above (constraint 6 is split into 5 per-class AUC checks, making 10 total). Score = fraction of passed constraints. Reward = score.
