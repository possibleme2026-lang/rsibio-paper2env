# Enzyme-substrate interaction ranking with co-folding-inspired features

You are an enzyme discovery scientist working on biosynthetic pathway elucidation. You have a panel of 30 enzymes and 150 candidate substrates. Your goal is to prioritize substrates for each enzyme by implementing a deterministic co-folding-inspired scoring pipeline. The pipeline uses sequence-derived active-site features, ligand physicochemical descriptors, global evolutionary context, and known reference interactions to fit a ridge regression model, then ranks all enzyme-substrate pairs.

## Inputs

All files are in `/app/data/`:

- `enzymes.fasta`: 30 enzyme sequences (320-780 aa), one per enzyme.
- `enzyme_annotations.tsv`: tab-separated with columns `enzyme_id`, `family`, `catalytic_residues` (semicolon-separated `H56;D102;S213`), `organism`, `length`.
- `substrates.smi`: 150 substrates, tab-separated `substrate_id` and `SMILES`.
- `reference_pairs.tsv`: 400 known enzyme-substrate pairs with columns `enzyme_id`, `substrate_id`, `interaction_score` (0-1).
- `gen_inputs.py`: script that generated the input files (for reproducibility).

## Task

Implement the following pipeline and write outputs to `/app/output/`:

1. **Ligand descriptors**: For each substrate, compute with RDKit: MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, RingCount, FractionCSP3, FormalCharge. Call them `lig_mw`, `lig_logp`, `lig_tpsa`, `lig_hbd`, `lig_hba`, `lig_rings`, `lig_fsp3`, `lig_charge`.

2. **Active-site pocket features**: For each enzyme, parse `catalytic_residues`. For each catalytic residue at 1-based position `p`, take the window `[max(1,p-8), min(L,p+8)]`. Take the union of all windows as the active-site set `A`. Compute:
   - `pocket_hydrophobicity` = fraction of residues in `A` in {A,V,I,L,M,F,W,Y}.
   - `pocket_polarity` = fraction of residues in `A` in {S,T,N,Q,C,H,K,R,D,E}.
   - `pocket_charge` = (sum over residues in `A` of {D,E:-1, K,R:+1, H:+0.1, else 0}) / |A|.
   - `pocket_aromatic` = fraction of residues in `A` in {F,W,Y,H}.

3. **Global evolutionary context**: For each enzyme, compute the 3-mer frequency vector (with pseudocount 0.5 for each of the 20^3 possible 3-mers). Compute a leave-one-out family consensus: for enzyme `e` in family `f`, average the 3-mer frequency vectors of all other enzymes in family `f`. If family `f` has fewer than 2 other enzymes, use the global average over all other enzymes. Compute the Jensen-Shannon distance (base 2 logarithm) between the enzyme's 3-mer frequency and the consensus. Call this `kmer_jsd`.

4. **Known substrate similarity**: For each enzyme-substrate pair, compute the maximum Tanimoto similarity (Morgan fingerprint, radius 2, 2048 bits) between the candidate substrate and any substrate in `reference_pairs.tsv` with the same enzyme and `interaction_score >= 0.7`. If none, use 0.0. Call this `tanimoto_max`.

5. **Feature vector**: For each of the 30*150 = 4500 pairs, assemble the 14 features in this exact order:
   `lig_mw`, `lig_logp`, `lig_tpsa`, `lig_hbd`, `lig_hba`, `lig_rings`, `lig_fsp3`, `lig_charge`, `pocket_hydrophobicity`, `pocket_polarity`, `pocket_charge`, `pocket_aromatic`, `kmer_jsd`, `tanimoto_max`.

6. **Ridge regression**: Using the 400 reference pairs as training data, standardize each feature to zero mean and unit variance using the reference pairs' mean and population standard deviation (ddof=0; if std=0, set to 1). Center the `interaction_score` by its mean. Fit a ridge regression with intercept (unregularized) and `alpha=1.0` on the standardized features. Predict scores for all 4500 pairs, then clip to [0,1].

7. **Output**: Write two tab-separated files:
   - `/app/output/features.tsv`: columns `enzyme_id`, `substrate_id`, followed by the 14 feature names in the order above. Values rounded to 4 decimal places.
   - `/app/output/predictions.tsv`: columns `enzyme_id`, `substrate_id`, `predicted_score`, `rank`. `rank` is 1..150 within each enzyme (1 = highest score). Scores rounded to 4 decimal places.

## Constraints

1. Ligand descriptors must be computed with RDKit using the specified functions (MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, RingCount, FractionCSP3, FormalCharge). Grader recomputes sampled descriptors and checks tolerance 0.05.
2. Active-site windows must use ±8 residues around each catalytic residue, 1-based inclusive, union of windows. Pocket fractions must use the exact residue sets. Grader recomputes and checks tolerance 0.02.
3. 3-mer JSD must use leave-one-out family consensus, pseudocount 0.5, base-2 logarithm. Grader recomputes and checks tolerance 0.02.
4. Tanimoto similarity must use Morgan fingerprint radius 2, 2048 bits, max over reference substrates with `interaction_score >= 0.7` for the same enzyme. Grader recomputes and checks tolerance 0.05.
5. Ridge regression must use `alpha=1.0`, standardization with reference-pair mean and population std (ddof=0), and clip predictions to [0,1]. Grader compares scores to gold.
6. Output files must have exactly 4500 rows (all pairs), no missing values, correct columns, and ranks must be 1..150 per enzyme with scores non-increasing with rank.
7. Predictive performance: RMSE against gold scores <= 0.15.
8. Predictive performance: average per-enzyme Spearman correlation >= 0.70.
9. Predictive performance: top-5 average overlap with gold >= 0.60.

## Scoring

The grader checks each of the 9 constraints independently. Score = fraction of constraints passed (partial credit). The reward is the same as score.
