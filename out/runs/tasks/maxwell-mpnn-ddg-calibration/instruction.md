# MAXWELL: calibrating ProteinMPNN log-probability landscapes to mutation-induced stability change

You are reproducing the core of the MAXWELL post-training recipe (Matrix-wise Landscape
Learning). A structure-conditioned inverse-folding protein language model (ProteinMPNN-style)
has been run on 12 proteins and for every residue position it emitted a 20-dimensional score
vector over the canonical amino acids. A parallel set of single point mutations has been
measured experimentally for folding stability change (ddG, kcal/mol). Your job is to turn the
model's *probabilistic output* into a calibrated mutational landscape that predicts ddG for
substitutions that were never measured.

## Inputs

All paths are relative to `/app`.

- `data/mpnn/<protein_id>.csv` - 12 files (P001 ... P012). Header:
  `position,wt,A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y`.
  `position` is 1-based, `wt` is the wild-type one-letter residue, and the remaining 20 columns
  are the language model's **unnormalised per-residue logits** (higher = the model considers
  that amino acid more compatible with the local structure). Protein lengths are 128-230, so
  each file is an L x 20 matrix.
- `data/sequences.fasta` - wild-type sequences for the 12 proteins.
- `data/train_ddg.csv` - 3000 experimentally measured mutations:
  `protein_id,mutation,ddg`. The mutation string is `<wt><position><mut>` (1-based), e.g. `M12A`.
  ddG is in kcal/mol; negative means destabilising.
- `data/test_mutations.csv` - 840 unlabelled mutations in the same string format. Their
  experimental values are held out and are **not** present anywhere in the container.
- `data/landscape_proteins.txt` - three protein ids (one per line) for which a complete
  20-amino-acid landscape is required.

Available Python packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `biopython`.

## Task

1. Parse the L x 20 logit matrices and the wild-type sequence of every protein.
2. Build a **per-mutation feature representation out of the matrix itself** (the logit of the
   mutant residue relative to the logit of the wild-type residue at that position is the natural
   starting point) and learn a calibration from that representation to the measured ddG using
   `data/train_ddg.csv`. Add whatever residue-level context you find necessary and justify your
   feature choices in code comments.
3. Apply the calibrated model to all 840 mutations in `data/test_mutations.csv`.
4. Apply the *same* calibrated model to every (position, amino acid) pair of the three proteins
   listed in `data/landscape_proteins.txt`, producing a complete L x 20 ddG landscape per protein.
5. Write the two output artefacts described below and verify them against the constraints.

## Constraints

1. `output/predictions.csv` must exist with columns `protein_id,mutation,predicted_ddg` and must
   contain **exactly** the 840 mutations of `data/test_mutations.csv` - one row each, no missing
   rows, no duplicates, no extra rows.
2. Every `predicted_ddg` must be a finite number inside the physically plausible window
   `[-9.0, 9.0]` kcal/mol.
3. The Spearman rank correlation between your predictions and the held-out experimental ddG over
   all 840 test mutations must be **>= 0.65**.
4. The RMSE against the held-out experimental ddG must be **<= 1.60** kcal/mol.
5. For each protein P in `data/landscape_proteins.txt`, `output/landscape/P.csv` must exist with
   a header `position,A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y`, one row per position (1..L, in
   order), and every value finite and inside `[-9.0, 9.0]` kcal/mol.
6. Wild-type self-consistency: in every landscape, the cell for substituting a position with its
   own wild-type residue must be within +/- 0.5 kcal/mol of 0 (substituting a residue by itself
   has zero effect by definition).
7. Cross-artefact consistency: for every test mutation that belongs to one of the three landscape
   proteins, the value written in `predictions.csv` must equal the corresponding landscape cell
   to within 1e-3 kcal/mol.

## Output

- `/app/output/predictions.csv`

  ```
  protein_id,mutation,predicted_ddg
  P001,M12A,-1.834
  P001,K45L,0.402
  ...
  ```

  (`predicted_ddg` is signed: negative destabilising, positive stabilising.)

- `/app/output/landscape/<protein_id>.csv` (one per protein listed in
  `data/landscape_proteins.txt`)

  ```
  position,A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y
  1,0.000,-0.412,-0.733,...,0.000
  2,-1.204,-0.556,...,0.110
  ...
  ```

  Row order must follow the 1-based position indexing of `data/mpnn/<protein_id>.csv`.

## Scoring

The grader recomputes the held-out experimental ddG itself and checks each constraint
independently. Partial credit is awarded per constraint with the following weights:

| constraint | weight |
| --- | --- |
| 1 predictions complete & well formed | 0.10 |
| 2 predictions inside [-9, 9] | 0.10 |
| 3 Spearman >= 0.65 (linear ramp from 0.20) | 0.25 |
| 4 RMSE <= 1.60 (linear ramp from 2.40) | 0.15 |
| 5 landscape files present, correct shape & range | 0.15 |
| 6 landscape wild-type diagonal within +/- 0.5 | 0.10 |
| 7 landscape / prediction consistency | 0.15 |

`score` (0.0-1.0) is written to `/logs/verifier/reward.json`. Note that constraints 3 and 4
cannot be passed by predicting a constant, by ignoring the logit matrix, or by using only the
mutant identity; the matrix carries real signal and the calibration must exploit it.
