# Kukulu CDR-H3 Loop Reconstruction

You are implementing a computational pipeline for antibody CDR-H3 loop reconstruction, inspired by the Kukulu paper (structure-aware JEPA + diffusion). The task is to reconstruct missing CDR-H3 loops in a set of antibody-antigen complexes.

## Inputs
- `/app/data/complexes/complex_XX.pdb`: 8 synthetic antibody-antigen complexes. Each has chain H (heavy), chain L (light), and chain A (antigen). In chain H, the CDR-H3 loop is missing (a gap in residue numbering). All atoms outside the gap are present.
- `/app/data/fragments/frag_XXXX.pdb`: 200 candidate loop fragments (length 6-12). Each fragment contains a short peptide with N, CA, C, O, CB atoms. You may use these as templates.

## Task
For each complex, reconstruct the missing CDR-H3 loop and output a complete PDB file. The loop must fill the gap in chain H, connect the anchor residues with proper peptide bonds, have realistic backbone geometry, avoid steric clashes, and make favorable contacts with the antigen (chain A). You may use the fragment library, but you may also generate loops from scratch.

## Constraints
1. Output files must be at `/app/output/complex_XX_reconstructed.pdb` for each input complex.
2. All atoms outside the missing loop must be copied exactly from the input (same coordinates, atom names, chains, and residue numbers).
3. The loop must contain the correct number of residues, numbered consecutively to fill the gap in chain H.
4. The loop sequence must be one of the sequences found in the fragment library.
5. The N-terminal N of the first loop residue must be within 1.5 Å of the C atom of the preceding framework residue; the C-terminal C of the last loop residue must be within 1.5 Å of the N atom of the following framework residue.
6. Loop backbone bond lengths (N-CA, CA-C, C-N) must be within 0.1 Å of ideal values (1.46, 1.52, 1.33 Å). Consecutive CA-CA distances must be 3.8 ± 0.5 Å.
7. No steric clashes: any non-bonded atom pair in the loop must have a distance > 2.0 Å.
8. Docking score: the number of heavy-atom contacts (distance < 4.5 Å) between the loop and antigen (chain A) should be at least 3 (partial credit for ≥5).

## Output
For each input complex, write a PDB file to `/app/output/complex_XX_reconstructed.pdb`. The file must contain all original chains plus the reconstructed loop in chain H.

## Scoring
The grader evaluates each complex independently. For each complex, you earn points for passing each constraint. The docking constraint gives 0.5 points for ≥3 contacts and 1.0 point for ≥5 contacts. The final score is the total points divided by the maximum possible points across all complexes. Partial credit is awarded per constraint.
