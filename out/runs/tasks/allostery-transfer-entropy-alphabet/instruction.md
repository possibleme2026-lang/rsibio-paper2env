# Allosteric Transfer Entropy Alphabet

You are a computational biologist implementing a transferable allosteric descriptor based on the alphabet of allostery concept. The method computes a per-residue allosteric track (net transfer entropy, source, sink, switch) from protein structures using a Gaussian Network Model (GNM) and a dictionary of local contact words.

## Inputs
- `/app/data/background/`: 40 PDB files (bg_00.pdb ... bg_39.pdb) with C-alpha coordinates. Each 80-160 residues.
- `/app/data/targets/`: 9 target proteins, each with `target_XX_apo.pdb` and `target_XX_holo.pdb` (XX = 00..08). 100-150 residues.
- `/app/data/sequences.fasta`: FASTA sequences for all proteins.
- `/app/data/gen_inputs.py`: deterministic generator used to create the data.

## Task
Implement the following pipeline in Python. Write your results to `/app/results/`.

### 1. Parse structures
For each PDB file, extract the C-alpha coordinates and the amino-acid sequence (one-letter codes) for each residue in order. Residue indices start at 1.

### 2. Contact map and GNM
For a protein of length N, define a contact between residues i and j (1-based) if |i-j| >= 3 and the Euclidean distance between their C-alpha atoms is <= 8.0 Angstrom.
Build the N x N Kirchhoff matrix Gamma:
- Gamma[i,i] = number of contacts of residue i.
- Gamma[i,j] = Gamma[j,i] = -1 if i and j are in contact, else 0.
Compute the pseudo-inverse C = pinv(Gamma) using numpy.linalg.pinv with rcond=1e-10.
For each contact (i,j), compute the partial correlation rho_ij = 1 / sqrt(Gamma[i,i] * Gamma[j,j]).
Define directed transfer entropy:
TE(i->j) = rho_ij * (C[i,i] - C[j,j]) / (C[i,i] + C[j,j] + 1e-8)
Then for each residue i:
- source(i) = sum_{j in contacts(i)} TE(i->j)
- sink(i) = sum_{j in contacts(i)} TE(j->i)
- net_te(i) = source(i) - sink(i)

### 3. Build the alphabet of local contact words
For each residue i in a background structure:
- Let contacts(i) be the set of residues j with |i-j| >= 3 and distance <= 8.0 Angstrom.
- If |contacts(i)| < 3, skip this residue.
- Sort contacts(i) by distance from i (ascending), breaking ties by residue index ascending, and take the top 3 closest.
- Form a set of 4 residues: i and the top 3. Sort these 4 residues by sequence index.
- The word's sequence part is the one-letter amino-acid codes of these 4 residues in that sorted order (e.g., 'AGST').
- The word's contact part is a 6-bit string for all unordered pairs (a,b) among the 4 residues, in the order (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) where indices are positions in the sorted 4-residue list. Bit = '1' if |r_a - r_b| >= 3 and distance(r_a, r_b) <= 8.0 Angstrom, else '0'.
- The full word is seq_4|bits (e.g., 'AGST|101010').
- Record the net_te value of the central residue i for this word.

Pool all occurrences over the 40 background structures. For each unique word, compute the mean and population standard deviation (ddof=0) of its net_te values. Require at least 2 occurrences to compute a standard deviation; otherwise standard deviation = 0.

### 4. Project onto target proteins
For each target apo structure (`target_XX_apo.pdb`):
- Compute contacts, GNM, source, sink, net_te as above.
- For each residue i, determine its word using the same definition.
- If the word exists in the alphabet with count >= 2, set switch(i) = population standard deviation of net_te for that word. Otherwise switch(i) = 0.
- Write a row for every residue to `/app/results/allosteric_tracks.tsv` with columns:
  protein, residue_index, residue_name, net_te, source, sink, switch
  where protein is e.g. target_00, residue_index is 1-based, residue_name is the one-letter code. Format floats with at least 6 decimal places.

### 5. Validation metrics
For each target protein, parse the apo and holo C-alpha coordinates. Superpose holo onto apo using the Kabsch algorithm on all C-alpha atoms (1-to-1 correspondence by residue index). Compute the per-residue RMSD as the Euclidean distance between the superposed holo C-alpha and the apo C-alpha.
Compute the Spearman correlation between:
- source(i) and RMSD(i) across residues within each target protein.
- sink(i) and RMSD(i) across residues within each target protein.
Write `/app/results/validation_metrics.json` with exactly these keys:
- source_rmsd_mean_rho: mean of the 9 source-RMSD Spearman rho values.
- source_rmsd_positive_count: number of target proteins with source-RMSD rho > 0.
- sink_rmsd_mean_rho: mean of the 9 sink-RMSD Spearman rho values.
- sink_rmsd_negative_count: number of target proteins with sink-RMSD rho < 0.
Round to 6 decimal places.

### 6. Switch vocabulary
From the background alphabet, consider only words with count >= 5. For each such word, compute:
- std_net_te: population standard deviation (ddof=0) of its net_te values.
- hydrophobic_fraction: fraction of the 4 amino acids in the word's sequence part that are in the hydrophobic set {A, V, I, L, M, F, W, Y, P}.
Sort by std_net_te descending and take the top 20 words. Write `/app/results/switch_vocabulary.tsv` with columns:
word, std_net_te, count, hydrophobic_fraction.
Format floats with at least 6 decimal places.

## Constraints
1. The output files must exist at the specified paths and have the exact column headers.
2. allosteric_tracks.tsv must contain exactly one row per residue for all 9 target proteins (total residues = sum of lengths).
3. net_te, source, and sink values must match the reference implementation within an absolute tolerance of 1e-6.
4. switch values must match the reference implementation within an absolute tolerance of 1e-6.
5. validation_metrics.json must contain the four required keys with values matching the reference within 0.01 for mean rho and exact for counts.
6. The sink channel must discriminate held-out allosteric sites: the grader will compute the pooled ROC-AUC of your sink scores against a hidden set of allosteric residues (top 20% sink per target). A correct implementation should achieve AUC >= 0.65.
7. switch_vocabulary.tsv must list the top 20 words by std_net_te (count >= 5) with correct std_net_te, count, and hydrophobic_fraction within 1e-6, and the mean hydrophobic_fraction of these top 20 words must be >= 0.4.

## Output
- /app/results/allosteric_tracks.tsv
- /app/results/validation_metrics.json
- /app/results/switch_vocabulary.tsv

## Scoring
The grader checks each constraint independently. Score = fraction of constraints passed. Partial credit is given per constraint.
