# Structure-Conditioned Binder Sequence Design with TriFlow-lite

You are a computational protein engineer. Your goal is to design a 12-residue binder sequence that optimally interacts with a provided target protein structure, using a simplified mean-field discrete flow-matching procedure inspired by TriFlow. The binder backbone is fixed; you must only change the amino acid identities.

## Inputs

All inputs are in `/app/data/`:
- `target.pdb`: Target protein structure (80 residues, chain A). Only CA atoms are used.
- `binder.pdb`: Binder backbone structure (12 residues, chain B). Initially all residues are ALA.
- `potential.csv`: 20x20 pairwise amino acid interaction matrix. Rows/columns are 3-letter codes in the order: ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL. Values are in arbitrary energy units.
- `ss_propensity.csv`: Secondary structure propensities. Columns: aa, H, E, C. Rows for each amino acid.
- `config.json`: Parameters:
  - `L`: 12 (binder length)
  - `seq_init`: "AAAAAAAAAAAA"
  - `fixed_positions`: {"1": "CYS", "12": "CYS"}
  - `T_start`: 2.0
  - `T_end`: 0.1
  - `n_steps`: 20
  - `cutoff`: 10.0
  - `kernel_decay`: 8.0
  - `ss_penalty_weight`: 0.5
  - `ss_assignment`: ["H", "H", ..., "H"] (12 times)
- `aa_mapping.json`: Mapping from 1-letter to 3-letter amino acid codes.

## Task

Implement the following deterministic mean-field sequence design algorithm. Use only the CA coordinates from the PDB files. Distances are Euclidean in Å.

### Energy definitions

For a binder sequence `X` (list of 12 three-letter codes), define:
- For position `i` (0-indexed) and amino acid `a`:
  - `E_target(i, a) = sum_{j=1..80} V[a][target_j] * exp(-(d_ij / 8.0)^2)` if `d_ij < 10.0`, else 0.
  - `E_ss(i, a) = -0.5 * P[a][SS_i]` where `SS_i` is the secondary structure of binder position `i` from `config['ss_assignment']`.
  - `E_pair(i, a, X) = sum_{k=0..11, k != i} V[a][X_k] * exp(-(d_ik / 8.0)^2)` if `d_ik < 10.0`, else 0.
  - Field: `H_i(a) = E_target(i,a) + E_ss(i,a) + E_pair(i,a,X)`

- Total energy of sequence `X`:
  - `E_total = sum_{i=0..11} [ E_target(i, X_i) + E_ss(i, X_i) ] + sum_{i=0..10} sum_{k=i+1..11} V[X_i][X_k] * exp(-(d_ik / 8.0)^2)` (only if `d_ik < 10.0`).

### Mean-field update

1. Initialize `q_i(a) = 1/20` for all `i` (0..11) and all 20 amino acids `a`.
2. For each fixed position (1 and 12 in 1-indexed, i.e., indices 0 and 11), set `q_i(a) = 1.0` if `a` is the fixed amino acid, else `0.0`.
3. For step `t = 1` to `n_steps`:
   - Set current sequence `X_i = argmax_a q_i(a)`. Tie-break by the order of amino acids in the `potential.csv` header (i.e., the order listed above).
   - Compute `H_i(a)` for all `i`, `a` using the current `X` for the pairwise term.
   - Temperature: `T_t = T_start * (T_end / T_start) ** ((t-1) / (n_steps-1))` (for `n_steps=20`).
   - Compute logits: `logit_i(a) = -H_i(a) / T_t + log(q_i(a) + 1e-12)`.
   - Update `q_i(a) = exp(logit_i(a) - max_a logit_i(a))`, then normalize so that `sum_a q_i(a) = 1`.
   - Re-enforce fixed positions as in step 2.
   - Compute `E_total` of the current sequence `X` using the total energy formula.
   - Record: step number `t`, sequence `X` as a 1-letter string, and `E_total`.

After the loop, the final sequence `X` (from step `n_steps`) is your designed binder sequence.

## Constraints

1. **Output format**: `/app/output/designed.fasta` must contain exactly two lines: a header starting with `>` and a sequence line of length 12 using standard 1-letter amino acid codes. `/app/output/trajectory.csv` must have header `step,sequence,energy` and exactly 20 rows (steps 1..20), with `sequence` as a 1-letter string of length 12 and `energy` as a float.
2. **Fixed positions**: Residues at 1-indexed positions 1 and 12 must be CYS (one-letter `C`).
3. **Sequence identity**: The designed sequence must match the gold reference sequence at ≥ 90% of positions (≥ 11 out of 12).
4. **Energy threshold**: The total energy of your sequence must be ≤ gold total energy + 1.0.
5. **Per-position energy**: The maximum per-position field `H_i(X_i)` must be ≤ gold maximum per-position field + 1.0.
6. **Hydrophobic fraction**: The fraction of hydrophobic residues (A, V, I, L, M, F, W, Y) in the designed sequence must be between 0.25 and 0.75 inclusive.
7. **Trajectory monotonicity**: The `energy` column must be non-increasing (within 0.01) across steps.
8. **Final sequence match**: The sequence in the last row of `trajectory.csv` must match the sequence in `designed.fasta`.
9. **Initial sequence**: The first row of `trajectory.csv` must have sequence `CAAAAAAAAAAC` (1-indexed positions 1 and 12 are C, positions 2-11 are A).

## Output

Write your results to:
- `/app/output/designed.fasta`
- `/app/output/trajectory.csv`

Example FASTA:
```
>designed_binder
CAAAAAAAAAAC
```

Example trajectory.csv:
```
step,sequence,energy
1,CAAAAAAAAAAC,-12.34
2,CAAAAAAAAAAC,-12.35
...
20,CAAAAAAAAAAC,-15.67
```

## Scoring

The grader checks the 9 constraints above. Each constraint passed contributes 1/9 to the final score. The score is the fraction of constraints passed. The grader recomputes the gold sequence using the exact algorithm described above from a protected seed.
