# Triage CKEA-generated anti-PD-1 antibodies by MD contact analysis and developability filters

You are a computational antibody engineer. A conditional kernel-elastic autoencoder (CKEA) has generated a library of 24 humanized anti-PD-1 antibody variants by interpolating between nivolumab and pembrolizumab. Each variant has been simulated for 200 frames in complex with the PD-1 FG-loop peptide. Your job is to analyze the simulations, compute a binding score based on interface contacts, filter the variants using developability rules, and rank the top candidates.

## Inputs
- `/app/data/variants.fasta`: 24 variants. Each variant has two entries: `>P2N_<i>_H` (heavy chain) and `>P2N_<i>_L` (light chain). Sequences are amino-acid strings.
- `/app/data/pd1_epitope.fasta`: the PD-1 epitope peptide sequence (chain `P` in structures).
- `/app/data/hotspot_weights.csv`: columns `residue` (1-based index in PD-1 peptide) and `weight`.
- `/app/data/structures/<variant_id>.pdb`: CA-only PDB structures. Chains: `H` (heavy), `L` (light), `P` (PD-1). Residue numbering is sequential 1..N per chain.
- `/app/data/trajectories/<variant_id>.dcd`: 200-frame DCD trajectories. Atom order matches the PDB. Only CA atoms are present.

## Task
1. For each variant, number the heavy and light chains using **abnumber** with the **IMGT** scheme (`Chain(seq, scheme='imgt')`). Extract the CDR1, CDR2, CDR3 regions for both chains. Use these CDR definitions for all subsequent analysis.
2. Compute the following developability properties for the full antibody sequence (VH + VL concatenated):
   - `net_charge`: net charge at pH 7.4 using `Bio.SeqUtils.ProtParam.ProteinAnalysis.charge_at_pH(7.4)`.
   - `pI`: isoelectric point using `ProteinAnalysis.isoelectric_point()`.
   - `n_glyc_motifs`: count of N-X-S/T sequons (X != P) in any CDR.
   - `deamidation_motifs`: count of NG, NS, NT motifs in any CDR.
   - `met_in_cdr`: count of methionine residues in any CDR.
3. For each variant, analyze the MD trajectory:
   - Load the PDB and DCD with **MDAnalysis**.
   - Identify all CA atoms belonging to CDR residues (from step 1) in chains H and L, and all CA atoms in chain P (PD-1).
   - For each frame, compute the distance between every CDR CA atom and every PD-1 CA atom. A contact is defined as distance < 8.0 Å.
   - Compute the contact frequency matrix `F[i,j]` = fraction of the 200 frames in which CDR atom `i` and PD-1 atom `j` are in contact.
   - Compute the binding score:
     `binding_score = sum_j ( weight_j * max_i F[i,j] )`
     where `weight_j` is the weight for PD-1 residue `j` (1-based) from `hotspot_weights.csv`.
4. Determine `passes_filters` for each variant: `True` if and only if **all** of the following hold:
   - `n_glyc_motifs == 0`
   - `deamidation_motifs == 0`
   - `met_in_cdr == 0`
   - `-2.0 <= net_charge <= 4.0`
   - `7.0 <= pI <= 9.5`
   - `binding_score >= 2.0`
5. Rank all variants by `binding_score` descending. Break ties by `variant_id` ascending (e.g., `P2N_2` before `P2N_10`). Assign `rank` 1..24.
6. Identify the top 3 variants that pass all filters, in ranked order.

## Constraints
1. CDR definitions must be exactly those obtained from `abnumber` with `scheme='imgt'`. Any deviation (e.g., using a different numbering scheme) will fail the CDR-dependent checks.
2. The binding score must be computed with the stated formula using all CDR residues and all PD-1 residues, with the 8.0 Å cutoff. No other cutoff or weighting is allowed.
3. Developability motif counts must be computed only within CDR regions (not the full sequence).
4. `net_charge` and `pI` must be computed on the concatenated VH+VL sequence using Biopython's `ProtParam`.
5. `passes_filters` must reflect the exact thresholds given above.
6. Ranking must be strictly by `binding_score` descending, with ties broken by `variant_id` ascending.
7. Output files must be written to `/app/submission/analysis.csv` and `/app/submission/top3.csv` with the exact column names and order specified below.

## Output
Write two CSV files:

**`/app/submission/analysis.csv`** — one row per variant (24 rows), columns exactly:
`variant_id,binding_score,net_charge,pI,n_glyc_motifs,deamidation_motifs,met_in_cdr,passes_filters,rank`
- `binding_score`: float rounded to 4 decimal places.
- `net_charge`: float rounded to 2 decimal places.
- `pI`: float rounded to 2 decimal places.
- `n_glyc_motifs`, `deamidation_motifs`, `met_in_cdr`: integers.
- `passes_filters`: boolean (`True`/`False`).
- `rank`: integer 1..24.

Example row:
`P2N_1,3.4567,1.23,8.45,0,0,0,True,1`

**`/app/submission/top3.csv`** — top 3 passing variants, columns exactly:
`rank,variant_id,binding_score,net_charge,pI,cdr_h3_sequence`
- `rank`: 1, 2, 3.
- `cdr_h3_sequence`: the heavy-chain CDR3 amino-acid sequence as identified by abnumber IMGT.

Example row:
`1,P2N_1,3.4567,1.23,8.45,CATNDDYW`

## Scoring
The grader recomputes all values from the raw data and compares to your submission. Partial credit is awarded per constraint:
- 5%: both output files exist and contain the correct variant IDs.
- 20%: `binding_score` correct (within 0.05) for each variant (fraction of variants correct).
- 10%: `net_charge` correct (within 0.1).
- 10%: `pI` correct (within 0.1).
- 15%: all three motif counts correct for each variant.
- 15%: `passes_filters` correct for each variant.
- 10%: `rank` correct for each variant.
- 15%: `top3.csv` exactly matches the expected top 3 (IDs and order).

The final score is the weighted sum. Write it to `/logs/verifier/reward.json` as `{"score": <value>, "reward": <value>}`.
