# Structural Analysis of NDH-1MS' CO2 Hydration Site

You are analyzing the cryo-EM structure of the cyanobacterial NDH-1MS' complex from *Thermosynechococcus vestitus*. The paper proposes a CupB-centered candidate CO2 hydration site, a proton transfer network, a blocked proton export channel in NdhF4, and a link between NdhV association and plastoquinone stability.

## Inputs
Two PDB files are provided in `/app/data/`:
- `ndh1ms_ndhv_bound.pdb`: NdhV-bound state (chains A=CupB, B=NdhF4, C=NdhD4, D=NdhV).
- `ndh1ms_ndhv_free.pdb`: NdhV-free state (chains A, B, C only).
Both contain waters (HOH), a CO2 ligand (resname CO2) in CupB, and plastoquinone (resname PQ). Coordinates are synthetic but realistic in size.

## Task
Write a program to perform a structural analysis and output the results to `/app/answer.json`. The analysis must include:

1. **Active-site residues**: Identify all CupB (chain A) residues with any atom within 5.0 Å of any CO2 ligand atom. Exclude the CO2 ligand itself. Output residues as `CupB:<resseq>:<resname>`, sorted by resseq.

2. **Proton transfer network**: Build a graph of water oxygen atoms (HOH) and polar side-chain atoms from chains A, B, C. Polar residues are SER, THR, TYR, ASN, GLN, HIS, GLU, ASP, LYS, ARG. Use side-chain atoms: OG, OG1, OH, OD1, ND2, OE1, NE2, ND1, OE2, NZ, NH1, NH2. Add an edge between two nodes if their Euclidean distance ≤ 3.5 Å. Find the shortest path (by number of edges) from any CO2 atom to any atom in chain B (NdhF4). Output the ordered list of nodes on the path, excluding CO2. Use format `HOH:<resseq>` for waters and `<ChainName>:<resseq>:<resname>:<atomname>` for residues (ChainName: CupB, NdhF4, NdhD4). Also output the terminal NdhF4 residue as `<ChainName>:<resseq>:<resname>`.

3. **Blocked channel**: In chain B (NdhF4), find all bulky hydrophobic residues (PHE, TRP, TYR, LEU, ILE, VAL, MET) that have any atom within 8.0 Å of the terminal NdhF4 atom identified in step 2. Rank them by side-chain volume descending, breaking ties by resseq ascending. Use standard volumes: TRP=237.7, PHE=203.4, TYR=203.4, LEU=166.9, ILE=166.9, MET=162.9, VAL=140.0. Output the top 4 as `NdhF4:<resseq>:<resname>`.

4. **Plastoquinone stabilization**: In both PDB files, count the number of protein atoms (ATOM records) within 4.5 Å of any PQ atom. Report `pq_contacts_bound` and `pq_contacts_free`. Then superimpose the backbone atoms (N, CA, C, O) of chains A, B, C from the bound and free structures using the Kabsch algorithm, and compute the RMSD of the PQ atoms between the two states. Report `pq_rmsd` rounded to 2 decimals. Set `ndhv_stabilizes_pq` to `true` if `pq_contacts_bound > pq_contacts_free`, otherwise `false`.

## Constraints
1. Output must be a valid JSON file at `/app/answer.json`.
2. Active-site residue list must exactly match the set of CupB residues within 5.0 Å of CO2, sorted by resseq.
3. Proton wire list must exactly match the shortest path nodes in order.
4. Terminal NdhF4 residue must match the endpoint of the proton wire.
5. Blocked channel list must exactly match the top 4 bulky hydrophobic residues as specified.
6. `pq_contacts_bound` and `pq_contacts_free` must be exact integers.
7. `pq_rmsd` must be within 0.1 of the computed value.
8. `ndhv_stabilizes_pq` must match the comparison of contacts.

## Output
`/app/answer.json` with keys:
- `active_site_residues`: list of strings
- `proton_wire`: list of strings
- `terminal_ndhf4_residue`: string
- `blocked_channel_residues`: list of strings
- `pq_contacts_bound`: int
- `pq_contacts_free`: int
- `pq_rmsd`: float
- `ndhv_stabilizes_pq`: bool

Example:
```json
{
  "active_site_residues": ["CupB:45:GLY", "CupB:78:SER", ...],
  "proton_wire": ["HOH:1", "HOH:2", ..., "NdhF4:410:HIS:NE2"],
  "terminal_ndhf4_residue": "NdhF4:410:HIS",
  "blocked_channel_residues": ["NdhF4:515:TRP", ...],
  "pq_contacts_bound": 12,
  "pq_contacts_free": 6,
  "pq_rmsd": 1.50,
  "ndhv_stabilizes_pq": true
}
```

## Scoring
The grader recomputes each expected value from the PDB files. Each of the 8 constraints is checked independently. Score = (number of passed constraints) / 8. Partial credit is awarded per constraint.
