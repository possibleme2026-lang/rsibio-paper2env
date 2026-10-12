# MGMG-Lite: Morphology-Guided Molecule Ranking

You are a computational chemist implementing a simplified version of MGMG. You are given:
- `data/reference_compounds.csv`: 500 compounds with known morphological profiles (256 features) and textual descriptions.
- `data/candidates.csv`: 20000 candidate molecules with SMILES and textual descriptions (text embeddings provided as 32 columns).
- `data/query_profile.csv`: a 256-dim morphological profile of a disease phenotype.
- `data/query_text_embedding.csv`: a 32-dim text embedding of the desired bioactivity description.

Your task: Train a multi-output Ridge regression to predict morphological profiles from molecular features, then rank candidates by combined morphological and textual similarity, and select the top 20 drug-like, diverse molecules.

## Algorithm (must be followed exactly)

1. **Feature construction for each molecule:**
   - Compute Morgan fingerprint (radius 2, 2048 bits) from SMILES using RDKit (`AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)`).
   - Compute 10 physicochemical descriptors in this order: MW (`Descriptors.MolWt`), logP (`Crippen.MolLogP`), TPSA (`Descriptors.TPSA`), HBD (`Descriptors.NumHDonors`), HBA (`Descriptors.NumHAcceptors`), rotatable bonds (`Descriptors.NumRotatableBonds`), aromatic rings (`Descriptors.NumAromaticRings`), QED (`QED.qed`), SA score (`sascorer.calculateScore`), fraction Csp3 (`Descriptors.FractionCSP3`).
   - Use the provided `text_emb_00` ... `text_emb_31` columns for text embedding.
   - Concatenate in this order: [Morgan fingerprint (2048), descriptors (10), text embedding (32)] -> 2090 features.

2. **Train Ridge regression:**
   - Use `sklearn.linear_model.Ridge(alpha=1.0)`.
   - Fit on reference compounds: X = features, y = 256 morphological features (`morph_000` ... `morph_255`).
   - Use all default settings.

3. **Predict morphological profiles for candidates:**
   - Use the trained model to predict 256-dim profiles for each candidate.

4. **Score candidates:**
   - Morphology score = cosine similarity between predicted profile and `query_profile.csv` (flattened vector).
   - Text score = cosine similarity between candidate text embedding and `query_text_embedding.csv` (flattened vector).
   - Combined score = 0.8 * morphology_score + 0.2 * text_score.

5. **Filter candidates (in order):**
   - Valid SMILES (RDKit parse).
   - Molecular weight 250-500 Da.
   - logP between -0.5 and 4.5.
   - TPSA <= 120.
   - HBD <= 5, HBA <= 10.
   - QED >= 0.5.
   - SA score <= 4.0.
   - No PAINS alerts (RDKit `FilterCatalog` with `FilterCatalogs.PAINS`).
   - No reactive groups: exclude if SMILES contains any of: `[N+](=O)[O-]`, `C(=O)Cl`, `S(=O)(=O)Cl`, `C(=O)Br`.
   - Diversity: iterate through candidates sorted by combined score descending; select a candidate if its Tanimoto similarity (Morgan r=2, 2048 bits) to all already selected molecules is <= 0.4.

6. **Select top 20** after filtering.

7. **Write output CSV** to `/app/output/selected_molecules.csv` with columns:
   `rank, candidate_id, smiles, combined_score, morphology_score, text_score, MW, logP, QED, SA_score`
   - `rank` starts at 1 for the highest combined score.
   - `smiles` should be the canonical SMILES (`Chem.MolToSmiles(mol)`).
   - Scores rounded to 4 decimal places.

## Example output row
```
rank,candidate_id,smiles,combined_score,morphology_score,text_score,MW,logP,QED,SA_score
1,CAND01234,CC(=O)Nc1ccc(cc1)S(=O)(=O)N,0.9123,0.9345,0.8234,312.4,1.23,0.78,2.1
```

## Scoring
The grader recomputes the expected top-20 using the exact same algorithm. It checks:
1. Output file exists with correct columns and exactly 20 rows.
2. All SMILES are valid and unique.
3. All selected molecules pass the drug-likeness filters.
4. No PAINS alerts.
5. No reactive groups.
6. Diversity: max pairwise Tanimoto <= 0.4.
7. Reported scores match recomputed scores within 1e-3.
8. At least 18 of the 20 selected candidate IDs match the gold top-20.
9. The top-1 candidate ID matches the gold top-1.
10. The exact set of 20 candidate IDs matches the gold top-20.

Your score is the fraction of these 10 constraints that pass.
