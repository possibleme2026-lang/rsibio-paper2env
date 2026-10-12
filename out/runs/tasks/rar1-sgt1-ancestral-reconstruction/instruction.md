# Ancestral reconstruction of RAR1-SGT1 binding specificity in land plants

You are a computational evolutionary biologist studying the co-chaperone complex RAR1-SGT1, which is important for plant immunity. The RAR1-SGT1 interaction interface has diverged in leptosporangiate ferns, leading to lineage-specific binding. Your goal is to reconstruct the evolutionary history of the binding interface and predict binding specificities.

## Inputs
All files are in `/app/data/`:
- `rar1_alignment.fasta`: MSA of RAR1 orthologs (40 species, length 150 aa).
- `sgt1_alignment.fasta`: MSA of SGT1 orthologs (40 species, length 200 aa).
- `species_tree.nwk`: Phylogenetic tree of the 40 species in Newick format with branch lengths and internal node labels (`Root`, `FernAncestor`, `CoreFernAncestor`).
- `interface_positions.tsv`: Positions of the four key interface residues (RAR1 position 75; SGT1 positions 100, 130, 160). Positions are 1-indexed.
- `binding_assays.tsv`: Experimental yeast two-hybrid binding results for 15 RAR1-SGT1 pairs. Columns: `species_A`, `species_B`, `RAR1_75`, `SGT1_100`, `SGT1_130`, `SGT1_160`, `binding` (1 = interacts, 0 = no interaction). These pairs are a subset of all possible combinations.

## Task
1. **Ancestral state reconstruction**: Use the provided MSAs and the species tree to reconstruct the ancestral states at all internal nodes. You may use IQ-TREE (installed) or implement your own maximum likelihood method (e.g., Felsenstein's pruning algorithm). Extract the interface residues (RAR1_75, SGT1_100, SGT1_130, SGT1_160) for the nodes `Root`, `FernAncestor`, and `CoreFernAncestor`.
2. **Infer the binding rule**: From `binding_assays.tsv`, deduce the rules that determine whether RAR1 and SGT1 interact based on the four interface residues. Output the rule as a list of allowed 4-tuples (combinations of residues that result in binding).
3. **Predict extant binding**: For every species in the alignments, predict whether its own RAR1 and SGT1 proteins interact, using the rule you inferred. Output a dictionary mapping species name to boolean.
4. **Determine mutation order**: In the fern lineage, three mutations occurred at the interface: SGT1_130, SGT1_100, and RAR1_75. Determine the order in which these mutations became fixed. The order should be consistent with the ancestral states and the binding rule (e.g., binding may be maintained through the intermediate steps).
5. **Output**: Write a JSON file to `/app/output/solution.json` with the following structure:
```json
{
  "ancestral_states": {
    "Root": {"RAR1_75": "X", "SGT1_100": "X", "SGT1_130": "X", "SGT1_160": "X"},
    "FernAncestor": {"RAR1_75": "X", "SGT1_100": "X", "SGT1_130": "X", "SGT1_160": "X"},
    "CoreFernAncestor": {"RAR1_75": "X", "SGT1_100": "X", "SGT1_130": "X", "SGT1_160": "X"}
  },
  "binding_rule": [
    {"RAR1_75": "X", "SGT1_100": "X", "SGT1_130": "X", "SGT1_160": "X"},
    ...
  ],
  "extant_binding": {
    "Seed_01": true,
    "Seed_02": false,
    ...
  },
  "mutation_order": ["SGT1_130", "RAR1_75", "SGT1_100"]
}
```
Note: The above values are placeholders. You must determine the correct values from your analysis.

## Constraints
1. The ancestral states for `Root`, `FernAncestor`, and `CoreFernAncestor` must be based on a proper ancestral state reconstruction (e.g., maximum likelihood) using the provided alignments and tree. Do not simply guess.
2. The binding rule must correctly classify all combinations in a held-out test set (the grader will test your rule on unseen combinations).
3. The extant binding predictions must achieve an F1 score of at least 0.9 against the true binding status of all 40 species.
4. The mutation order must list the three mutations in the exact order they occurred in the fern lineage.
5. The output JSON must strictly follow the schema above. All four fields must be present.
6. The analysis must be reproducible: include a file `/app/output/analysis.log` with the commands you ran (e.g., IQ-TREE command line). (Not graded, but good practice.)

## Scoring
The grader will compute a score as the average of four components:
- Fraction of correct ancestral states (12 total positions across 3 nodes).
- F1 score for extant binding predictions.
- 1.0 if mutation order is exactly correct, else 0.0.
- F1 score for the binding rule on a test set of 100+ combinations.
Partial credit is given for each component. The final reward is the average.
