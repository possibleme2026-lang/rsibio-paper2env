# Metaproteomics quantification: AUC vs SpC and MBR artifact detection

You are analyzing data-dependent acquisition (DDA) metaproteomics data from a defined mock microbial community. The dataset contains six runs: three replicates of sample A and three of sample B. Some peptide-spectrum matches (PSMs) were identified by matching between runs (MBR). Your goal is to implement the quantification and benchmarking pipeline from the paper: compare MBR-free area-under-the-curve (AUC) quantification with spectral counting (SpC) quantification, and detect falsely identified proteins introduced by MBR.

## Inputs
All inputs are in `/app/data/`:
- `psm_table.tsv`: columns `run`, `spectrum_id`, `peptide_sequence`, `charge`, `protein_id`, `score`, `is_mbr`, `mbr_target_spectrum`. `is_mbr=1` indicates an MBR-transferred PSM.
- `feature_table.tsv`: columns `run`, `feature_id`, `peptide_sequence`, `charge`, `mz`, `rt`, `auc`. MS1 precursor features with integrated AUC.
- `peptide_protein_map.tsv`: columns `peptide_sequence`, `protein_id`, `mapping_type`. `mapping_type` is `unique` or `shared`.
- `sample_metadata.tsv`: columns `run`, `sample`, `replicate`, `condition`. Six rows.

## Task
1. Parse all tables. For protein-level quantification, use only peptides with `mapping_type == 'unique'`. Exclude shared peptides from quantification.
2. Compute SpC quantification: for each run and protein, count non-MBR PSMs (`is_mbr == 0`) whose peptide is unique. Write `/app/output/protein_spc.tsv` with columns `run`, `protein_id`, `spc`. Sort by `run` then `protein_id`.
3. Compute MBR-free AUC quantification: for each run and protein, sum the AUC of all MS1 features that match a non-MBR PSM by (`run`, `peptide_sequence`, `charge`). If multiple PSMs match the same feature, count the feature only once. Write `/app/output/protein_auc_nombr.tsv` with columns `run`, `protein_id`, `auc`. Sort by `run` then `protein_id`.
4. Compute MBR-enabled AUC quantification: same as step 3 but include MBR PSMs (`is_mbr == 1`). Write `/app/output/protein_auc_mbr.tsv` with columns `run`, `protein_id`, `auc`. Sort by `run` then `protein_id`.
5. Detect MBR false-positive proteins: for each run, a protein is an MBR false positive if it has at least one MBR PSM (`is_mbr == 1`) and zero non-MBR PSMs (`is_mbr == 0`) in that run. Consider all PSMs, including shared peptides. Write `/app/output/mbr_false_positive_proteins.tsv` with columns `run`, `protein_id`, `mbr_psm_count`. Sort by `run` then `protein_id`.
6. Compute `/app/output/report.json` with exactly these numeric fields:
   - `dynamic_range_auc`: log10(max AUC / min positive AUC) across all run-protein pairs in the MBR-free AUC table.
   - `dynamic_range_spc`: log10(max SpC / min positive SpC) across all run-protein pairs in the SpC table.
   - `quantified_proteins_auc`: number of rows in the MBR-free AUC table.
   - `quantified_proteins_spc`: number of rows in the SpC table.
   - `mbr_false_positive_proteins`: number of rows in the false-positive table.
   - `mbr_false_positive_psms`: sum of `mbr_psm_count` in the false-positive table.
   All values must be JSON numbers (int or float).

## Constraints
1. Output files must exist at the exact paths and have the exact column names and order specified above. TSV files must be tab-separated with no index column.
2. Protein-level quantification must use only unique peptides; shared peptides must be excluded from SpC, AUC_nombr, and AUC_mbr tables.
3. AUC values must be summed only for features that match a PSM by `run`, `peptide_sequence`, and `charge`. Duplicate feature matches must be counted once per `run`, `protein_id`, `feature_id`.
4. MBR false positives are defined as run-protein pairs with >0 MBR PSMs and 0 non-MBR PSMs, using all peptides.
5. `report.json` must contain exactly the six specified fields with numeric values; dynamic ranges must be computed on positive values only and must be 0.0 if fewer than two positive values exist.
6. The MBR-free AUC quantification must be biologically accurate: the grader will compare it to the hidden known composition of the mock community. Pearson correlation (log10 AUC vs true log2 abundance) must be >= 0.8 for both samples A and B. SpC correlation must be >= 0.4.
7. All tables must be sorted as specified and contain no duplicate (`run`, `protein_id`) pairs.

## Output
Write all outputs to `/app/output/`. Example rows:
`protein_spc.tsv`:
```
run	protein_id	spc
A_rep1	PROT_0001	3
```
`protein_auc_nombr.tsv`:
```
run	protein_id	auc
A_rep1	PROT_0001	123456.78
```
`mbr_false_positive_proteins.tsv`:
```
run	protein_id	mbr_psm_count
A_rep1	PROT_0123	1
```
`report.json` example:
```json
{"dynamic_range_auc": 4.2, "dynamic_range_spc": 1.8, "quantified_proteins_auc": 1234, "quantified_proteins_spc": 1100, "mbr_false_positive_proteins": 30, "mbr_false_positive_psms": 30}
```

## Scoring
The grader checks seven constraints independently: (1) file format and sorting, (2) SpC table match (row-level F1), (3) MBR-free AUC table match (row-level F1), (4) MBR-enabled AUC table match, (5) MBR false-positive table match, (6) report field accuracy, (7) biological accuracy against hidden truth. Partial credit is given as the fraction of rows/fields matching. The final score is the unweighted mean of the seven constraint scores.
