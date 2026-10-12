# KMT2D/MLL4 loss-of-function, tumour immunity and cisplatin response in colorectal cancer

You are a computational biologist in a translational oncology lab. KMT2D (MLL4) is a histone H3 lysine 4 methyltransferase that is frequently inactivated in colorectal cancer (CRC). The working hypothesis is that KMT2D loss-of-function (LoF) mutations produce an immunologically "hot" tumour — increased immune-checkpoint expression, elevated T-effector / IFN-γ signatures, increased CD8+ T-cell / NK / macrophage infiltration with reduced Tregs — and that this translates into better immune-checkpoint-inhibitor (ICI) response and increased cisplatin sensitivity.

You are given a realistic CRC cohort (targeted transcriptome, somatic mutations, clinical follow-up) plus a cell-line cisplatin screen. Reproduce the key analyses end-to-end and report every statistic in a single JSON file.

## Inputs
All files live in `/app/data`:

| file | content | shape |
|---|---|---|
| `expression_matrix.tsv` | log2(TPM+1) targeted RNA panel; first column `gene`, remaining columns are TCGA-style sample IDs | 60 genes x 240 samples |
| `kmt2d_mutations.tsv` | somatic KMT2D variant calls with columns `sample_id`, `gene`, `variant_classification`, `hgvsp` | only mutated samples appear |
| `clinical.tsv` | `sample_id`, `cancer_type`, `ici_treated` (yes/no), `response` (CR/PR/SD/PD/NA), `os_months`, `os_event` | 240 samples |
| `cell_line_drug_screen.tsv` | `cell_line`, `kmt2d_status` (LOF/WT), `cisplatin_auc` (lower = more sensitive) | 60 cell lines |

A sample that does not appear in `kmt2d_mutations.tsv` has no KMT2D variant. `variant_classification` takes values `Nonsense_Mutation`, `Frame_Shift_Del`, `Frame_Shift_Ins`, `Splice_Site`, `Missense_Mutation`.

## Task
1. **Classify samples.** A sample is KMT2D-LoF if and only if it carries a truncating variant (`Nonsense_Mutation`, `Frame_Shift_Del`, `Frame_Shift_Ins` or `Splice_Site`). `Missense_Mutation` carriers count as KMT2D wild-type. Every one of the 240 samples must be assigned.
2. **Immune-checkpoint expression.** For `CD274` (PD-L1), `CTLA4` and `CD8A`, compare log2 expression between KMT2D-LoF and KMT2D wild-type samples with a two-sided Mann–Whitney U test. Report `delta = mean(LoF) - mean(WT)` and the p-value.
3. **Gene signatures.** Score every sample for two published signatures as the mean of per-gene z-scores (z computed across all 240 samples with the population standard deviation, ddof=0):
   - `T_effector`: GZMA, GZMB, PRF1, IFNG, CXCL9, CXCL10
   - `IFN_gamma`: IDO1, STAT1, HLA-DRA, CXCL11, IFNG
   Compare signature scores between LoF and WT (Mann–Whitney U), reporting `delta` and p-value.
4. **Immune deconvolution.** Using the same z-score / mean-marker approach, estimate infiltration for four cell types:
   - `CD8_T_cell`: CD8A, CD8B, GZMK
   - `NK_cell`: NKG7, KLRD1, GNLY
   - `Macrophage`: CD68, CD163, CSF1R
   - `Treg`: FOXP3, IL2RA, IKZF2
   Report `delta` and p-value for each cell type.
5. **ICI response.** Restrict to `ici_treated == "yes"`. A responder has `response` in {CR, PR}. Build the 2x2 table (rows LoF then WT; columns responder then non-responder), and report the Fisher exact odds ratio, its p-value, and the response rate in each group.
6. **Survival.** On the same ICI-treated subset run a log-rank test (LoF vs WT) and a univariable Cox proportional-hazards model with a LoF indicator; report the log-rank p-value and the hazard ratio for LoF.
7. **Cisplatin sensitivity.** From `cell_line_drug_screen.tsv`, compare `cisplatin_auc` between KMT2D-LoF and KMT2D-WT cell lines (Mann–Whitney U); report `delta = mean(LOF) - mean(WT)` and the p-value.

## Constraints
1. Sample classification must follow the truncating-variant rule exactly; missense carriers are wild-type and `n_lof + n_wt` must equal 240.
2. Every reported `delta` is a difference of group means of the already log2-transformed (genes) or z-scored (scores) quantity — never a ratio and never computed on raw TPM.
3. Z-scores use the population standard deviation across all 240 samples (`ddof=0`) and are averaged within a signature / cell-type marker set before testing.
4. All p-values are two-sided; gene, signature and deconvolution p-values come from the Mann–Whitney U test.
5. ICI-response and survival analyses use only `ici_treated == "yes"` samples; the 2x2 table is ordered LoF, WT by responder, non-responder.
6. The sign of every reported effect must match the biological hypothesis: checkpoint / signature / CD8 / NK / macrophage deltas > 0, Treg delta < 0, cisplatin delta < 0, ICI odds ratio > 1, Cox hazard ratio < 1.
7. Exactly one JSON object is written to `/app/results.json`; every leaf value is a JSON number.

## Output
Write `/app/results.json`:

```json
{
  "n_lof": 58,
  "n_wt": 182,
  "immune_markers": {
    "CD274": {"delta": 0.812, "p_value": 3.1e-09},
    "CTLA4": {"delta": 0.665, "p_value": 1.2e-07},
    "CD8A":  {"delta": 0.734, "p_value": 5.6e-08}
  },
  "signatures": {
    "T_effector": {"delta": 0.703, "p_value": 2.2e-09},
    "IFN_gamma":  {"delta": 0.641, "p_value": 8.9e-08}
  },
  "deconvolution": {
    "CD8_T_cell": {"delta": 0.712, "p_value": 1.4e-08},
    "NK_cell":    {"delta": 0.588, "p_value": 2.7e-06},
    "Macrophage": {"delta": 0.503, "p_value": 4.1e-05},
    "Treg":       {"delta": -0.612, "p_value": 3.3e-07}
  },
  "ici_response": {
    "odds_ratio": 3.47,
    "p_value": 0.00031,
    "response_rate_lof": 0.545,
    "response_rate_wt": 0.222
  },
  "survival": {"logrank_p": 0.00042, "hazard_ratio": 0.58},
  "drug_sensitivity": {"delta": -2.31, "p_value": 1.1e-08}
}
```

The numbers above are illustrative only — you must compute the real ones from the data.

## Scoring
`tests/grade.py` recomputes every statistic directly from `/app/data` and scores 10 independently checked constraints: (1) group sizes, (2) immune-marker deltas (tolerance ±0.03), (3) immune-marker direction & significance, (4) signature effects (tolerance ±0.03 + direction + p<0.05), (5) deconvolution effects, (6) deconvolution significance, (7) ICI response, (8) survival, (9) cisplatin sensitivity, (10) JSON schema. Score = fraction of constraints passed; partial credit is awarded per constraint.
