# Spatial ligand–receptor annotations carry conditional heritability signal across human tissues

You are rebuilding **EdgeMap**: a method that turns spatially organised ligand–receptor (LR) co-expression into gene-level annotations, projects those annotations onto SNPs with an LD reference, and tests the resulting SNP annotation **jointly** with a cell-intrinsic annotation against GWAS summary statistics, tissue by tissue.

## Inputs (all under `/app/data`)

- `spatial/<tissue>_counts.tsv` — raw spot × gene UMI counts for five human tissues (`adipose`, `brain`, `heart`, `liver`, `lung`). Header: `spot_id`, `x`, `y`, then 200 gene columns. Each file has 900 spots; `x`,`y` form a complete 30×30 grid.
- `lr_pairs.tsv` — 90 ligand–receptor pairs: `pair_id`, `ligand`, `receptor`, `class`.
- `genes.tsv` — 200 genes: `gene_id`, `chr`, `tss`.
- `intrinsic_annotations.tsv` — `gene_id`, `weight`: the competing cell-intrinsic annotation.
- `snps.tsv` — 800 SNPs: `snp_id`, `chr`, `pos`, `maf`.
- `ld_matrix.tsv` — LD reference, long format: `snp_i`, `snp_j`, `r2` (upper triangle only; symmetric; pairs with r² ≥ 0.005; the diagonal is 1, absent pairs are 0).
- `gwas_sumstats.tsv` — `trait`, `snp_id`, `z`, `n` for 6 traits × 800 SNPs.
- `trait_tissue.tsv` — `trait`, `matched_tissue`: the biologically matched tissue of each trait.

## Task

Work in `/app` and write results to `/app/output`. Implement the pipeline exactly as specified.

**1. Spot normalisation.** For spot s with gene counts c_gs and library size T_s = Σ_g c_gs, use `u_gs = log1p(10^4 · c_gs / T_s)`.

**2. LR context activity.** For pair p = (L,R) in tissue t and spot s, `a_ps = sqrt(u_Ls · u_Rs)`. Quantify its spatial organisation with Moran's I on the rook-adjacency graph of the 30×30 grid (spots s,s' adjacent iff |Δx| + |Δy| = 1):

```
I_pt = (n / W) · Σ_{s,s'} w_ss' (a_ps − ā_p)(a_ps' − ā_p) / Σ_s (a_ps − ā_p)²
```

with n = 900, w_ss' = 1 for adjacent spots (0 otherwise), and W = Σ_{s,s'} w_ss' = 3480.

**3. Gene edge score.** For every gene g and tissue t, `e_gt` = mean of `I_pt` over all pairs p containing g (as ligand or receptor); 0 if g appears in no pair.

**4. SNP-level annotation.** For a gene vector v (edge: `v_g = e_gt`; intrinsic: `v_g = weight_g · mean_s u_gs`) build

```
x_j = Σ_g v_g · exp(−|pos_j − tss_g| / 2·10^5) · 1[chr_g = chr_j  and  |pos_j − tss_g| ≤ 10^6]
```

then divide the whole vector by its mean over the 800 SNPs (if the mean is 0, leave the vector unchanged).

**5. LD scores.** `l_k(j) = Σ_{j'} r²_jj' · x_k(j')` using the supplied LD matrix.

**6. Joint regression.** For every (trait, tissue) combination regress `y_j = z_j²` on `[1, l_edge(j), l_intrinsic(j)]` by ordinary least squares over all 800 SNPs. Use classical OLS standard errors (σ² = RSS/(n − 3)); report the coefficients as `edge_tau` / `intrinsic_tau` and `z = coefficient / se`.

**7. Conditional contrast.** `delta_z = edge_z − intrinsic_z`.

**8. Gene prioritisation.** For trait `LDL` in its matched tissue `liver`, take every distinct gene appearing in at least one LR pair and build a single-gene edge annotation (Step 4 with `v_g = e_g,liver` for that gene only, all other genes 0). Regress LDL `z²` on `[1, l_gene, l_intrinsic]` and report `z_g = b_gene / se_gene`. Mark the ten genes with the largest `z_g` as prioritised (ties broken by ascending gene id).

## Constraints

1. **Files and rows.** `/app/output/edge_z_scores.tsv` must have exactly 30 data rows (6 traits × 5 tissues, one per combination), sorted ascending by `(trait, tissue)`. `/app/output/prioritized_genes.tsv` must have exactly one row per distinct gene occurring in `lr_pairs.tsv`, sorted by descending `z_g` (ties: ascending `gene`).
2. **Columns.** `edge_z_scores.tsv`: `trait`, `tissue`, `edge_tau`, `intrinsic_tau`, `edge_z`, `intrinsic_z`, `delta_z` (tab-separated). `prioritized_genes.tsv`: `gene`, `edge_score`, `z_g`, `prioritized` with `prioritized ∈ {0,1}`.
3. **Numerical agreement.** `edge_z`, `intrinsic_z`, `delta_z` and `z_g` must match the reference implementation to within 0.05; `edge_score` to within 1e-4; `edge_tau`/`intrinsic_tau` must follow the Step-4 normalisation (do not rescale the annotation after dividing by its mean).
4. **Finiteness.** All reported values must be finite (no NaN/Inf) and both files must be plain tab-separated text with the header row exactly as above (no quoting, no extra columns).
5. **Tissue specificity.** For each trait the matched tissue must reproduce the reference ordering: its `edge_z` above the mean `edge_z` of the four unmatched tissues in exactly the traits where the reference shows this.
6. **Prioritisation.** `prioritized = 1` for exactly the ten genes with the largest `z_g` in your own table; correctness is scored as the F1 of that set against the reference top ten.

## Output

`/app/output/edge_z_scores.tsv` (header + 30 rows):

```
trait	tissue	edge_tau	intrinsic_tau	edge_z	intrinsic_z	delta_z
LDL	adipose	0.04310000	0.01210000	0.912345	0.423456	0.488889
```

`/app/output/prioritized_genes.tsv` (header + one row per LR gene, sorted by z_g descending):

```
gene	edge_score	z_g	prioritized
G0031	0.28410000	2.413790	1
```

## Scoring

The grader regenerates every input from the protected seed, recomputes the complete reference pipeline, and compares your two tables. Partial credit (weights in brackets): file/format rules [0.10]; `edge_z` accuracy over the 30 rows [0.30]; conditional quantities (`intrinsic_z`, `delta_z`) [0.15]; matched-vs-unmatched tissue pattern [0.15]; per-gene `z_g` and `edge_score` agreement [0.15]; F1 of the prioritised gene set [0.15]. The score is the weighted sum of the passing fractions and is written to `/logs/verifier/reward.json` as `{"score": <float>, "reward": <float>}`.
