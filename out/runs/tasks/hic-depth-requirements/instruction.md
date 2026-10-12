# Sequencing-depth requirements for Hi-C-derived structural metrics

Hi-C contact maps quantify chromatin architecture over several genomic scales, but different structural features need very different amounts of sequencing before they can be measured reproducibly. You are given eight subsampling levels of four simulated biological replicates of a Hi-C library. Your job is to establish, for five structural metrics, the sequencing depth at which subsampling scatter first becomes no larger than the scatter between independent biological replicates - the accuracy that additional sequencing cannot improve.

## Inputs

All paths below are relative to `/app`.

- `data/matrices/res100kb/rep<K>/depth_<D>M_sub<J>.npz` - 96 files (K = 1..4, D = 1, 2, 5, 10, 20, 30, 60, 100 million read pairs, J = 1, 2, 3). Each file is a scipy sparse CSR matrix written with `scipy.sparse.save_npz`: a symmetric 300 x 300 `int32` count matrix, one row/column per 100 kb bin of a simulated 30 Mb chromosome. `M[i, j]` is the number of simulated read pairs joining bin i and bin j.
- `data/matrices/res25kb/rep<K>/depth_<D>M_sub<J>.npz` - the same layout at 25 kb resolution: symmetric 400 x 400 `int32` matrices covering a simulated 10 Mb region (96 files).
- `data/README.txt` - format notes.

Load a matrix with `scipy.sparse.load_npz(path)` and convert it to a dense float array with `np.asarray(M.todense(), dtype=float)` before analysis.

## Metric definitions

Let `M` be a dense n x n count matrix. The diagonal holds self-ligation products and is excluded from every sum. Natural logarithms unless stated otherwise.

### A. loop_density and loop_size_kb (res100kb, n = 300)

1. `P(k)` = mean of `M[i, i+k]` over i = 0 .. n-1-k, for k = 1 .. n-1.
2. `Ps(k)` = mean of `P` over the window {k-2, ..., k+2} clipped to [1, n-1].
3. `d(k) = (ln Ps(k+1) - ln Ps(k)) / (ln(k+1) - ln(k))` for k = 1 .. n-2.
4. `alpha` = mean of `d(k)` over k = 90 .. 149.
5. `r(k) = alpha - d(k)` for k = 2 .. 40.
6. `loop_density` = max of `r(k)`; `loop_size_kb` = 100 x the smallest k attaining that maximum.

### B. insulation_strength and boundary_count (res25kb, n = 400)

Use the insulation window w = 8 bins.

1. For each centre i in [w-1, n-w] = [7, 392]:
   `IS_raw(i)` = sum of `M[i-a, i+b]` over a = 0..w-1 and b = 1..w (the w x w block immediately up-right of position i).
2. `bg(i)` = median of `IS_raw(j)` over centres j with |j - i| <= 40, clipped to [7, 392].
3. `IS_z(i) = log2( (IS_raw(i) + 1) / (bg(i) + 1) )`.
4. `IS_smooth(i)` = mean of `IS_z` over {i-1, i, i+1}, for i in [8, 391].
5. Candidate boundaries: centres i in [8, 391] with `IS_smooth(i) < IS_smooth(i-1)`, `IS_smooth(i) < IS_smooth(i+1)` and `IS_smooth(i) < -0.25`.
6. Sort candidates by `IS_smooth` ascending; accept a candidate if no already accepted centre lies within 20 bins of it. The accepted centres form the boundary set B.
7. `insulation_strength` = mean over i in B of `-IS_smooth(i)` (0.0 if B is empty); `boundary_count` = number of elements of B.

### C. compartment_strength (res100kb, n = 300)

1. `E(k)` = mean of `M[i, i+k]` over i = 0 .. n-1-k, k = 1 .. n-1.
2. `OE[i, j] = M[i, j] / E(|i-j|)` for i != j; `OE[i, i] = 0` (entries with E = 0 become 0).
3. Pearson correlation matrix C of the rows of OE: subtract each row mean, divide each row by its population standard deviation (ddof = 0), then `C = X X^T / n`. Rows with zero standard deviation are left uncorrelated with every other row and get self-correlation 1.
4. `v` = unit eigenvector of C with the largest eigenvalue (`np.linalg.eigh`; its sign is arbitrary).
5. `compartment_strength = mean(v[v > 0]) - mean(v[v < 0])`.

## Depth analysis

For every metric m, replicate K and depth D, compute the metric on each of the three subsample matrices and take their arithmetic mean:

  `value[m][K][D] = mean over J of metric(m, matrix(res(m), K, D, J))`

with res(m) = res100kb for loop_density, loop_size_kb and compartment_strength, res25kb for insulation_strength and boundary_count. Then:

- subsampling scatter: `scatter[m][D] = sqrt( mean over K of Var(metric on the three subsamples of replicate K at depth D, ddof=1) )` - i.e. the per-replicate sample standard deviation across the three subsamples, squared, averaged over the four replicates, then square-rooted.
- biological variability: `bio[m]` = sample standard deviation (ddof=1) of `value[m][K][100]` over K = 1..4.
- threshold: the smallest depth D in the grid with `scatter[m][D] <= bio[m]`. If no grid depth satisfies it, report 100.

## Output

Write exactly three files.

### 1. `/app/output/depth_requirements.json`

```
{
  "depths_M": [1, 2, 5, 10, 20, 30, 60, 100],
  "metrics": {
    "loop_density": {
      "values_by_replicate": {"1": {"1": 0.0521, "2": 0.0563, ... "100": 0.0712}, "2": {...}, "3": {...}, "4": {...}},
      "subsampling_scatter": {"1": 0.0181, "2": 0.0124, ... "100": 0.0021},
      "biological_variability": 0.0236,
      "threshold_M": 5
    },
    "loop_size_kb": {...},
    "insulation_strength": {...},
    "boundary_count": {...},
    "compartment_strength": {...}
  }
}
```

All JSON object keys are strings; depth keys are the decimal strings "1", "2", ... "100"; replicate keys are "1" .. "4".

### 2. `/app/output/boundaries_rep1_depth100M.tsv`

One integer per line: the accepted boundary centres (0-based 25 kb bin indices) obtained from `data/matrices/res25kb/rep1/depth_100M_sub1.npz` with the rule of section B. Sorted ascending.

### 3. `/app/output/pc1_rep1_depth100M.tsv`

300 lines of `bin_index<TAB>pc1_value` (bin_index = 0..299) obtained from `data/matrices/res100kb/rep1/depth_100M_sub1.npz` with the rule of section C.

## Constraints

1. `depth_requirements.json` must contain all five metric names, the depth grid as decimal strings, one value per (replicate, depth), one scatter value per depth, one biological variability and one threshold per metric, in the schema above.
2. Every metric must follow the definitions exactly (smoothing window, alpha range, search range 2..40, median window +/-40, the -0.25 boundary cut, the 20-bin minimum separation and the eigenvector convention are all checkable). Off-the-shelf Hi-C toolkits use different normalisations and will not reproduce these numbers.
3. Subsampling scatter and biological variability must use the aggregation specified above (RMS over replicates of the ddof=1 subsample standard deviation; ddof=1 standard deviation across the four replicates at 100 M).
4. `threshold_M` must be a value of the depth grid and must be the smallest grid depth at which scatter <= biological variability (100 if the criterion is never met).
5. Threshold ordering: `loop_density <= loop_size_kb <= min(insulation_strength, boundary_count)` and `max(insulation_strength, boundary_count) <= compartment_strength`.
6. Both TSV side files must exist with the layout described above.
7. The analysis must be a pure function of the input matrices: no random subsampling, no hard-coded thresholds.

## Scoring

The grader re-derives every quantity from the raw matrices and awards partial credit (score = sum of weights of the checks that pass):

- 0.05 output schema and both side files parseable
- 0.08 loop_density values of replicate 1 at all eight depths (10% tolerance)
- 0.08 loop_size_kb for all replicates and depths (100 kb tolerance)
- 0.08 insulation_strength values of replicate 2 at all depths (10%)
- 0.10 boundary F1 against the reference boundary set (2-bin matching; needs F1 >= 0.8)
- 0.10 |Pearson r| between the submitted PC1 vector and the reference (needs >= 0.9)
- 0.06 subsampling scatter of all five metrics at all depths (25%)
- 0.10 biological variability of all five metrics (15%)
- 0.25 thresholds, 0.05 per metric (within one grid step of the reference)
- 0.10 ordering of the five thresholds
