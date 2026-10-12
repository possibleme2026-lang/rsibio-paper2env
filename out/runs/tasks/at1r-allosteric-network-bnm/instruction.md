# Reconstruct the AT1R allosteric coupling network from an MD ensemble

## Background
Allosteric communication in class A GPCRs can be read out of molecular-dynamics (MD)
ensembles as a sparse network of conditional residue-residue dependencies: two residues
are coupled when their local interaction-energy fluctuations stay correlated after
conditioning on every other residue. That conditional-dependence graph is the
Gaussian / Bayesian-network analogue of the residue-level Bayesian network model used to
map AT1R allostery. You are given one such ensemble for the angiotensin II type 1
receptor (AT1R), a whole-receptor mutational scan, and a structural annotation of the
receptor. You must recover the coupling network, its allosteric communities, its
functional hotspots, and the communication pathway that links the AngII binding site to
the G-protein interface.

## Inputs (all in /app/data)

1. `at1r_energies.npy` - float32 array of shape (1500, 359): local residue interaction
   energy in kcal/mol. Row t is MD frame t (frames are temporally correlated samples of
   one 1.5 microsecond trajectory); column j holds residue j. Residues are numbered
   1..359 and column j corresponds to `residue_id = R{j+1:03d}` (column 0 -> R001).
2. `at1r_mutscan.tsv` - 359 data rows, tab separated, header
   `index / residue_id / wt_aa / mean_ddg / n_substitutions / gq_activity_score`.
   `mean_ddg` is the mean destabilisation (kcal/mol) over all substitutions introduced
   at that position; higher = more sensitive.
3. `at1r_annotation.tsv` - 359 data rows, tab separated, header
   `index / residue_id / residue_number / region / tm_helix / generic_number / site / is_pocket`.
   `site` is one of `angII_site`, `gprotein_interface`, `other`; `is_pocket` is 1 for
   residues lining the cryptic intracellular pocket.

## Task
Implement the following deterministic analysis pipeline and write its results to
`/app/output/`, then package the same pipeline as a reusable CLI script
`/app/analyze.py` (see the last section - the grader re-runs it on a held-out ensemble).

**Step 1 - coupling matrix.** Standardise every column of the energy matrix to zero mean
and unit variance (population standard deviation, ddof=0). Estimate the shrunken
covariance of the standardised ensemble with the Ledoit-Wolf estimator
`sklearn.covariance.ledoit_wolf(X, assume_centered=True)`. Invert it, `Theta = inv(shrunk)`,
and convert to partial correlations
`P[i,j] = -Theta[i,j] / sqrt(Theta[i,i]*Theta[j,j])`, then set `P[i,i] = 1`.
Because the columns are standardised this is the partial-correlation matrix.

**Step 2 - allosteric network.** Keep the unordered pair (i, j) with i < j as an edge
when `abs(P[i,j]) >= 0.20` and `j - i >= 4` (the separation filter removes intra-helical
neighbours). `weighted_degree[i]` is the sum of `abs(P[i,j])` over all kept edges incident
to i.

**Step 3 - hotspots.** `hotspot_score[i] = z(weighted_degree)[i] + z(mean_ddg)[i]`, where
z() is the z-score over all 359 residues (population standard deviation, ddof=0) and
`mean_ddg` comes from at1r_mutscan.tsv. Rank residues by descending hotspot_score, ties
broken by ascending residue index. The best 25 residues are the hotspots.

**Step 4 - communities.** Run greedy modularity maximisation
(`networkx.algorithms.community.greedy_modularity_communities`, resolution 1, edge weight
= `abs(P[i,j])`) on the subgraph induced by residues that have at least one edge. Residues
with no edge are assigned community -1. Communities are numbered 0, 1, 2, ... in the order
returned by the function.

**Step 5 - pathways.** Build the same graph with edge cost `c(i,j) = -log(abs(P[i,j]))`.
For every `angII_site` residue s and every `gprotein_interface` residue t compute the
minimum-cost path with Dijkstra. Sort all (s, t) pairs by (total cost, s, t) ascending and
keep the first three pairs whose source residues are pairwise distinct.

**Step 6 - pocket enrichment.** With N = 359 residues, K = number of `is_pocket == 1`
residues, n = 25 and k = number of pocket residues among the 25 hotspots, report the
upper-tail hypergeometric p-value `scipy.stats.hypergeom.sf(k-1, N, K, n)`.

## Outputs

All files go to `/app/output/`. Indices are 0-based column indices of the energy matrix.

1. `coupling_matrix.npy` - the 359x359 partial-correlation matrix (float32 or float64).
2. `network_edges.tsv` - header `residue_i / residue_j / partial_corr / abs_partial_corr`,
   one row per kept network edge, always i < j, no duplicate pairs.
3. `residue_scores.tsv` - header
   `index / residue_id / weighted_degree / mut_ddg / hotspot_score / community / hotspot_rank`,
   exactly 359 rows in ascending index order.
4. `hotspots.tsv` - header `rank / index / residue_id / hotspot_score / community`,
   25 rows, rank 1..25 in descending hotspot_score.
5. `pathways.tsv` - header
   `pathway_id / start_residue / end_residue / n_hops / total_cost / path_residues`,
   three rows; `path_residues` is the semicolon-joined residue_id chain from start to end.
6. `summary.json` - JSON object with the keys `n_frames`, `n_residues`, `threshold`,
   `min_separation`, `n_edges`, `n_communities`, `n_pocket_in_top25`,
   `pocket_enrichment_p`, `top_hotspot_residue_id`.

Worked example (fields separated by TAB):

`network_edges.tsv`
`residue_i<TAB>residue_j<TAB>partial_corr<TAB>abs_partial_corr`
`41<TAB>129<TAB>-0.3712518<TAB>0.3712518`

`residue_scores.tsv`
`index<TAB>residue_id<TAB>weighted_degree<TAB>mut_ddg<TAB>hotspot_score<TAB>community<TAB>hotspot_rank`
`0<TAB>R001<TAB>0.8123456<TAB>1.4021<TAB>-0.7743<TAB>3<TAB>214`

`hotspots.tsv`
`rank<TAB>index<TAB>residue_id<TAB>hotspot_score<TAB>community`
`1<TAB>129<TAB>R130<TAB>3.914200<TAB>0`

`pathways.tsv`
`pathway_id<TAB>start_residue<TAB>end_residue<TAB>n_hops<TAB>total_cost<TAB>path_residues`
`1<TAB>R034<TAB>R312<TAB>7<TAB>5.724193<TAB>R034;R062;R130;R167;R222;R290;R306;R312`

`summary.json`
`{"n_frames": 1500, "n_residues": 359, "threshold": 0.2, "min_separation": 4, "n_edges": 388, "n_communities": 9, "n_pocket_in_top25": 6, "pocket_enrichment_p": 0.00312841, "top_hotspot_residue_id": "R130"}`

## Deliverable script

Write `/app/analyze.py` implementing exactly the pipeline above and accepting:

`python /app/analyze.py --energies <energies.npy> --mutations <mutscan.tsv> --annotation <annotation.tsv> --outdir <dir>`

plus the optional flags `--threshold` (default 0.20), `--min-separation` (default 4),
`--top-hotspots` (default 25) and `--n-paths` (default 3). It must write the six output
files listed above into `--outdir` using only the three input paths it is given (do not
read anything else from /app/data).

## Constraints

1. **C1 coupling matrix** - `coupling_matrix.npy` is 359x359, symmetric, unit diagonal,
   and every entry agrees with the reference partial-correlation matrix to within 5e-3.
2. **C2 network edges** - `network_edges.tsv` uses the exact four-column header, all rows
   satisfy i < j, `abs(partial_corr) >= 0.20`, `j - i >= 4`, contain no duplicate pairs,
   and the edge set has Jaccard >= 0.85 with the reference network.
3. **C3 residue scores** - 359 rows in ascending index order; `weighted_degree`,
   `hotspot_score` (z-score sum) and `hotspot_rank` (a permutation of 1..359 consistent
   with descending score, ties by ascending index) all match the reference to 1e-3.
4. **C4 communities** - a valid partition of all residues: isolated residues get -1, all
   other residues get community ids 0..k-1, 4 <= k <= 40, and the modularity of the
   partition on the reference network is at least 0.6 x the reference modularity.
5. **C5 hotspots** - `hotspots.tsv` has exactly 25 rows with ranks 1..25 in descending
   score order, at least 22 of the 25 residue indices occur in the reference top-25, and
   the reported scores agree with the reference to 1e-3.
6. **C6 pathways** - three internally valid paths: the first residue is an `angII_site`
   residue, the last is a `gprotein_interface` residue, every consecutive pair is an edge
   of your own network, `total_cost` equals the sum of `-log(abs(partial_corr))` along the
   path, `n_hops == len(path) - 1`, and the reported paths match the reference top-3.
7. **C7 summary / pocket enrichment** - `summary.json` contains every required key;
   `n_edges`, `n_communities` and `n_pocket_in_top25` match the reference integers and
   `pocket_enrichment_p` matches the reference hypergeometric upper-tail p-value.

## Scoring

The grader first recomputes the whole pipeline itself and scores C1-C7 on the six files
in `/app/output/`. It then regenerates a **held-out ensemble** (same format, different
random seed) and re-runs `/app/analyze.py` on it, scoring the same C1-C7 on those outputs.
The final score is `0.35 * public + 0.65 * held-out`, i.e. 65% of the reward comes from the
re-run, so the script must genuinely implement the pipeline rather than reproduce the
public answer. Each constraint contributes partial credit (numeric deviations, edge-set
Jaccard, hotspot overlap and pathway agreement are graded continuously), and the score is
the fraction of constraints passed, weighted as above. `tests/test.sh` writes
`{"score": <value>, "reward": <value>}` to `/logs/verifier/reward.json`.
