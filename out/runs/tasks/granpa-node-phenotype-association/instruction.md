# GRANPA-style node-phenotype association on a pangenome variation graph

You are reconstructing the association step of GRANPA (Graph Node-Phenotype Association) on a pangenome variation graph (PVG) built from 24 *Oryza sativa* accessions scored for a qualitative (case/control) trait, in the spirit of the Sub1A submergence-tolerance locus analysis. In a PVG, every allele - from a SNP to a large structural insertion - is a node, and an individual "carries" a node when its genome path traverses that node. The goal is to score each node against the phenotype, correct for multiple testing, and collapse the significant nodes into phenotype-associated regions of the graph.

## Inputs

* `/app/data/graph.gfa` - a GFA 1.0 pangenome variation graph (~180-200 segments, 24 sample paths).
  * `S` lines: `S<TAB>segment_name<TAB>sequence`. Segment sequences are 80 bp - 2.5 kb of DNA.
  * `L` lines: `L<TAB>from<TAB>from_orient<TAB>to<TAB>to_orient<TAB>overlap` - an undirected adjacency between two segments.
  * `P` lines: `P<TAB>sample_id<TAB>seg1[+-],seg2[+-],...<TAB>*` - the path of one individual. The trailing `+`/`-` is the traversal orientation and must be ignored when testing presence. A segment repeated on a path still counts once for that individual.
* `/app/data/phenotypes.tsv` - TSV with header `sample_id<TAB>phenotype`, 24 rows, phenotype value is `case` or `control`.

## Task

1. Parse the GFA and build, for every segment, the set of individuals whose path traverses it (orientation-insensitive, de-duplicated per individual).
2. Join those carrier sets with the phenotype table.
3. For every segment build the 2x2 contingency table (a = #cases carrying, b = #cases not carrying, c = #controls carrying, d = #controls not carrying), compute the Phenotype Score `PS = a / n_cases - c / n_controls`, and the one-sided association p-value `p = P(X >= a)` with `X ~ Hypergeometric(N = n_cases + n_controls, K = a + c, n = n_cases)`. This is exactly the upper-tailed Fisher exact test on the 2x2 table; `p = 1` for degenerate tables (segment in every individual or in none).
4. Correct for multiple testing with Benjamini-Hochberg over **all** segments: with `m` segments and p-values sorted ascending `p_(1) <= ... <= p_(m)`, set `q_(i) = min over j >= i of ( m * p_(j) / j )`, capped at 1.0. A segment is significant when `q <= 0.05`.
5. Group the significant segments into regions: build the undirected graph induced on the significant segments using the `L` adjacencies (orientation ignored) and take the connected components. A significant segment with no significant neighbour is a region of size 1.
6. Write the three result files described below.

## Constraints

1. `/app/results/node_association.tsv` must have exactly the header `node_id<TAB>n_case_carriers<TAB>n_control_carriers<TAB>PS<TAB>p_value<TAB>q_value<TAB>significant` and exactly one row per `S` line of the GFA - no missing, no duplicated and no extra segments.
2. `n_case_carriers` and `n_control_carriers` are per-individual carrier counts, not path-occurrence counts.
3. `PS` is written with exactly 6 decimal places (fixed point) and must equal `a / n_cases - c / n_controls` to within 2e-6.
4. `p_value` is the exact one-sided upper-tail hypergeometric p-value and is written in `%.6e` scientific notation; relative error must be below 1e-6.
5. `q_value` is the Benjamini-Hochberg value **with** the monotone step-up correction (`min over j >= i`), written in `%.6e`.
6. `significant` is `True` exactly when `q_value <= 0.05` and `False` otherwise.
7. Node rows must be sorted by `node_id` in ascending ASCII order.
8. `/app/results/regions.tsv` must have exactly the header `region_id<TAB>n_nodes<TAB>node_ids<TAB>region_score<TAB>min_q_value`. `node_ids` is the sorted (ascending ASCII) member segment list joined with `;`. `region_score` is the arithmetic mean of the members' `PS` (6 decimals) and `min_q_value` is the smallest member `q_value` (%.6e). Regions are numbered `R1, R2, ...` and sorted by `min_q_value` ascending, ties broken by the smallest member `node_id` ascending. A component may have a single member.
9. `/app/results/summary.json` must be a JSON object with keys `n_nodes`, `n_samples`, `n_cases`, `n_controls`, `n_significant_nodes`, `n_regions`, `top_region`, where `top_region` is the sorted member list of the first region in `regions.tsv` (or `null` if there is none).

## Output

Three files under `/app/results/`.

`node_association.tsv` - example rows (illustrative values, not the answer):

```
node_id	n_case_carriers	n_control_carriers	PS	p_value	q_value	significant
s0001	10	14	0.000000	1.000000e+00	1.000000e+00	False
s0002	9	1	0.828571	7.189165e-05	4.411234e-03	True
```

`regions.tsv`:

```
region_id	n_nodes	node_ids	region_score	min_q_value
R1	2	s0002;s0003	0.828571	4.411234e-03
```

`summary.json`:

```json
{"n_nodes": 190, "n_samples": 24, "n_cases": 10, "n_controls": 14,
 "n_significant_nodes": 5, "n_regions": 2, "top_region": ["s0002", "s0003"]}
```

## Scoring

The verifier regenerates the graph and phenotypes deterministically from a protected seed, recomputes the full reference table (carrier counts, PS, exact hypergeometric p-values, monotone BH q-values, significance, link-connected regions) and compares your files against it. Nine independent checks are scored with partial credit: table structure / node coverage (0.10), carrier counts (0.12), phenotype scores (0.12), p-values (0.14), q-values (0.10), significance flags (0.10), region sets plus their scores (0.17), ordering rules (0.05) and summary.json (0.10). The F1 of your significant-node set against the reference set is reported as the headline metric. Score = weighted fraction of checks passed.
