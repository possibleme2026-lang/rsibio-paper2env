# Benchmarking pretrained ITS classifiers on full-length vs ITS2 amplicons

You are reproducing a key benchmark from a fungal metabarcoding study: pretrained deep-learning ITS classifiers report high accuracy on full-length ITS records, but environmental surveys sequence only the ITS2 subregion. You have a synthetic-but-realistic reference database and query set. Your job is to run the comparison at both loci, including the flanking-graft ablation, and report the metrics.

## Inputs

All files are in `/app/data/`:
- `reference.fasta` — 3,000 full-length fungal ITS records (~840 bp each), 5 sequences per genus.
- `reference_taxonomy.tsv` — `id`, `phylum`, `family`, `genus` for each reference record.
- `queries.fasta` — 1,200 query sequences (~840 bp each): 600 "seen" genera (present in reference) and 600 "novel" genera (absent from reference).
- `queries_meta.tsv` — `query_id`, `phylum`, `family`, `genus`, `stratum` (seen/novel).
- `primers.tsv` — ITS3 and ITS4 primer sequences.
- `models/` — two pretrained classifiers (`mycoai_cnn.joblib`, `mycoai_transformer.joblib`) and `predict_mycoai.py` CLI. Usage: `python /app/data/models/predict_mycoai.py --model cnn --fasta <in.fa> --out <out.tsv>`. Output: `seq_id<TAB>predicted_family<TAB>confidence`.

## Task

1. **Extract ITS2 subregions** from both queries and reference using the provided primers (ITS3/ITS4). The ITS2 is the sequence between the ITS3 primer and the reverse complement of ITS4. Write `/app/output/query_its2.fasta` and `/app/output/reference_its2.fasta`.
2. **Run reference-based methods** at both loci:
   - Best-hit alignment: run `vsearch --usearch_global` with `--id 0.8 --maxaccepts 1` against the corresponding reference database (full-length for full locus, ITS2 database for ITS2 locus). Map the target to its family.
   - SINTAX: run `vsearch --sintax` with `--sintax_cutoff 0.8` against the corresponding reference database. Parse the family from the taxonomy string.
3. **Fit a HiTaC-style k-mer logistic regression** on 4-mer counts of the reference sequences (genus labels, C=1.0, max_iter=2000, random_state=42). Fit one model on full-length reference for the full locus, and another on reference ITS2 for the ITS2 locus. Predict query genus, then map to family.
4. **Run the two pretrained classifiers** on the full-length queries and on the query ITS2 sequences.
5. **Flanking-graft ablation**: for each query, select the first reference record in `reference.fasta` whose `phylum` differs from the query's phylum. Construct a grafted sequence by replacing that donor's ITS2 with the query's unaltered ITS2. Run both pretrained classifiers on the grafted sequences.
6. **Compute and report metrics** in `/app/output/metrics.json` (see Output).

## Constraints

1. `/app/output/query_its2.fasta` must contain exactly 1,200 sequences, each 150-350 bp, identical to the primer-bounded ITS2 of the corresponding query.
2. `/app/output/predictions.tsv` must have 12,000 rows (1,200 queries × 5 methods × 2 loci) with columns `query_id`, `method`, `locus`, `predicted_family`, `confidence`. Methods are `besthit`, `sintax`, `hitac`, `cnn`, `transformer`. Loci are `full`, `its2`. No missing values.
3. Best-hit full-length seen-genus family accuracy must be ≥ 0.70.
4. The CNN's ITS2 seen-genus family accuracy must be ≤ 0.40 and at least 0.30 lower than its full-length accuracy.
5. The Transformer's ITS2 seen-genus family accuracy must be ≤ 0.40 and at least 0.30 lower than its full-length accuracy.
6. HiTaC's ITS2 seen-genus family accuracy must be ≥ 0.70.
7. In the graft ablation, for both CNN and Transformer, the donor-family rate must be ≥ 0.20 and exceed the query-family rate by ≥ 0.10.
8. The CNN's confidence AUROC for separating novel from seen genera on ITS2 must be ≤ 0.65.
9. Best-hit alignment identity AUROC for separating novel from seen genera on ITS2 must be ≥ 0.75.
10. `metrics.json` must be internally consistent with `predictions.tsv` and `graft_predictions.tsv` (values match recomputation within 1e-6).

## Output

- `/app/output/metrics.json` with exactly these keys (all floats in [0,1]):
  - `besthit_full_seen_family_acc`, `besthit_its2_seen_family_acc`
  - `sintax_full_seen_family_acc`, `sintax_its2_seen_family_acc`
  - `hitac_full_seen_family_acc`, `hitac_its2_seen_family_acc`
  - `cnn_full_seen_family_acc`, `cnn_its2_seen_family_acc`
  - `transformer_full_seen_family_acc`, `transformer_its2_seen_family_acc`
  - `cnn_full_minus_its2_seen_family_acc`, `transformer_full_minus_its2_seen_family_acc`
  - `cnn_graft_donor_family_rate`, `cnn_graft_query_family_rate`
  - `transformer_graft_donor_family_rate`, `transformer_graft_query_family_rate`
  - `cnn_its2_novel_vs_seen_auroc`, `transformer_its2_novel_vs_seen_auroc`, `besthit_its2_novel_vs_seen_auroc`
- `/app/output/predictions.tsv` (tab-separated) with columns `query_id`, `method`, `locus`, `predicted_family`, `confidence`.
- `/app/output/graft_predictions.tsv` with columns `query_id`, `model`, `predicted_family`, `donor_family`, `query_family`, `confidence`.
- `/app/output/query_its2.fasta`, `/app/output/reference_its2.fasta`.

## Scoring

The grader checks each of the 10 constraints independently. Score = fraction of constraints passed. Partial credit is awarded per constraint. The grader recomputes all metrics from your prediction tables and verifies consistency with `metrics.json`.
