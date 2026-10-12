# Low-divergence local ancestry inference: implement the decoder, expose the tract-structure failure, estimate the switch rate

You are benchmarking local ancestry inference (LAI) in the regime that matters for fine-scale admixture mapping: two source populations that are only weakly diverged. Reference panels, admixed haplotypes and the simulated truth are provided for three divergence levels. Implement the two-state ancestry hidden Markov model (HMM) specified below **exactly**, run it on every admixed haplotype, and emit per-site posterior tables plus a machine-readable report that quantifies the per-site-accuracy / tract-structure trade-off and the maximum-likelihood switch rate.

## Inputs

Everything lives under `/app/data` (the dataset is produced by `gen_inputs.py --out /app/data --seed 20260730`; `simulation_manifest.json` records the realised FST of each level).

* `map.tsv` - 3000 SNPs of one 150 cM region, tab separated, header `snp_id  pos_bp  cM`, rows in genomic order. Consecutive-SNP genetic distance is `d_l = (cM(l+1) - cM(l)) / 100` Morgan.
* `<level>/ref_A.haplotypes.txt` and `<level>/ref_B.haplotypes.txt` - 60 reference haplotypes per source population, one haplotype per line, 3000 characters of `0`/`1` (`1` is the counted allele).
* `<level>/admixed_<sample>.txt` - one admixed haplotype, one line of 3000 `0`/`1` characters.
* `<level>/truth_<sample>.txt` - simulated ancestry of that haplotype, one line of 3000 `A`/`B` characters.

`<level>` is one of `fst_0002` (target FST 0.002), `fst_0020` (0.020), `fst_0150` (0.150); `<sample>` is one of `hap1`, `hap2`, `hap3`, `hap4`. The same four ancestry paths are re-used at all three levels, so only reference divergence changes.

## Model (implement exactly as written)

States `A` and `B`. For SNP `l` and source population `s`, let `k(s,l)` be the number of `1` alleles among the 60 panel haplotypes of `s` at `l`, and use the smoothed allele frequency

    p(s,l) = (k(s,l) + 0.5) / 61

Emission: `P(x_l = 1 | s) = p(s,l)`, `P(x_l = 0 | s) = 1 - p(s,l)`. Initial distribution `P(A) = P(B) = 0.5`.

Transition between consecutive SNPs `l` and `l+1`, with `d_l` in Morgan and switch rate `lambda > 0` (switches per Morgan):

    P(same ancestry)   = exp(-lambda * d_l)
    P(switch ancestry) = 1 - exp(-lambda * d_l)

`lambda` is a single scalar per (level, sample) pair. For each of the 12 pairs:

1. Run forward-backward with rescaling (or in log space) and report the posterior probabilities `gamma_l(A)`, `gamma_l(B)`.
2. Estimate `lambda_hat` by maximising the log-likelihood `log L(lambda)` (forward recursion, natural log) over the integer grid `lambda` in {1, 2, ..., 120}.
3. Posterior path: `A` if `gamma_l(A) >= gamma_l(B)`, else `B`.
4. Viterbi path: standard most probable state path; ties are resolved in favour of `A`.
5. Per-site maximum-likelihood baseline: `argmax_s P(x_l | s)`, ties resolved in favour of `A`.
6. Steps 3 and 4 are computed with the same `lambda_hat` as step 2.

Derived quantities (`switch count` = number of positions `l > 1` whose label differs from position `l-1`; `accuracy` = fraction of the 3000 sites agreeing with the truth line):

* `n_switches_posterior`, `n_switches_viterbi`
* `accuracy_posterior`, `accuracy_viterbi`, `accuracy_per_site_mle`
* implied admixture time `T_hat = lambda_hat / (2 * alpha * (1 - alpha))` generations with `alpha = 0.5`, i.e. `T_hat = 2 * lambda_hat`
* per level: `tract_inflation_factor` = (mean of `n_switches_posterior` over the four samples) / (mean of `n_switches_viterbi` over the four samples)
* per level: `feasible` = true iff the mean of `accuracy_viterbi` over the four samples is at least 0.95

## Outputs

1. `/app/output/posteriors/<level>/<sample>.tsv` - header line `snp_id<TAB>p_A<TAB>p_B`, then 3000 rows in `map.tsv` order, probabilities written with at least 6 decimals and summing to 1. Example row: `snp00042	0.98120000	0.01880000`
2. `/app/output/paths/<level>/<sample>.posterior.txt` and `/app/output/paths/<level>/<sample>.viterbi.txt` - a single line of 3000 characters `A`/`B`.
3. `/app/output/report.json`:
   * `per_chromosome`: 12 objects with keys `fst_level`, `sample`, `n_sites`, `lambda_hat`, `log_likelihood_at_lambda_hat`, `implied_admixture_time_generations`, `n_switches_posterior`, `n_switches_viterbi`, `accuracy_posterior`, `accuracy_viterbi`, `accuracy_per_site_mle`, `hmm_beats_per_site_mle`.
   * `per_fst_level`: 3 objects with keys `fst_level`, `mean_lambda_hat`, `mean_switches_posterior`, `mean_switches_viterbi`, `tract_inflation_factor`, `mean_accuracy_posterior`, `mean_accuracy_viterbi`, `mean_accuracy_per_site_mle`, `feasible`.
   * `summary`: keys `n_fst_levels`, `n_samples_per_level`, `n_feasible_fst_levels`.

Worked `per_chromosome` entry:
`{"fst_level": "fst_0002", "sample": "hap1", "n_sites": 3000, "lambda_hat": 37, "log_likelihood_at_lambda_hat": -4213.77, "implied_admixture_time_generations": 74.0, "n_switches_posterior": 214, "n_switches_viterbi": 61, "accuracy_posterior": 0.7903, "accuracy_viterbi": 0.9121, "accuracy_per_site_mle": 0.6312, "hmm_beats_per_site_mle": true}`

## Constraints

1. Every posterior table exists, carries the exact header, holds 3000 rows in `map.tsv` order, with `p_A + p_B = 1` to within 1e-6 and both values in [0, 1].
2. Every reported posterior probability agrees with the reference forward-backward posterior of the specified model (evaluated at the `lambda_hat` you report) to within 1e-4 absolute.
3. Every Viterbi path is the exact Viterbi path of the model (ties to `A`).
4. Every posterior path is the exact per-site argmax of the reported posterior.
5. `lambda_hat` is an integer in [1, 120] whose log-likelihood is within 0.5 nats of the grid maximum, and `log_likelihood_at_lambda_hat` matches the recomputed value to within 1.0.
6. `n_switches_posterior` and `n_switches_viterbi` equal the switch counts of the reference paths for that chromosome.
7. `accuracy_posterior`, `accuracy_viterbi` and `accuracy_per_site_mle` match the reference recomputation to within 0.002 absolute.
8. `hmm_beats_per_site_mle` equals `accuracy_viterbi > accuracy_per_site_mle` recomputed from the reference paths.
9. The per-level aggregates (`mean_lambda_hat`, `mean_switches_posterior`, `mean_switches_viterbi`, `tract_inflation_factor`, `mean_accuracy_posterior`, `mean_accuracy_viterbi`, `mean_accuracy_per_site_mle`) match the reference recomputation within 1 % relative or the absolute tolerance of the matching per-chromosome check, whichever is looser.
10. `implied_admixture_time_generations` equals `2 * lambda_hat` (1e-6), `feasible` follows the rule stated above for each level, and `summary.n_feasible_fst_levels` equals the number of levels flagged feasible.

## Scoring

The grader recomputes the whole reference decoder from a protected copy of the simulator; no stored answer exists. Score = weighted fraction of the ten constraints: posteriors 0.15, lambda_hat 0.15, Viterbi 0.12, switch counts 0.10, accuracies 0.10, per-level aggregates 0.10, format/artifacts 0.08, posterior paths 0.08, diagnostics (implied time, feasibility, summary count) 0.08, HMM-vs-MLE 0.04. Each constraint is scored as the fraction of its items (12 chromosomes, 3 levels, or 16 diagnostic items) that pass, so partial credit is granted per item.
