# Subtype-guided CXCR2 blockade in polymicrobial sepsis

You are analyzing a murine polymicrobial sepsis dataset to determine which transcriptomic subtype benefits from the CXCR2 antagonist Danirixin. The dataset contains paired blood and liver bulk RNA-seq counts, sample metadata, and MTS signature gene sets derived from human sepsis transcriptomic subtypes. Your job is to apply the signatures, classify each mouse, validate the subtypes against severity markers, quantify treatment effects, and recommend an endotype-guided strategy.

## Inputs
- `/app/data/metadata.tsv`: 36 mice (M01-M36) with columns `mouse_id`, `treatment` (Vehicle or Danirixin), `survival_7d` (0/1), `pathogen_log10` (log10 CFU), `IL6_pg_ml`, `NETosis_units`, `hepatic_injury_ALT`.
- `/app/data/blood_counts.tsv`: blood bulk RNA-seq counts, genes x samples (200 genes x 36 mice).
- `/app/data/liver_counts.tsv`: liver bulk RNA-seq counts, genes x samples (200 genes x 36 mice).
- `/app/data/signatures.json`: MTS1, MTS2, MTS3 signature gene sets (20 genes each).

## Task
1. Normalize each tissue count matrix to log2(CPM+1): compute CPM as counts / library_size * 1e6, then log2(CPM+1).
2. For each tissue and each gene, compute the z-score across the 36 samples using population standard deviation (ddof=0). If a gene has zero standard deviation, set its z-scores to 0.
3. For each MTS signature, compute the signature score per sample as the mean z-score of its genes.
4. For each mouse, compute the combined score for each MTS as the average of its blood signature score and liver signature score. Assign the mouse to the MTS with the highest combined score.
5. Merge the assigned subtypes with metadata. For each MTS, compute n, mean `pathogen_log10`, and mean `IL6_pg_ml`.
6. For each MTS, compute treatment effects as Danirixin minus Vehicle for: 7-day survival rate, mean `pathogen_log10`, mean `NETosis_units`, and mean `hepatic_injury_ALT`.
7. Recommend the MTS that should be treated with Danirixin and the MTS that should be avoided, based on the survival and bacterial dissemination trade-offs.

## Constraints
1. Write a single valid JSON object to `/app/output/result.json` with top-level keys `sample_subtypes`, `signature_scores`, `subtype_summary`, `treatment_effects`, `recommendation`.
2. `sample_subtypes` must contain exactly the 36 mouse IDs from metadata, each assigned to one of `MTS1`, `MTS2`, `MTS3`.
3. `signature_scores` must report the combined score for each mouse and each MTS, matching the reference z-score method within 1e-2.
4. Subtype assignment must agree with the reference method for at least 34 of 36 mice.
5. `subtype_summary` must report `n`, `mean_pathogen_log10`, and `mean_IL6` for each MTS, matching reference within 0.1 for pathogen and 5% for IL6. MTS3 must have higher mean pathogen burden and higher mean IL6 than MTS1.
6. `treatment_effects` must report `survival_diff`, `pathogen_diff`, `netosis_diff`, `hepatic_injury_diff` for each MTS, matching reference within 0.1 (survival), 0.2 (pathogen), 1.0 (NETosis), and 10.0 (hepatic injury). MTS3 survival_diff must be > 0; MTS1 pathogen_diff must be > 0, netosis_diff < 0, and hepatic_injury_diff < 0.
7. `recommendation` must set `preferred_subtype` to `MTS3` and `avoid_subtype` to `MTS1`, with a non-empty rationale string.

## Output
Write the JSON to `/app/output/result.json`. The object should contain:
- `sample_subtypes`: mapping mouse ID -> assigned MTS.
- `signature_scores`: mapping mouse ID -> mapping MTS -> combined score float.
- `subtype_summary`: mapping MTS -> {n, mean_pathogen_log10, mean_IL6}.
- `treatment_effects`: mapping MTS -> {survival_diff, pathogen_diff, netosis_diff, hepatic_injury_diff}.
- `recommendation`: {preferred_subtype, avoid_subtype, rationale}.

## Scoring
The grader recomputes the reference scores, labels, summaries, and effects from the input files and compares your output. It checks 10 sub-constraints across the 7 constraints above. Score = fraction of passed sub-constraints. Partial credit is awarded per sub-constraint. `tests/test.sh` writes `{"score": <score>, "reward": <score>}` to `/logs/verifier/reward.json`.
