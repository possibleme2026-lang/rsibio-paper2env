# Review notes

- Synthetic data only; no real sequencing data shipped (1.6 MB x 2 bedGraph,
  deterministic seeds 42/43; generator shipped as environment/data/gen_inputs.py).
- Gold standard is re-derived at verify time by tests/gold_gen.py from an
  independent oracle seed (1337) with boundary jitter; it is never baked into
  the agent image.
- Shortcut probe: copying peaks.gff or any shipped input into predictions.bed
  must score ~0 (F1 vs jittered oracle + gold-access constraint check).
- Grader applies partial credit: score = 0.7 * interval F1 + 0.3 * constraint
  pass rate (merge distance, min fold, sorting, gold access).
