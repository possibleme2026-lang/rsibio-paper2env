"""M6: pure metric functions for the four-stage protocol (paper section 4).

Known limitations deliberately NOT pinned by these tests — asserting them now
would freeze the wrong behaviour into the suite, and each one is written down
so it gets re-checked before any number reaches the paper:

  * ``variance_decomposition`` is only interpretable on a complete rectangular
    (model x harness) grid; with missing cells both components pick up
    different constant offsets and "which axis dominates" stops meaning
    anything. Tests therefore only use complete grids.
  * ``ngram_contamination`` can miss a verbatim leak when the probe text has
    fewer than n tokens (its single short n-gram can never match an 8-gram).
  * ``baseline_reproduction_consistency`` raises on an explicit
    ``tolerance=None`` instead of falling back to the default.
  * ``harness_delta`` treats a missing key as 0.0, so "absent" reads as
    "regressed".
  * ``stage1_summary`` does not expose the review/probe denominators.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from bioforge_harbor.validation.metrics import (
    Attempt,
    baseline_reproduction_consistency,
    build_success_rate,
    continuous_improvement_rate,
    control_contrast,
    exceeded_baseline_rate,
    expert_approval_rate,
    first_attempt_improvement_rate,
    harness_delta,
    ngram_contamination,
    shortcut_detection_rate,
    shortcut_success_rate,
    stage1_summary,
    stage2_summary,
    time_split,
    variance_decomposition,
)


def A(model: str, env: str, rnd: int, reward: float) -> Attempt:
    return Attempt(model=model, env=env, round=rnd, reward=reward)


# --------------------------------------------------------------------------
# Stage 1: environment quality (paper 4.1)
# --------------------------------------------------------------------------
class TestStage1Metrics(unittest.TestCase):
    def test_build_success_needs_build_and_run(self):
        recs = [{"built": True, "ran": True},      # shipped
                {"built": True, "ran": False},     # built, never ran
                {"built": False, "ran": False}]    # never built
        self.assertAlmostEqual(build_success_rate(recs), 1 / 3)

    def test_build_success_empty_is_none(self):
        # None, never 0.0: an empty set must not read as "everything failed"
        self.assertIsNone(build_success_rate([]))

    def test_reproduction_exact_and_beyond_tolerance(self):
        self.assertEqual(baseline_reproduction_consistency(
            [{"baseline": 0.80, "paper": 0.80}]), 1.0)
        # |0.65 - 0.80| == tolerance 0.15 -> agreement exactly 0
        self.assertEqual(baseline_reproduction_consistency(
            [{"baseline": 0.65, "paper": 0.80, "tolerance": 0.15}]), 0.0)
        # beyond tolerance is clamped, never negative
        self.assertEqual(baseline_reproduction_consistency(
            [{"baseline": 0.50, "paper": 0.80, "tolerance": 0.15}]), 0.0)

    def test_reproduction_gives_partial_credit(self):
        got = baseline_reproduction_consistency(
            [{"baseline": 0.55, "paper": 0.50, "tolerance": 0.10}])
        self.assertAlmostEqual(got, 0.5)

    def test_reproduction_skips_records_missing_either_side(self):
        recs = [{"baseline": 0.8, "paper": 0.8}, {"baseline": 0.5}, {"paper": 0.5}]
        self.assertEqual(baseline_reproduction_consistency(recs), 1.0)
        self.assertIsNone(baseline_reproduction_consistency([{}]))

    def test_expert_approval_denominator_is_reviewed_only(self):
        recs = [{"reviewed": True, "approved": True},
                {"reviewed": True, "approved": False},
                {"reviewed": False, "approved": True}]   # not reviewed: excluded
        self.assertEqual(expert_approval_rate(recs), 0.5)
        # nobody reviewed anything -> None, not a 0% approval headline
        self.assertIsNone(expert_approval_rate([{"reviewed": False, "approved": True}]))

    def test_shortcut_rates_point_in_opposite_directions(self):
        probes = [{"shortcut_passed": True}, {"shortcut_passed": False},
                  {"shortcut_passed": False}, {"shortcut_passed": False}]
        # success = the trivial strategy fooled the grader -> defect rate, LOWER
        self.assertAlmostEqual(shortcut_success_rate(probes), 0.25)
        # detection = it was caught -> HIGHER is better
        self.assertAlmostEqual(shortcut_detection_rate(probes), 0.75)
        self.assertIsNone(shortcut_success_rate([]))
        self.assertIsNone(shortcut_detection_rate([]))

    def test_stage1_summary_exposes_both_shortcut_directions(self):
        s = stage1_summary([{"built": True, "ran": True}], [], [{"shortcut_passed": True}])
        self.assertEqual(s["stage"], 1)
        self.assertEqual(s["n_envs"], 1)
        self.assertEqual(s["build_success_rate"], 1.0)
        self.assertEqual(s["shortcut_success_rate"], 1.0)
        self.assertEqual(s["shortcut_detection_rate"], 0.0)
        self.assertIsNone(s["expert_approval_rate"])


# --------------------------------------------------------------------------
# Stage 2: weight-free self-improvement (paper 4.2)
# --------------------------------------------------------------------------
class TestStage2Metrics(unittest.TestCase):
    def test_first_attempt_improvement_is_per_environment(self):
        attempts = [A("m1", "envA", 1, 0.2), A("m1", "envA", 2, 0.5),   # improved
                    A("m1", "envB", 1, 0.5), A("m1", "envB", 2, 0.5)]   # flat
        self.assertEqual(first_attempt_improvement_rate(attempts, {}), 0.5)

    def test_first_attempt_improvement_any_model_is_enough(self):
        attempts = [A("m1", "envA", 1, 0.3), A("m1", "envA", 2, 0.3),
                    A("m2", "envA", 1, 0.3), A("m2", "envA", 2, 0.9)]
        self.assertEqual(first_attempt_improvement_rate(attempts, {}), 1.0)

    def test_first_attempt_improvement_without_a_second_round(self):
        self.assertEqual(first_attempt_improvement_rate([A("m1", "envA", 1, 0.9)], {}), 0.0)
        self.assertIsNone(first_attempt_improvement_rate([], {}))

    def test_exceeded_baseline_uses_the_best_attempt_per_environment(self):
        attempts = [A("m1", "envA", 1, 0.3), A("m1", "envA", 2, 0.5),   # best 0.5 > 0.4
                    A("m1", "envB", 1, 0.5), A("m1", "envB", 2, 0.5)]   # best 0.5 < 0.6
        self.assertEqual(exceeded_baseline_rate(attempts, {"envA": 0.4, "envB": 0.6}), 0.5)
        self.assertIsNone(exceeded_baseline_rate([], {"envA": 0.4}))

    def test_continuous_improvement_is_strictly_monotone(self):
        up = [A("m1", "envA", 1, 0.1), A("m1", "envA", 2, 0.2), A("m1", "envA", 3, 0.3)]
        flat = [A("m1", "envB", 1, 0.5), A("m1", "envB", 2, 0.5), A("m1", "envB", 3, 0.6)]
        self.assertEqual(continuous_improvement_rate(up + flat), 0.5)
        # a one-round trajectory has no trend: excluded from the denominator
        self.assertEqual(continuous_improvement_rate([A("m1", "envC", 1, 0.9)] + up), 1.0)
        self.assertIsNone(continuous_improvement_rate([A("m1", "envC", 1, 0.9)]))

    def test_stage2_summary_shape(self):
        s = stage2_summary([A("m1", "envA", 1, 0.2), A("m2", "envA", 2, 0.4)],
                           {"envA": 0.3})
        self.assertEqual(s["stage"], 2)
        self.assertEqual(s["n_models"], 2)
        self.assertEqual(s["n_envs"], 1)
        self.assertEqual(s["n_rounds"], 2)


# --------------------------------------------------------------------------
# Stage 3: harness optimization (paper 4.3)
# --------------------------------------------------------------------------
class TestStage3Metrics(unittest.TestCase):
    def test_model_dominates_when_harnesses_agree(self):
        # A is good under both harnesses, B is bad under both: the spread is
        # between models, not between harnesses
        cells = {("A", "H1"): 0.9, ("A", "H2"): 0.9,
                 ("B", "H1"): 0.1, ("B", "H2"): 0.1}
        vd = variance_decomposition(cells)
        self.assertEqual(vd["dominant"], "model")
        self.assertGreater(vd["model_variance"], vd["harness_variance"])
        self.assertAlmostEqual(vd["harness_variance"], 0.0)

    def test_harness_dominates_when_models_agree(self):
        cells = {("A", "H1"): 0.9, ("A", "H2"): 0.1,
                 ("B", "H1"): 0.9, ("B", "H2"): 0.1}
        vd = variance_decomposition(cells)
        self.assertEqual(vd["dominant"], "harness")
        self.assertAlmostEqual(vd["model_variance"], 0.0)
        self.assertGreater(vd["harness_variance"], 0.0)

    def test_empty_grid_reports_nothing(self):
        vd = variance_decomposition({})
        self.assertIsNone(vd["model_variance"])
        self.assertIsNone(vd["harness_variance"])
        self.assertIsNone(vd["dominant"])

    def test_harness_delta(self):
        d = harness_delta({"a": 0.5, "b": 0.2}, {"a": 0.7, "b": 0.1})
        self.assertAlmostEqual(d["per_key"]["a"], 0.2)
        self.assertAlmostEqual(d["per_key"]["b"], -0.1)
        self.assertAlmostEqual(d["mean_delta"], 0.05)
        self.assertEqual(d["n_improved"], 1)
        self.assertEqual(d["n_regressed"], 1)


# --------------------------------------------------------------------------
# Stage 4: held-out evaluation (paper 4.4)
# --------------------------------------------------------------------------
class TestStage4Metrics(unittest.TestCase):
    def test_time_split_windows(self):
        recs = [{"paper_id": "p-old", "pub_year": 2020},
                {"paper_id": "p-edge", "pub_year": 2023},   # cutoff is inclusive
                {"paper_id": "p-gap", "pub_year": 2024},    # in the gap -> excluded
                {"paper_id": "p-h1", "pub_year": 2025},
                {"paper_id": "p-h2", "pub_year": 2026},
                {"paper_id": "p-unknown"}]                  # no year -> excluded
        s = time_split(recs, train_cutoff=2023, holdout_start=2025, holdout_end=2026)
        self.assertEqual(s["train"], ["p-old", "p-edge"])
        self.assertEqual(s["holdout"], ["p-h1", "p-h2"])
        self.assertEqual(s["excluded"], ["p-gap", "p-unknown"])

    def test_ngram_contamination_identical_and_disjoint(self):
        text = "we benchmark clustering with scanpy and report f1 score per method"
        self.assertEqual(ngram_contamination(text, text), 1.0)
        self.assertEqual(ngram_contamination(text, "an unrelated pottery glaze study"), 0.0)

    def test_ngram_contamination_empty_probe(self):
        self.assertEqual(ngram_contamination("", "anything at all here"), 0.0)

    def test_control_contrast_lift_over_no_environment(self):
        arms = {"no_environment": 0.30, "random_environment": 0.35,
                "auto_environment": 0.50, "auto_environment_reviewed": 0.52}
        lift = control_contrast(arms)["lift_over_no_env"]
        self.assertAlmostEqual(lift["auto_environment"], 0.20)
        self.assertAlmostEqual(lift["auto_environment_reviewed"], 0.22)
        # the causal ordering the paper claims: reviewed > auto > random > none
        self.assertGreater(lift["auto_environment_reviewed"], lift["auto_environment"])
        self.assertGreater(lift["auto_environment"], lift["random_environment"])

    def test_control_contrast_without_a_baseline_arm(self):
        out = control_contrast({"auto_environment": 0.5})
        self.assertEqual(out["lift_over_no_env"], {})
        self.assertEqual(out["arms"], {"auto_environment": 0.5})


if __name__ == "__main__":
    unittest.main()
