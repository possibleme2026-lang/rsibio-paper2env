#!/usr/bin/env python3
"""Golden fixture grader smoke test: runs grade.py's logic against a
controlled prediction to prove the reward path is executable offline.

Not part of gate5; exists so a broken fixture fails loudly in CI here
rather than silently inside verify_task_dir.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

G = Path(__file__).resolve().parent / "fixtures" / "golden_task"

# The suite is stdlib-only by contract (see .github/workflows/ci.yml: no install
# step). The golden fixture's gold_gen.py needs numpy, so when numpy is absent
# this module skips instead of turning into a loader error -- a missing optional
# dependency is not a regression.
try:
    import numpy  # noqa: F401
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("numpy not installed; golden grader fixture needs it")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGoldenGraderExecutes(unittest.TestCase):
    def test_grade_logic_on_perfect_prediction(self):
        grade = _load("ggrade", G / "tests" / "grade.py")
        gold = _load("ggold", G / "tests" / "gold_gen.py").gold_bed()
        pred = [(c, s, e, 10.0) for (c, s, e) in gold]
        f1 = grade.interval_f1(pred, gold)
        self.assertGreater(f1, 0.9, f"perfect prediction F1 {f1}")
        self.assertTrue(grade.check_merge_distance(pred))
        self.assertTrue(grade.check_min_fold(pred))
        self.assertTrue(grade.check_sorted(pred))

    def test_grader_has_depth(self):
        lines = (G / "tests" / "grade.py").read_text(encoding="utf-8").splitlines()
        self.assertGreaterEqual(len(lines), 60)


if __name__ == "__main__":
    unittest.main()
