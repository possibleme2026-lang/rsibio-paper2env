"""M6: Stage-4 held-out split lock + contamination audit (paper 4.4).

The point of this module is that the evaluation contract is frozen BEFORE any
weights move: the holdout set is locked with a corpus digest, and a later run
that quietly re-derives a different holdout set is a hard failure, not a
warning. These tests are the tripwire for that property.

Note: ``mini_corpus.json`` is train-heavy under the default window (11 train /
1 holdout / 0 excluded) and has no year-0 record, so the three-way split is
tested on synthetic years instead.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from bioforge_harbor.validation.stage4_heldout import (
    check_contamination,
    main as stage4_main,
    split_and_lock,
    verify_lock,
)

CORPUS = [
    {"paper_id": "epmc-PMC1", "pub_year": 2022},   # train
    {"paper_id": "epmc-PMC2", "pub_year": 2023},   # train (cutoff inclusive)
    {"paper_id": "epmc-PMC3", "pub_year": 2025},   # holdout
    {"paper_id": "epmc-PMC4", "pub_year": 2026},   # holdout
    {"paper_id": "epmc-PMC5", "pub_year": 2024},   # gap year -> excluded
    {"paper_id": "epmc-PMC6"},                     # unknown year -> excluded
]

LEAKED = ("we benchmark single-cell clustering with scanpy and report f1 score "
          "adjusted rand index for every method in the survey")


class Stage4Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.corpus = self.tmp / "corpus.json"
        self.corpus.write_text(json.dumps({"records": CORPUS}), encoding="utf-8")
        self.lock = self.tmp / "holdout.lock.json"


class TestSplitAndLock(Stage4Fixture):
    def test_three_way_split_is_frozen_into_the_lock(self):
        lock = split_and_lock(self.corpus, lock_path=self.lock)
        self.assertTrue(lock["locked"])
        self.assertEqual(lock["train"], ["epmc-PMC1", "epmc-PMC2"])
        self.assertEqual(lock["holdout"], ["epmc-PMC3", "epmc-PMC4"])
        self.assertEqual(lock["excluded"], ["epmc-PMC5", "epmc-PMC6"])
        self.assertTrue(self.lock.is_file())
        self.assertTrue(lock["corpus_digest"])

    def test_verify_lock_accepts_an_unchanged_corpus(self):
        split_and_lock(self.corpus, lock_path=self.lock)
        ok, msg = verify_lock(self.corpus, self.lock)
        self.assertTrue(ok, msg)

    def test_verify_lock_rejects_corpus_drift(self):
        split_and_lock(self.corpus, lock_path=self.lock)
        drifted = CORPUS + [{"paper_id": "epmc-PMC9", "pub_year": 2025}]
        self.corpus.write_text(json.dumps({"records": drifted}), encoding="utf-8")
        ok, msg = verify_lock(self.corpus, self.lock)
        self.assertFalse(ok)
        self.assertIn("digest mismatch", msg)

    def test_verify_lock_rejects_a_retuned_window(self):
        # Same corpus (digest still matches) but the lock's own window no longer
        # reproduces its holdout set: the split is stale, not the data.
        split_and_lock(self.corpus, lock_path=self.lock)
        lock = json.loads(self.lock.read_text(encoding="utf-8"))
        lock["params"] = {"train_cutoff": 2023, "holdout_start": 2024, "holdout_end": 2026}
        self.lock.write_text(json.dumps(lock), encoding="utf-8")
        ok, msg = verify_lock(self.corpus, self.lock)
        self.assertFalse(ok)
        self.assertIn("drifted", msg)


class TestContamination(Stage4Fixture):
    def setUp(self):
        super().setUp()
        split_and_lock(self.corpus, lock_path=self.lock)
        self.texts = self.tmp / "texts"
        self.texts.mkdir()
        (self.texts / "epmc-PMC1.txt").write_text(LEAKED, encoding="utf-8")          # train
        (self.texts / "epmc-PMC2.txt").write_text("independent methods section", encoding="utf-8")

    def test_verbatim_leak_is_flagged(self):
        (self.texts / "epmc-PMC3.txt").write_text(LEAKED, encoding="utf-8")
        (self.texts / "epmc-PMC4.txt").write_text(LEAKED, encoding="utf-8")
        rep = check_contamination(self.lock, self.texts)
        self.assertEqual(rep["n_holdout"], 2)
        self.assertEqual(rep["n_flagged"], 2)
        self.assertEqual(rep["max_overlap"], 1.0)

    def test_clean_holdout_is_not_flagged(self):
        (self.texts / "epmc-PMC3.txt").write_text(
            "a study of pottery glaze chemistry in ancient kilns across regions",
            encoding="utf-8")
        (self.texts / "epmc-PMC4.txt").write_text(
            "survey of ribosome profiling protocols and their failure modes",
            encoding="utf-8")
        rep = check_contamination(self.lock, self.texts)
        self.assertEqual(rep["n_with_text"], 2)
        self.assertEqual(rep["n_flagged"], 0)

    def test_missing_text_is_recorded_not_crashed(self):
        (self.texts / "epmc-PMC3.txt").write_text(LEAKED, encoding="utf-8")
        # epmc-PMC4 has no text file at all
        rep = check_contamination(self.lock, self.texts)
        self.assertEqual(rep["n_with_text"], 1)
        by_id = {r["paper_id"]: r for r in rep["per_paper"]}
        self.assertIsNone(by_id["epmc-PMC4"]["overlap"])
        self.assertEqual(by_id["epmc-PMC4"]["note"], "text missing")

    def test_cli_exit_codes(self):
        (self.texts / "epmc-PMC3.txt").write_text(LEAKED, encoding="utf-8")          # leaked
        (self.texts / "epmc-PMC4.txt").write_text(
            "unrelated ribosome profiling survey of budding yeast strains",
            encoding="utf-8")
        # split + verify are green
        self.assertEqual(stage4_main(["split-holdout", "--corpus", str(self.corpus),
                                      "--lock", str(self.tmp / "l2.json")]), 0)
        self.assertEqual(stage4_main(["verify-lock", "--corpus", str(self.corpus),
                                      "--lock", str(self.lock)]), 0)
        # a leak is a real finding -> non-zero exit; a green exit here would be
        # exactly the fake pass this protocol exists to prevent
        self.assertEqual(stage4_main(["check-contamination", "--lock", str(self.lock),
                                      "--texts-dir", str(self.texts)]), 1)
        report = self.tmp / "contamination.json"
        self.assertEqual(stage4_main(["check-contamination", "--lock", str(self.lock),
                                      "--texts-dir", str(self.texts),
                                      "--out", str(report)]), 1)
        self.assertTrue(report.is_file())

    def test_cli_clean_holdout_exits_zero(self):
        (self.texts / "epmc-PMC3.txt").write_text(
            "unrelated ribosome profiling survey of budding yeast strains",
            encoding="utf-8")
        (self.texts / "epmc-PMC4.txt").write_text(
            "another disjoint sentence about kiln temperatures and glazes",
            encoding="utf-8")
        self.assertEqual(stage4_main(["check-contamination", "--lock", str(self.lock),
                                      "--texts-dir", str(self.texts)]), 0)


if __name__ == "__main__":
    unittest.main()
