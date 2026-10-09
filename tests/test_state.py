"""M5: state machine — ordering, resume, atomicity, corrupt refusal."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from bioforge_harbor.pipeline.state import (
    STAGES, PaperState, StateError, load_state, save_state,
)


class TestStateMachine(unittest.TestCase):
    def test_stage_order(self):
        self.assertEqual(STAGES, ("curate", "construct", "reproduce", "review", "package"))

    def test_advance_requires_predecessor(self):
        s = PaperState(paper_id="p1")
        with self.assertRaises(StateError):
            s.advance("construct")  # curate not done yet
        s.advance("curate", status="ok")
        s.advance("construct", status="ok")
        self.assertEqual(s.completed(), ["curate", "construct"])

    def test_next_stage(self):
        s = PaperState(paper_id="p1")
        self.assertEqual(s.next_stage(), "curate")
        s.advance("curate", status="ok")
        self.assertEqual(s.next_stage(), "construct")

    def test_done_after_all(self):
        s = PaperState(paper_id="p1")
        for st in STAGES:
            s.advance(st, status="ok")
        self.assertTrue(s.is_done())
        self.assertIsNone(s.next_stage())

    def test_failed_attempt_blocks_advance(self):
        s = PaperState(paper_id="p1")
        s.advance("curate", status="ok")
        s.advance("construct", status="failed", error="boom")
        self.assertFalse(s.can_resume_into("reproduce"))

    def test_skipped_counts_as_passable(self):
        s = PaperState(paper_id="p1")
        s.advance("curate", status="ok")
        s.advance("construct", status="skipped")
        self.assertTrue(s.can_resume_into("reproduce"))

    def test_unknown_stage_raises(self):
        s = PaperState(paper_id="p1")
        with self.assertRaises(StateError):
            s.advance("teleport")


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        d = Path(tempfile.mkdtemp())
        s = PaperState(paper_id="p1")
        s.advance("curate", status="ok", digest="abc")
        save_state(d, s)
        back = load_state(d)
        self.assertEqual(back.paper_id, "p1")
        self.assertEqual(back.attempts["curate"].digest, "abc")

    def test_missing_state_fresh(self):
        d = Path(tempfile.mkdtemp()) / "nope"
        s = load_state(d)
        self.assertEqual(s.paper_id, "nope")   # derived from dir name
        self.assertEqual(s.completed(), [])

    def test_corrupt_state_refused(self):
        d = Path(tempfile.mkdtemp())
        (d / "state.json").write_text("{ not json", encoding="utf-8")
        with self.assertRaises(StateError):
            load_state(d)
        s = load_state(d, force=True)
        self.assertEqual(s.completed(), [])

    def test_atomic_no_temp_leftovers(self):
        d = Path(tempfile.mkdtemp())
        s = PaperState(paper_id="p1")
        for st in STAGES:
            s.advance(st, status="ok")
            save_state(d, s)
        leftovers = [f for f in d.iterdir() if f.name.startswith(".tmp-")]
        self.assertEqual(leftovers, [])

    def test_resume_semantics(self):
        # simulate crash after curate: reload -> next stage is construct
        d = Path(tempfile.mkdtemp())
        s = PaperState(paper_id="p1")
        s.advance("curate", status="ok")
        save_state(d, s)
        r = load_state(d)
        self.assertEqual(r.next_stage(), "construct")
        self.assertTrue(r.can_resume_into("construct"))
        self.assertFalse(r.can_resume_into("review"))


if __name__ == "__main__":
    unittest.main()
