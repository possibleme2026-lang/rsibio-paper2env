"""M1: data models — records, blueprint validation, report envelope."""

from __future__ import annotations

import unittest

from bioforge_harbor.models.records import PaperRecord, slugify
from bioforge_harbor.models.blueprint import TaskBlueprint, Provenance
from bioforge_harbor.models.reports import (
    Report, SelectionReport, EmitReport, VerificationReport, StageResult,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import good_blueprint, good_provenance  # noqa: E402


class TestPaperRecord(unittest.TestCase):
    def test_from_europepmc_minimal(self):
        hit = {"pmcid": "PMC1234567", "title": "T", "doi": "10.1/x",
               "pubYear": "2023", "isOpenAccess": "Y", "licence": "CC BY"}
        rec = PaperRecord.from_europepmc(hit)
        self.assertEqual(rec.paper_id, "epmc-PMC1234567")
        self.assertTrue(rec.open_access)
        self.assertEqual(rec.year_or_zero, 2023)

    def test_requires_id(self):
        with self.assertRaises(ValueError):
            PaperRecord.from_europepmc({"title": "no ids"})

    def test_roundtrip_dict(self):
        rec = PaperRecord(paper_id="p1", title="T", subjects=["a", "b"])
        back = PaperRecord.from_dict(rec.to_dict())
        self.assertEqual(back.paper_id, "p1")
        self.assertEqual(back.subjects, ["a", "b"])

    def test_year_from_published_fallback(self):
        rec = PaperRecord(paper_id="p", title="t", published="2019-05-01")
        self.assertEqual(rec.year_or_zero, 2019)

    def test_slugify(self):
        self.assertEqual(slugify("ChIP-seq: Peak Calling!!"), "chip-seq-peak-calling")
        self.assertEqual(slugify("!!!"), "untitled")


class TestBlueprint(unittest.TestCase):
    def test_good_blueprint_valid(self):
        self.assertEqual(good_blueprint().validate(), [])

    def test_bad_name_flagged(self):
        b = good_blueprint(task_name="bad name")
        self.assertTrue(any("task_name" in e for e in b.validate()))

    def test_missing_test_sh_flagged(self):
        b = good_blueprint(test_files={"tests/other.py": "x"})
        self.assertTrue(any("test.sh" in e for e in b.validate()))

    def test_disallowed_base_flagged(self):
        b = good_blueprint(dockerfile="FROM evil/untrusted:latest\nRUN x\n")
        self.assertTrue(any("base" in e for e in b.validate()))

    def test_no_from_line_flagged(self):
        b = good_blueprint(dockerfile="RUN echo hi\n")
        self.assertTrue(any("base" in e for e in b.validate()))

    def test_missing_provenance_flagged(self):
        b = good_blueprint(provenance=None)
        self.assertTrue(any("provenance" in e for e in b.validate()))

    def test_roundtrip_dict(self):
        b = good_blueprint()
        back = TaskBlueprint.from_dict(b.to_dict())
        self.assertEqual(back.task_name, b.task_name)
        self.assertIsInstance(back.provenance, Provenance)
        self.assertEqual(back.provenance.paper_id, "epmc-PMC1234567")


class TestReports(unittest.TestCase):
    def test_ok_semantics(self):
        self.assertTrue(Report(stage="x", status="ok").ok)
        self.assertTrue(Report(stage="x", status="warning").ok)
        self.assertFalse(Report(stage="x", status="failed", errors=["e"]).ok)
        self.assertFalse(Report(stage="x", status="ok", errors=["e"]).ok)

    def test_lost_semantics_field(self):
        r = EmitReport(stage="emit", lost_semantics=["gpu_types dropped"])
        d = r.to_dict()
        self.assertEqual(d["lost_semantics"], ["gpu_types dropped"])
        self.assertEqual(EmitReport.from_dict(d).lost_semantics, ["gpu_types dropped"])

    def test_verification_real_gate_status(self):
        v = VerificationReport(gates={"harbor_real": {"status": "UNVERIFIED"}})
        self.assertEqual(v.real_gate_status, "UNVERIFIED")
        self.assertEqual(VerificationReport().real_gate_status, "missing")

    def test_stage_result_roundtrip(self):
        s = StageResult(stage="curate", status="ok", digest="abc")
        self.assertEqual(StageResult.from_dict(s.to_dict()).digest, "abc")


if __name__ == "__main__":
    unittest.main()
