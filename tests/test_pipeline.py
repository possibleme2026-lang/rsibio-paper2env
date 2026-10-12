"""M5: end-to-end pipeline over MockBackend — every stage really executes."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from bioforge_harbor.backends import MockBackend
from bioforge_harbor.models.records import PaperRecord
from bioforge_harbor.pipeline.runner import curate_corpus, run_pipeline
from bioforge_harbor.pipeline.state import load_state
from bioforge_harbor.harbor_emit import verify_task_dir

MOCK = ROOT / "mock_responses"


def backend() -> MockBackend:
    return MockBackend.from_dir(MOCK)


def rec(pid: str = "epmc-PMC100") -> PaperRecord:
    return PaperRecord(
        paper_id=pid, title="scRNA-seq clustering benchmark with F1 and AUROC",
        abstract=("We benchmark single-cell RNA-seq clustering using scanpy "
                  "across datasets, reporting F1 score and adjusted Rand index "
                  "for each method with cross-validation."),
        pmcid=pid.replace("epmc-", ""), pub_year=2022, open_access=True,
        license="CC BY", has_code_link=True,
    )


class TestEndToEnd(unittest.TestCase):
    def test_full_chain_packages(self):
        tmp = Path(tempfile.mkdtemp())
        summary = run_pipeline([rec()], backend(), runs_dir=tmp / "runs")
        self.assertEqual(summary.packaged, 1, summary.to_dict())
        # the packaged task exists and passes the offline gates
        task_dirs = [d for d in (tmp / "tasks").iterdir() if d.is_dir()]
        self.assertEqual(len(task_dirs), 1)
        out = verify_task_dir(task_dirs[0], run_harbor_gate=False)
        self.assertEqual(out.status, "ok", out.to_dict())
        # set-level manifest written
        manifest = (tmp / "tasks" / "manifest.jsonl").read_text(encoding="utf-8")
        self.assertIn("bioforge/scrna-cluster-bench", manifest)

    def test_state_records_all_stages(self):
        tmp = Path(tempfile.mkdtemp())
        run_pipeline([rec()], backend(), runs_dir=tmp / "runs")
        st = load_state(tmp / "runs" / "epmc-PMC100")
        self.assertEqual(st.completed(),
                         ["curate", "construct", "reproduce", "review", "package"])

    def test_resume_skips_completed(self):
        tmp = Path(tempfile.mkdtemp())
        b1 = backend()
        run_pipeline([rec()], b1, runs_dir=tmp / "runs")
        calls_after_first = len(b1.calls)
        b2 = backend()
        summary = run_pipeline([rec()], b2, runs_dir=tmp / "runs", resume=True)
        # nothing re-executed: zero backend calls on the resume pass
        self.assertEqual(b2.calls, [])
        self.assertEqual(summary.packaged, 1)
        self.assertGreater(calls_after_first, 0)

    def test_force_reruns(self):
        tmp = Path(tempfile.mkdtemp())
        run_pipeline([rec()], backend(), runs_dir=tmp / "runs")
        b2 = backend()
        run_pipeline([rec()], b2, runs_dir=tmp / "runs", force=True)
        self.assertGreater(len(b2.calls), 0)

    def test_broken_blueprint_fails_closed(self):
        tmp = Path(tempfile.mkdtemp())
        b = backend()
        b.set("constructor.build", {"error": "not computable"})
        summary = run_pipeline([rec()], b, runs_dir=tmp / "runs")
        self.assertEqual(summary.failed, 1)
        status = next(iter(summary.per_paper.values()))
        self.assertTrue(status.startswith("failed"), status)

    def test_biosafety_block_stops_packaging(self):
        tmp = Path(tempfile.mkdtemp())
        b = backend()
        # constructor emits a task whose instruction trips the denylist
        bp = json.loads((MOCK / "constructor.build.json").read_text(encoding="utf-8"))
        bp["instruction_md"] += "\nGoal: gain of function enhancement\n"
        b.set("constructor.build", bp)
        summary = run_pipeline([rec()], b, runs_dir=tmp / "runs")
        self.assertEqual(summary.failed, 1)
        st = load_state(tmp / "runs" / "epmc-PMC100")
        self.assertEqual(st.attempts["review"].status, "failed")
        self.assertNotIn("package", st.completed())

    def test_curate_then_run(self):
        tmp = Path(tempfile.mkdtemp())
        closed = rec("epmc-PMC104")
        closed.open_access = False
        corpus = [rec("epmc-PMC100"), rec("epmc-PMC101"), closed]
        report = curate_corpus(corpus, select_n=10)
        self.assertEqual(report.selected, 2)  # PMC104 not open access
        summary = run_pipeline(corpus, backend(), runs_dir=tmp / "runs",
                               selected_ids=set(report.selected_ids))
        self.assertEqual(summary.papers, 2)


if __name__ == "__main__":
    unittest.main()
