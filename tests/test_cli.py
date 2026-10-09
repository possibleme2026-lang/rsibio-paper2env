"""M6: CLI surface — every command family has an offline green path.

These tests exist because the CLI is the part that silently rots: a parser can
accept a command that no longer reaches its handler. The validation commands
(``validate-envs`` / ``run-weightfree`` / ``compare-harness`` / ``stage4``)
bypass argparse entirely and pass their argv straight through, so their flags
are exercised here end to end rather than being trusted to a REMAINDER
positional that argparse refuses to fill with a leading ``--flag``.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from bioforge_harbor import cli
from bioforge_harbor.models.records import PaperRecord

MINI_CORPUS = ROOT / "fixtures" / "corpus" / "mini_corpus.json"
MOCK = ROOT / "mock_responses"

PASSTHROUGH = {"validate-envs", "run-weightfree", "compare-harness", "stage4"}


def run(argv) -> tuple[int, str]:
    """Invoke the CLI the way a shell would.

    The argparse paths (--version, --help, bad usage) exit by raising
    SystemExit, so the status has to be recovered from the exception.
    """
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = cli.main(list(argv))
    except SystemExit as e:
        rc = e.code
    return rc, out.getvalue()


def pmc100() -> dict:
    """One screened paper: the same shape test_pipeline runs end to end."""
    return PaperRecord(
        paper_id="epmc-PMC100",
        title="scRNA-seq clustering benchmark with F1 and AUROC",
        abstract=("We benchmark single-cell RNA-seq clustering using scanpy across "
                  "datasets, reporting F1 score and adjusted Rand index for each "
                  "method with cross-validation."),
        pmcid="PMC100", pub_year=2022, open_access=True, license="CC BY",
        has_code_link=True,
    ).to_dict()


class TestParserBasics(unittest.TestCase):
    def test_version(self):
        rc, out = run(["--version"])
        self.assertEqual(rc, 0)
        self.assertIn("bioforge-harbor", out)

    def test_missing_and_unknown_command_exit_two(self):
        self.assertEqual(run([])[0], 2)
        self.assertEqual(run(["frobnicate"])[0], 2)

    def test_passthrough_table_covers_every_validation_command(self):
        # the dispatch table is derived from the parser; a command added there
        # without being tagged would otherwise never be reachable
        self.assertEqual(set(cli._passthrough_handlers()), PASSTHROUGH)

    def test_help_is_green_for_every_command(self):
        for cmd in ["curate", "run"] + sorted(PASSTHROUGH):
            self.assertEqual(run([cmd, "--help"])[0], 0, f"{cmd} --help")


class TestCurateCommand(unittest.TestCase):
    def test_curate_writes_report(self):
        tmp = Path(tempfile.mkdtemp())
        out = tmp / "curate.json"
        rc, _ = run(["curate", "--corpus", str(MINI_CORPUS), "--select-n", "3",
                     "--out", str(out)])
        self.assertEqual(rc, 0)
        rep = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(rep["screened"], 12)
        self.assertEqual(rep["selected"], 3)
        self.assertEqual(len(rep["selected_ids"]), 3)

    def test_curate_prints_when_no_out(self):
        rc, out = run(["curate", "--corpus", str(MINI_CORPUS), "--select-n", "2"])
        self.assertEqual(rc, 0)
        self.assertIn("selected_ids", out)


class TestRunCommand(unittest.TestCase):
    def _corpus(self, tmp: Path) -> Path:
        p = tmp / "corpus.json"
        p.write_text(json.dumps([pmc100()]), encoding="utf-8")
        return p

    def test_mock_backend_runs_the_whole_chain(self):
        tmp = Path(tempfile.mkdtemp())
        rc, out = run(["run", "--corpus", str(self._corpus(tmp)),
                       "--runs-dir", str(tmp / "runs"), "--backend", "mock",
                       "--mock-responses", str(MOCK)])
        self.assertEqual(rc, 0, out)
        self.assertEqual(json.loads(out)["packaged"], 1)
        # the task really landed on disk under tasks/<slug>, not just in state
        self.assertTrue((tmp / "tasks" / "mock-peak-caller" / "task.toml").is_file())
        self.assertTrue((tmp / "tasks" / "manifest.jsonl").is_file())

    def test_empty_mock_backend_fails_loud(self):
        # no --mock-responses: the mock has nothing to answer with, and the
        # pipeline must fail rather than ship an empty environment
        tmp = Path(tempfile.mkdtemp())
        rc, out = run(["run", "--corpus", str(self._corpus(tmp)),
                       "--runs-dir", str(tmp / "runs"), "--backend", "mock"])
        self.assertEqual(rc, 1, out)
        self.assertEqual(json.loads(out)["failed"], 1)


class TestValidationCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_validate_envs_without_args_is_a_usage_error(self):
        self.assertEqual(run(["validate-envs"])[0], 2)

    def test_validate_envs_manifest_mode(self):
        manifest = self.tmp / "results.json"
        manifest.write_text(json.dumps({"records": [
            {"env": "env-a", "built": True, "ran": True},
            {"env": "env-b", "built": True, "ran": False},
        ]}), encoding="utf-8")
        out = self.tmp / "stage1.json"
        rc, _ = run(["validate-envs", "--out", str(out), "--manifest", str(manifest)])
        self.assertEqual(rc, 0)
        rep = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(rep["n_envs"], 2)
        self.assertAlmostEqual(rep["build_success_rate"], 0.5)
        self.assertEqual(rep["unverified"], 0)
        self.assertIsNone(rep["expert_approval_rate"])

    def test_validate_envs_hybrid_merges_disk_and_facts(self):
        env_dir = self.tmp / "envs" / "env-a"
        env_dir.mkdir(parents=True)
        (env_dir / "task.toml").write_text('schema_version = "1.4"\n', encoding="utf-8")
        facts = self.tmp / "facts.json"
        facts.write_text(json.dumps([{"env": "env-a", "built": True, "ran": True}]),
                         encoding="utf-8")
        out = self.tmp / "stage1h.json"
        rc, _ = run(["validate-envs", "--out", str(out), "--envs-dir", str(self.tmp / "envs"),
                     "--executor", "hybrid", "--manifest", str(facts)])
        self.assertEqual(rc, 0)
        rep = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(rep["n_envs"], 1)
        self.assertEqual(rep["build_success_rate"], 1.0)

    def test_run_weightfree_dry_run_plans_every_attempt(self):
        cfg = self.tmp / "cfg.json"
        cfg.write_text(json.dumps({"models": ["m1", "m2"], "envs": ["e1", "e2", "e3"],
                                   "rounds": 3}), encoding="utf-8")
        out = self.tmp / "stage2.json"
        rc, _ = run(["run-weightfree", "--config", str(cfg), "--out", str(out),
                     "--dry-run"])
        self.assertEqual(rc, 0)
        rep = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(rep["dry_run"])
        self.assertEqual(rep["planned_attempts"], 18)   # 3 envs x 3 rounds x 2 models
        self.assertEqual(rep["plan_sample"][0], {"env": "e1", "round": 1, "model": "m1"})

    def test_run_weightfree_replay_scores_offline(self):
        cfg = self.tmp / "cfg.json"
        cfg.write_text(json.dumps({"models": ["m1"], "envs": ["e1", "e2"], "rounds": 2,
                                   "baseline": {"e1": 0.3, "e2": 0.9}}), encoding="utf-8")
        rows = [{"model": "m1", "env": "e1", "round": 1, "reward": 0.2},
                {"model": "m1", "env": "e1", "round": 2, "reward": 0.4},
                {"model": "m1", "env": "e2", "round": 1, "reward": 0.5},
                {"model": "m1", "env": "e2", "round": 2, "reward": 0.5}]
        replay = self.tmp / "attempts.jsonl"
        replay.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        out = self.tmp / "stage2r.json"
        rc, _ = run(["run-weightfree", "--config", str(cfg), "--out", str(out),
                     "--replay", str(replay)])
        self.assertEqual(rc, 0)
        rep = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(rep["first_attempt_improvement_rate"], 0.5)   # e1 up, e2 flat
        self.assertEqual(rep["exceeded_baseline_rate"], 0.5)           # 0.4>0.3, 0.5<0.9
        self.assertEqual(rep["continuous_improvement_rate"], 0.5)

    def test_compare_harness_reports_which_axis_dominates(self):
        base = self.tmp / "base.json"
        base.write_text(json.dumps({"cells": {"m1||h1": 0.2, "m1||h2": 0.2,
                                              "m2||h1": 0.8, "m2||h2": 0.8}}),
                        encoding="utf-8")
        variant = self.tmp / "variant.json"
        variant.write_text(json.dumps({"cells": {"m1||h1": 0.9, "m1||h2": 0.1,
                                                 "m2||h1": 0.9, "m2||h2": 0.1}}),
                           encoding="utf-8")
        out = self.tmp / "stage3.json"
        rc, _ = run(["compare-harness", "--baseline", str(base), "--variant", str(variant),
                     "--out", str(out)])
        self.assertEqual(rc, 0)
        rep = json.loads(out.read_text(encoding="utf-8"))
        # models differ in the baseline, harnesses differ in the variant
        self.assertTrue(rep["claim_check"]["model_dominant_baseline"])
        self.assertFalse(rep["claim_check"]["model_dominant_variant"])
        self.assertAlmostEqual(rep["per_model_delta"]["per_key"]["m1"], 0.3)

    def test_stage4_split_verify_and_contamination_exit_codes(self):
        corpus = self.tmp / "corpus.json"
        corpus.write_text(json.dumps([{"paper_id": "epmc-PMC1", "pub_year": 2022},
                                      {"paper_id": "epmc-PMC2", "pub_year": 2025}]),
                          encoding="utf-8")
        lock = self.tmp / "lock.json"
        self.assertEqual(run(["stage4", "split-holdout", "--corpus", str(corpus),
                              "--lock", str(lock)])[0], 0)
        self.assertEqual(json.loads(lock.read_text(encoding="utf-8"))["holdout"],
                         ["epmc-PMC2"])
        self.assertEqual(run(["stage4", "verify-lock", "--corpus", str(corpus),
                              "--lock", str(lock)])[0], 0)

        texts = self.tmp / "texts"
        texts.mkdir()
        shared = "we benchmark clustering with scanpy and report f1 score per method"
        (texts / "epmc-PMC1.txt").write_text(shared, encoding="utf-8")
        (texts / "epmc-PMC2.txt").write_text(shared, encoding="utf-8")     # verbatim leak
        self.assertEqual(run(["stage4", "check-contamination", "--lock", str(lock),
                              "--texts-dir", str(texts)])[0], 1)
        (texts / "epmc-PMC2.txt").write_text(
            "a completely unrelated ribosome profiling survey of yeast strains",
            encoding="utf-8")
        self.assertEqual(run(["stage4", "check-contamination", "--lock", str(lock),
                              "--texts-dir", str(texts)])[0], 0)


if __name__ == "__main__":
    unittest.main()
