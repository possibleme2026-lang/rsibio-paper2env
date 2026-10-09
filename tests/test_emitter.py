"""M4: emitter — deterministic Harbor output + lost-semantics accounting."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from helpers import good_blueprint  # noqa: E402
from bioforge_harbor.harbor_emit import emit_task  # noqa: E402
from bioforge_harbor.harbor_emit.tomlx import dump_toml  # noqa: E402


class TestToml(unittest.TestCase):
    def test_scalars_and_tables(self):
        text = dump_toml({
            "schema_version": "1.4",
            "task": {"name": "bioforge/x", "version": "1.0.0"},
            "environment": {"cpus": 2, "memory_mb": 2048, "gpus": 0},
            "agent": {"timeout_sec": 120.0},
        })
        self.assertIn('schema_version = "1.4"', text)
        self.assertIn("[task]", text)
        self.assertIn("cpus = 2", text)
        self.assertIn("timeout_sec = 120.0", text)
        # root scalars must precede any [table] header
        self.assertLess(text.index("schema_version"), text.index("[task]"))

    def test_arrays(self):
        text = dump_toml({"task": {"name": "a/b", "keywords": ["x", "y"]}})
        self.assertIn('keywords = ["x", "y"]', text)

    def test_roundtrip_via_tomllib(self):
        import tomllib
        data = {
            "schema_version": "1.4",
            "task": {"name": "bioforge/x"},
            "metadata": {"k": "v", "n": 3},
            "environment": {"cpus": 1},
            "steps": [{"name": "s1"}, {"name": "s2"}],
        }
        text = dump_toml(data)
        back = tomllib.loads(text)
        self.assertEqual(back["task"]["name"], "bioforge/x")
        self.assertEqual(back["metadata"], {"k": "v", "n": 3})
        self.assertEqual([s["name"] for s in back["steps"]], ["s1", "s2"])


class TestEmitter(unittest.TestCase):
    def _emit(self, bp):
        d = Path(tempfile.mkdtemp())
        rep = emit_task(bp, d)
        return d, rep

    def test_good_blueprint_emits_all_files(self):
        d, rep = self._emit(good_blueprint())
        self.assertEqual(rep.status, "ok")
        self.assertTrue((d / "task.toml").is_file())
        self.assertTrue((d / "instruction.md").is_file())
        self.assertTrue((d / "environment" / "Dockerfile").is_file())
        self.assertTrue((d / "tests" / "test.sh").is_file())
        self.assertTrue((d / "solution" / "solve.sh").is_file())
        self.assertEqual(rep.lost_semantics, [])

    def test_env_files_land_under_environment(self):
        d, rep = self._emit(good_blueprint(
            env_files={"environment/data/README.txt": "x"}))
        self.assertTrue((d / "environment" / "data" / "README.txt").is_file())
        self.assertFalse((d / "environment" / "environment").exists())

    def test_all_text_files_are_lf(self):
        d, _ = self._emit(good_blueprint())
        for f in d.rglob("*"):
            if f.is_file():
                self.assertNotIn(b"\r", f.read_bytes(), f"CRLF in {f}")

    def test_deterministic_output(self):
        d1, _ = self._emit(good_blueprint())
        d2, _ = self._emit(good_blueprint())
        for rel in ("task.toml", "instruction.md", "environment/Dockerfile"):
            self.assertEqual((d1 / rel).read_bytes(), (d2 / rel).read_bytes(), rel)

    def test_invalid_blueprint_fails_without_writing(self):
        d, rep = self._emit(good_blueprint(task_name="bad name"))
        self.assertEqual(rep.status, "failed")
        self.assertTrue(any("task_name" in e for e in rep.errors))
        self.assertFalse((d / "task.toml").exists())

    def test_path_escape_recorded_as_lost(self):
        d, rep = self._emit(good_blueprint(
            env_files={"../evil.txt": "x", "environment/ok.txt": "y"}))
        self.assertTrue(any("escape" in s or "rejected" in s for s in rep.lost_semantics))
        self.assertFalse((d.parent / "evil.txt").exists())
        self.assertEqual(rep.status, "warning")

    def test_name_mismatch_recorded_as_lost(self):
        bp = good_blueprint(task_toml={
            "schema_version": "1.4",
            "task": {"name": "other/stale"},
        })
        d, rep = self._emit(bp)
        self.assertTrue(any("task_name" in s for s in rep.lost_semantics))
        import tomllib
        body = tomllib.loads((d / "task.toml").read_text(encoding="utf-8"))
        # stale name must NOT silently win — blueprint name is authoritative
        self.assertEqual(body["task"]["name"], "bioforge/chipseq-peak-calling")

    def test_provenance_merged_into_metadata(self):
        d, _ = self._emit(good_blueprint())
        import tomllib
        body = tomllib.loads((d / "task.toml").read_text(encoding="utf-8"))
        self.assertEqual(body["metadata"]["source_paper_id"], "epmc-PMC1234567")
        self.assertEqual(body["metadata"]["source_pmcid"], "PMC1234567")

    def test_notes_become_review_md(self):
        d, _ = self._emit(good_blueprint(notes=["check GPU need", "shortcut risk"]))
        text = (d / "REVIEW.md").read_text(encoding="utf-8")
        self.assertIn("check GPU need", text)


if __name__ == "__main__":
    unittest.main()
