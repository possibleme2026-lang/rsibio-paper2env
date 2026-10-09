"""M1: strict schema validator — must catch the silent-swallow trap."""

from __future__ import annotations

import unittest

from bioforge_harbor.schemas import default_validator, validate_against


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.v = default_validator()

    def test_good_task_toml_passes(self):
        good = {
            "schema_version": "1.4",
            "task": {"name": "bioforge/hi", "version": "1.0.0", "keywords": ["a"]},
            "agent": {"timeout_sec": 120.0},
            "verifier": {"timeout_sec": 60.0, "env": {"K": "V"}},
            "environment": {"cpus": 2, "memory_mb": 2048, "gpus": 0, "os": "linux"},
            "solution": {"env": {}},
        }
        r = self.v.validate("task_toml_1_4", good)
        self.assertTrue(r.ok, r.summary())

    def test_unknown_key_is_caught(self):
        # 'cpu' instead of 'cpus' — pydantic extra='ignore' would silently drop it.
        bad = {"schema_version": "1.4", "task": {"name": "bioforge/hi"},
               "environment": {"cpu": 2}}
        r = self.v.validate("task_toml_1_4", bad)
        self.assertFalse(r.ok)
        self.assertIn("cpu", r.summary())

    def test_bad_name_format_caught(self):
        for name in ["nope", "bioforge/", "/hi", "has space/x"]:
            r = self.v.validate("task_toml_1_4", {"task": {"name": name}})
            self.assertFalse(r.ok, f"expected reject for {name!r}")

    def test_bad_enum_caught(self):
        r = self.v.validate("task_toml_1_4",
                            {"task": {"name": "bioforge/hi"},
                             "environment": {"os": "macos"}})
        self.assertFalse(r.ok)

    def test_ref_resolution_nested(self):
        # verifier.environment uses $ref to the environment definition.
        r = self.v.validate("task_toml_1_4",
                            {"task": {"name": "bioforge/hi"},
                             "verifier": {"environment": {"bogus_key": 1}}})
        self.assertFalse(r.ok)
        self.assertIn("bogus_key", r.summary())

    def test_type_check_number_vs_string(self):
        r = self.v.validate("task_toml_1_4",
                            {"task": {"name": "bioforge/hi"},
                             "agent": {"timeout_sec": "fast"}})
        self.assertFalse(r.ok)

    def test_artifacts_string_or_object(self):
        good = {"task": {"name": "bioforge/hi"},
                "artifacts": ["out.log", {"source": "/app/x", "destination": "y"}]}
        self.assertTrue(self.v.validate("task_toml_1_4", good).ok)
        bad = {"task": {"name": "bioforge/hi"}, "artifacts": [{"nope": 1}]}
        self.assertFalse(self.v.validate("task_toml_1_4", bad).ok)

    def test_metadata_is_free_form(self):
        good = {"task": {"name": "bioforge/hi"},
                "metadata": {"anything": [1, 2, {"x": "y"}]}}
        self.assertTrue(self.v.validate("task_toml_1_4", good).ok)

    def test_blueprint_schema(self):
        good = {
            "task_name": "bioforge/x",
            "instruction_md": "do it",
            "dockerfile": "FROM python:3.11",
            "test_files": {"tests/test.sh": "#!/bin/bash"},
            "provenance": {"paper_id": "epmc-PMC1"},
        }
        self.assertTrue(self.v.validate("blueprint_1", good).ok)
        missing = dict(good); missing.pop("provenance")
        self.assertFalse(self.v.validate("blueprint_1", missing).ok)

    def test_report_envelope_status_enum(self):
        self.assertTrue(self.v.validate("stage_report_1",
                                        {"stage": "curate", "status": "ok"}).ok)
        self.assertFalse(self.v.validate("stage_report_1",
                                         {"stage": "curate", "status": "great"}).ok)


if __name__ == "__main__":
    unittest.main()
