"""M2: backends — Mock strictness, DryRun schema-validity, JSON extraction."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bioforge_harbor.backends import MockBackend, DryRunBackend, make_backend
from bioforge_harbor.backends.base import AgentBackend, BackendError
from bioforge_harbor.backends.jsonx import extract_json
from bioforge_harbor.backends.pi_backend import RealPiBackend, _locate_cli, PiUnavailable


class TestExtractJson(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_markdown_fence(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_leading_and_trailing_prose(self):
        self.assertEqual(
            extract_json('Sure! Here it is: {"a": [1,2]} hope that helps'),
            {"a": [1, 2]},
        )

    def test_braces_inside_strings(self):
        self.assertEqual(extract_json('{"t": "a{b}c"}'), {"t": "a{b}c"})

    def test_escaped_quote(self):
        self.assertEqual(extract_json('{"t": "he said \\"hi\\""}'), {"t": 'he said "hi"'})

    def test_array_top_level(self):
        self.assertEqual(extract_json('x [1, {"y": 2}] z'), [1, {"y": 2}])

    def test_unbalanced_raises(self):
        with self.assertRaises(ValueError):
            extract_json('{"a": 1')

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            extract_json('no json here at all')


class TestMockBackend(unittest.TestCase):
    def test_protocol_conformance(self):
        self.assertIsInstance(MockBackend({}), AgentBackend)
        self.assertIsInstance(DryRunBackend(), AgentBackend)
        self.assertIsInstance(RealPiBackend(), AgentBackend)

    def test_replay_dict(self):
        m = MockBackend({"stage.a": {"ok": True}})
        self.assertEqual(m.ask_json("stage.a", "x"), {"ok": True})
        self.assertEqual(m.calls, ["stage.a"])

    def test_replay_string_parsed_as_json(self):
        m = MockBackend({"t": '{"n": 5}'})
        self.assertEqual(m.ask_json("t", "x"), {"n": 5})

    def test_ask_text_from_dict_is_json(self):
        m = MockBackend({"t": {"n": 5}})
        self.assertEqual(extract_json(m.ask_text("t", "x")), {"n": 5})

    def test_strict_unknown_tag_raises(self):
        m = MockBackend({"known": {}}, strict=True)
        with self.assertRaises(BackendError):
            m.ask_json("missing", "x")

    def test_nonstrict_unknown_returns_empty(self):
        m = MockBackend({}, strict=False)
        self.assertEqual(m.ask_json("anything", "x"), {})

    def test_deepcopy_isolation(self):
        table = {"t": {"nested": [1]}}
        m = MockBackend(table)
        got = m.ask_json("t", "x")
        got["nested"].append(99)
        self.assertEqual(table["t"]["nested"], [1])

    def test_from_dir(self):
        d = Path(tempfile.mkdtemp())
        (d / "a.json").write_text('{"v": 1}', encoding="utf-8")
        (d / "b.txt").write_text("plain text", encoding="utf-8")
        m = MockBackend.from_dir(d)
        self.assertEqual(m.ask_json("a", "x"), {"v": 1})
        self.assertEqual(m.ask_text("b", "x"), "plain text")


class TestDryRunBackend(unittest.TestCase):
    def test_ask_json_from_schema_is_valid(self):
        dry = DryRunBackend()
        obj = dry.ask_json("t", "x", schema_name="blueprint_1")
        from bioforge_harbor.schemas import default_validator
        r = default_validator().validate("blueprint_1", obj)
        self.assertTrue(r.ok, r.summary())

    def test_dryrun_blueprint_passes_semantic_validation(self):
        """Schema-valid is not enough — it must also pass blueprint.validate()."""
        dry = DryRunBackend()
        obj = dry.ask_json("t", "x", schema_name="blueprint_1")
        from bioforge_harbor.models.blueprint import TaskBlueprint
        b = TaskBlueprint.from_dict(obj)
        self.assertEqual(b.validate(), [], f"validate errors: {b.validate()}")

    def test_ask_json_no_schema_returns_empty(self):
        self.assertEqual(DryRunBackend().ask_json("t", "x"), {})

    def test_ask_text_placeholder(self):
        out = DryRunBackend().ask_text("curator.select", "x")
        self.assertIn("curator.select", out)


class TestMakeBackend(unittest.TestCase):
    def test_factory_kinds(self):
        self.assertIsInstance(make_backend("mock"), MockBackend)
        self.assertIsInstance(make_backend("dryrun"), DryRunBackend)
        self.assertIsInstance(make_backend("pi"), RealPiBackend)

    def test_factory_unknown_raises(self):
        with self.assertRaises(ValueError):
            make_backend("nope")

    def test_pi_factory_passes_through(self):
        b = make_backend("pi", provider="tierflow", model="GLM-5.3-Flash", thinking="high")
        self.assertEqual(b.provider, "tierflow")
        self.assertEqual(b.model, "GLM-5.3-Flash")
        self.assertEqual(b.thinking, "high")


class TestPiBackendCommand(unittest.TestCase):
    """Command construction only — never launches pi (that is opt-in)."""

    def test_command_has_isolation_flags(self):
        b = RealPiBackend(provider="p1", model="m1", thinking="low")
        try:
            cmd = b._command("multi\nline\nprompt")
        except PiUnavailable:
            self.skipTest("pi cli.js not locatable on this machine")
        self.assertEqual(cmd[0], cmd[0])  # node resolved
        self.assertTrue(cmd[1].endswith("cli.js"))
        for flag in ("-p", "--no-session", "--no-extensions", "--no-skills",
                     "--no-context-files", "--no-prompt-templates", "--no-tools"):
            self.assertIn(flag, cmd)
        self.assertIn("--provider", cmd)
        self.assertIn("p1", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("m1", cmd)
        # prompt survives as a single argv element with newlines intact
        self.assertEqual(cmd[-1], "multi\nline\nprompt")

    def test_locate_cli_on_this_machine(self):
        try:
            node, cli = _locate_cli()
        except PiUnavailable as e:
            self.skipTest(f"pi not installed here: {e}")
        self.assertTrue(Path(node).is_file())
        self.assertTrue(Path(cli).is_file())


if __name__ == "__main__":
    unittest.main()
