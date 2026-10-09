"""M1: util.io LF/atomic guarantees + util.subproc deadlock-safe timeout."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bioforge_harbor.util.io import (  # noqa: E402
    write_text_lf, atomic_write_text, write_json, read_json, sha256_text, short_digest,
)
from bioforge_harbor.util.subproc import run_to_file  # noqa: E402


class TestIo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_write_text_lf_no_crlf(self):
        p = Path(self.tmp) / "a.txt"
        write_text_lf(p, "line1\nline2\n")
        raw = p.read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertEqual(raw, b"line1\nline2\n")

    def test_atomic_write_no_temp_leftover(self):
        p = Path(self.tmp) / "b.json"
        atomic_write_text(p, '{"x": 1}\n')
        self.assertEqual(json_loads(p.read_text()), {"x": 1})
        leftovers = [f for f in os.listdir(self.tmp) if f.startswith(".tmp-")]
        self.assertEqual(leftovers, [])

    def test_write_read_json_roundtrip(self):
        p = Path(self.tmp) / "c.json"
        write_json(p, {"a": [1, 2], "b": "中文"})
        self.assertEqual(read_json(p), {"a": [1, 2], "b": "中文"})
        self.assertNotIn(b"\r", p.read_bytes())

    def test_digests_stable(self):
        self.assertEqual(sha256_text("abc")[:8], "ba7816bf")
        self.assertEqual(len(short_digest("abc")), 12)


def json_loads(s):
    import json
    return json.loads(s)


class TestSubproc(unittest.TestCase):
    def test_simple_success(self):
        r = run_to_file([sys.executable, "-c", "print('hello')"], timeout_sec=30)
        self.assertTrue(r.ok)
        self.assertIn("hello", r.stdout)

    def test_nonzero_exit(self):
        r = run_to_file([sys.executable, "-c", "import sys; sys.exit(3)"], timeout_sec=30)
        self.assertEqual(r.returncode, 3)
        self.assertFalse(r.ok)

    def test_timeout_kills_tree(self):
        # Child spawns a grandchild then sleeps; a pipe-based capture would
        # deadlock here. File sink + tree kill must return within ~timeout.
        code = (
            "import subprocess,sys,time\n"
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
            "time.sleep(30)\n"
        )
        r = run_to_file([sys.executable, "-c", code], timeout_sec=2)
        self.assertTrue(r.timed_out)
        self.assertFalse(r.ok)

    def test_stdin_text(self):
        r = run_to_file([sys.executable, "-c",
                         "import sys; sys.stdout.write(sys.stdin.read().upper())"],
                        timeout_sec=30, stdin_text="abc")
        self.assertEqual(r.stdout.strip(), "ABC")


if __name__ == "__main__":
    unittest.main()
