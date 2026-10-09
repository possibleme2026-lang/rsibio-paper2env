"""Aggregate test runner: stdlib-only, no pytest, no pip install.

Usage:  python tests/run_checks.py            # run everything
        python tests/run_checks.py curator    # only modules matching 'curator'

Adds ``src`` to sys.path so the package is importable without installation.
Exit code is non-zero if any test fails — this is what CI gates on.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))


def main(argv: list[str]) -> int:
    filters = argv[1:]
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT / "tests"), pattern="test_*.py")

    if filters:
        suite = _filter(suite, filters)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def _filter(suite: unittest.TestSuite, needles: list[str]) -> unittest.TestSuite:
    out = unittest.TestSuite()
    for test in _flatten(suite):
        tid = test.id()
        if any(n.lower() in tid.lower() for n in needles):
            out.addTest(test)
    return out


def _flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
