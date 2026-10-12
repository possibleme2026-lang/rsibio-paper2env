"""Scan for the ``float(<array expr>).<method>()`` precedence bug.

VERIFIED on hpabc-ml-directed-evolution tests/grade.py:49:

    return float(math.sqrt(float((np.log(p) - np.log(g)) ** 2).mean()))

``float(...)`` wraps a *vector* expression and then ``.mean()`` is called on
the resulting scalar -> ``TypeError: only length-1 arrays can be converted to
Python scalars``. The grader dies on its very first RMSE, so the task scores 0
for every harness regardless of the answer. The intended code is

    return float(math.sqrt(((np.log(p) - np.log(g)) ** 2).mean()))

i.e. ``float`` must wrap the scalar *after* ``.mean()``, not the vector.

This scanner finds ``float(`` immediately followed by an expression that
contains a ``**`` / arithmetic on numpy-array-looking names and then a
``).<method>`` -- the tell-tale misplaced parenthesis.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else
            Path(__file__).resolve().parent.parent / "out/runs/tasks")

# float( ... ** ... ) . <method> (   -- the misplaced parenthesis
PAT = re.compile(
    r"float\([^;\n]*\*\*[^;\n]*\)\s*\.\s*(mean|sum|max|min|std|var|dot)\s*\(")


def main() -> None:
    n = 0
    for d in sorted(ROOT.iterdir()):
        tests = d / "tests"
        if not tests.is_dir():
            continue
        for f in list(tests.glob("*.py")) + list((d / "solution").glob("*.py")) \
                + list((d / "solution").glob("*.sh")):
            if not f.is_file():
                continue
            t = f.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(t.splitlines(), 1):
                if PAT.search(line):
                    n += 1
                    print(f"{d.name}/{f.name}:{i}")
                    print(f"    {line.strip()[:150]}")
    print(f"\n=> {n} hit(s)")


if __name__ == "__main__":
    main()
