"""Find tasks whose reference solution is a *stub*, not a solution.

VERIFIED on kukulu-cdr-reconstruction: solve.sh's own comments say
"# For brevity, write a placeholder output (reference solution is
illustrative)" and it writes a file containing only ``END``. The grader then
crashes on the empty structure. No harness can ever score such a task, and
no mechanical repair can invent the missing algorithm -- these tasks must be
EXCLUDED from the eval set (or regenerated), not patched.

Detection is deliberately conservative: a task is flagged only when the
solution writes an obviously empty/trivial artifact OR carries an explicit
"placeholder"/"illustrative"/"for brevity"/"TODO" admission. Missing a stub
is safer than dropping a real task.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else
            Path(__file__).resolve().parent.parent / "out/runs/tasks")

ADMIT = re.compile(
    r"(placeholder|illustrative|for brevity|simplified stub|not implemented|"
    r"\bTODO\b|FIXME|dummy output|write a stub|stub output)", re.I)

# writing an artifact whose entire content is a single trivial token
TRIVIAL_WRITE = re.compile(
    r"""write\(\s*(['"])(END|TODO|placeholder|stub|0|\{\}|\[\])\1\s*\)|"""
    r"""write_text\(\s*(['"])(END|TODO|placeholder|stub|\{\}|\[\])\3\s*\)|"""
    r"""f\.write\(\s*(['"])(END|\{\}|\[\]|0)\5\s*\)""", re.I)


def scan_script(p: Path) -> list[str]:
    t = p.read_text(encoding="utf-8", errors="replace")
    hits = []
    for i, line in enumerate(t.splitlines(), 1):
        if ADMIT.search(line):
            hits.append(f"{p.name}:{i}  {line.strip()[:110]}")
        elif TRIVIAL_WRITE.search(line):
            hits.append(f"{p.name}:{i}  {line.strip()[:110]}")
    return hits


def main() -> None:
    rows = []
    for d in sorted(ROOT.iterdir()):
        sol = d / "solution"
        if not sol.is_dir():
            continue
        hits: list[str] = []
        for f in sol.rglob("*"):
            if f.is_file() and f.suffix in (".sh", ".py"):
                hits += scan_script(f)
        if hits:
            rows.append((d.name, hits))
    print(f"{len(rows)} tasks whose reference solution looks like a stub\n")
    for name, hits in rows:
        print(f"===== {name}")
        for h in hits[:4]:
            print("   ", h)


if __name__ == "__main__":
    main()
