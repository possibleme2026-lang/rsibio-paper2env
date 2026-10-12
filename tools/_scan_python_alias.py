"""Detect tasks whose graders/solvers invoke a bare ``python`` that the image
does not provide.

biographx-rna-localization is FROM ubuntu:22.04 and installs only ``python3``
(apt has no ``python`` package there), yet solve.sh:187 runs ``python
/app/solve.py`` -> exit 127 ``python: command not found``. The verifier's
``test.sh`` then fails the same way, so BOTH the oracle and every harness see
reward 0 for a reason that has nothing to do with the task.

This is purely a *provisioning* mismatch: the image language is fixed by the
Dockerfile, and the scripts assume a Debian-style ``python`` symlink. The
scanner reports tasks where a script calls bare ``python`` and the Dockerfile
never establishes such a symlink.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else
            Path(__file__).resolve().parent.parent / "out/runs/tasks")


def has_python_symlink(dockerfile: str) -> bool:
    """True if the image makes a bare ``python`` resolvable."""
    if re.search(r"python-is-python3", dockerfile):
        return True
    if re.search(r"ln\s+-s[^\n]*python3[^\n]*\bpython\b", dockerfile):
        return True
    # a python:X image already ships ``python``
    if re.search(r"^FROM\s+python:", dockerfile, re.M):
        return True
    if re.search(r"update-alternatives[^\n]*python", dockerfile):
        return True
    # pip installing a python binary is not a thing; ignore
    return False


def main() -> None:
    rows = []
    for d in sorted(ROOT.iterdir()):
        df = d / "environment" / "Dockerfile"
        if not df.is_file():
            continue
        dockerfile = df.read_text(encoding="utf-8", errors="replace")
        if has_python_symlink(dockerfile):
            continue
        callers = []
        for sub in ("solution/solve.sh", "tests/test.sh"):
            p = d / sub
            if not p.is_file():
                continue
            t = p.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"(?<![\w./-])python\s+([^\s|;&]+)", t):
                arg = m.group(1)
                if arg in ("-c", "-m", "-u", "-"):  # heredoc / module / inline
                    continue
                callers.append(f"{sub}:{t[:m.start()].count(chr(10))+1} python {arg}")
        if callers:
            rows.append((d.name, callers, dockerfile.splitlines()[0]))

    print(f"{len(rows)} tasks call bare ``python`` but the image may not have it\n")
    for name, callers, base in rows:
        print(f"===== {name}  ({base})")
        for c in callers[:6]:
            print("   ", c)


if __name__ == "__main__":
    main()
