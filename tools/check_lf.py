"""Guard: no CRLF and no BOM in tracked text files.

Line endings are the classic silent breakage in this repo: a task directory is
emitted with LF, shipped into a Linux container, and one stray CRLF turns a
``tests/test.sh`` into a syntax error or a here-doc into garbage. ``.gitattributes``
and the emitters already write LF; this guard is the last line that says so out
loud, and it fails on Windows too (where CRLF is the default).

Usage:  python tools/check_lf.py [--root .]
Exit:   0 = clean, 1 = at least one offending file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".bz2",
    ".xz", ".whl", ".so", ".dll", ".exe", ".pyd", ".pyc", ".woff", ".woff2",
    ".ttf", ".otf", ".mp4", ".db", ".sqlite3",
}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "build", "dist", "out",
             ".mypy_cache", ".pytest_cache", "runs"}


def tracked_files(root: Path) -> list[Path]:
    """Tracked *and* untracked-but-not-ignored files.

    Untracked matters: a file written minutes ago and not yet committed is
    exactly the one that can carry a stray CRLF into the repo, and a guard
    that only inspects the index would wave it through. Ignored paths (runs/,
    out/, build/) are skipped on purpose — they are generated output.
    """
    try:
        rel: set[str] = set()
        for args in (["ls-files"], ["ls-files", "--others", "--exclude-standard"]):
            out = subprocess.run(["git"] + args, cwd=str(root),
                                 capture_output=True, text=True, check=True)
            rel.update(line.strip() for line in out.stdout.splitlines() if line.strip())
        return [root / r for r in sorted(rel)]
    except Exception:  # noqa: BLE001 - git absent or not a repo: walk instead
        files = []
        for p in root.rglob("*"):
            if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
                files.append(p)
        return files


def check(root: Path) -> list[str]:
    problems: list[str] = []
    for path in tracked_files(root):
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        if b"\x00" in raw[:4096]:          # binary sniff
            continue
        rel = path.relative_to(root).as_posix()
        if raw.startswith(b"\xef\xbb\xbf"):
            problems.append(f"{rel}: starts with a UTF-8 BOM")
        if b"\r\n" in raw:
            first = raw.split(b"\r\n")[0][:60].decode("utf-8", "replace")
            n = raw.count(b"\r\n")
            problems.append(f"{rel}: {n} CRLF line ending(s), e.g. {first!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="check_lf",
                                 description="fail on CRLF or BOM in tracked files")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    problems = check(Path(a.root).resolve())
    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        print(f"{len(problems)} file(s) with line-ending problems")
    else:
        print("OK: every tracked text file is LF, no BOM")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
