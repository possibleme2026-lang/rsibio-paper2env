"""Find graders whose score path can silently swallow grader-crash exceptions.

A grader wrapped in ``try: ... except Exception: score = 0.0`` with no error
record anywhere in the except block is *indistinguishable* from "agent
produced a wrong answer" -- the oracle sees reward=0 either way, and every
harness comparison built on that task is invalid. We flag two things:

  1. TOP-LEVEL bare swallow: the whole scoring body sits in one try/except
     that maps *any* exception (including grader bugs) to score 0.
  2. Per-checker swallow: each sub-checker has its own except that returns 0
     for that component -- acceptable if the except *records* the error,
     dangerous if it is bare.

This reports, for every task grader:
  top_bare   : a top-level ``except Exception:`` with score=0 and no record
  top_bare_line : line number
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else
            Path(__file__).resolve().parent.parent / "out/runs/tasks")


def top_level_silent(t: str) -> list[int]:
    """Top-level try/except wrapping the whole scoring body, bare -> 0."""
    lines = t.splitlines()
    hits = []
    for i, l in enumerate(lines):
        if re.match(r"^\s*except(\s+Exception)?(\s+as\s+\w+)?\s*:", l):
            # collect the whole except block body (following lines with deeper indent)
            j = i + 1
            body = []
            base = re.match(r"^(\s*)", lines[i]).group(1)
            while j < len(lines):
                line = lines[j]
                if line.strip() == "":
                    body.append(line); j += 1; continue
                ind = re.match(r"^(\s*)", line).group(1)
                if len(ind) <= len(base):
                    break
                body.append(line); j += 1
            block = "\n".join(body)
            has_zero = re.search(r"(score|reward|result|total)\s*=\s*0(\.0)?", block)
            records = re.search(
                r"(error|err\b|exc|exception|traceback|repr|print|details|"
                r"report|message|str\()", block, re.I)
            if has_zero and not records:
                hits.append(i + 1)
    return hits


def main() -> None:
    print(f"{'task':60s} top_bare_silent")
    n_danger = 0
    for d in sorted(ROOT.iterdir()):
        g = d / "tests" / "grade.py"
        if not g.is_file():
            continue
        t = g.read_text(encoding="utf-8", errors="replace")
        hits = top_level_silent(t)
        if hits:
            n_danger += 1
            print(f"{d.name:60s} lines={hits}")
    print(f"\n=> {n_danger} graders with silent top-level swallow")


if __name__ == "__main__":
    main()
