"""Guard: the two READMEs keep the same shape.

A bilingual README pair drifts the moment one language gets a section the other
does not, and nothing in a compiler or a test catches it — only a reader who
happen to open both. This guard compares the parts that must be identical
regardless of language: the heading *outline* (level sequence), the number of
code blocks, and the set of external links. Prose differs by definition; the
skeleton must not.

Usage:  python tools/check_readme_i18n.py [--root .]
Exit:   0 = clean, 1 = structural drift.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EN = "README.md"
ZH = "README.zh-CN.md"

HEADING = re.compile(r"^(#{1,6})\s+\S")
FENCE = re.compile(r"^\s{0,3}(```|~~~)")
URL = re.compile(r"https?://[^\s)>\]\"']+")


def outline(text: str) -> list[int]:
    """Heading levels in document order: the language-independent skeleton."""
    levels: list[int] = []
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING.match(line)
        if m:
            levels.append(len(m.group(1)))
    return levels


def fence_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if FENCE.match(line))


def links(text: str) -> set[str]:
    return {u.rstrip(".,;") for u in URL.findall(text)}


def check(path_a: Path, path_b: Path) -> list[str]:
    problems: list[str] = []
    for p in (path_a, path_b):
        if not p.is_file():
            problems.append(f"missing {p.name}")
    if problems:
        return problems

    ta = path_a.read_text(encoding="utf-8")
    tb = path_b.read_text(encoding="utf-8")
    oa, ob = outline(ta), outline(tb)

    if not oa:
        problems.append(f"{path_a.name}: no headings found")
    if oa != ob:
        problems.append(f"heading outline differs: {path_a.name}={oa} {path_b.name}={ob}")

    fa, fb = fence_count(ta), fence_count(tb)
    if fa != fb:
        problems.append(f"code block count differs: {path_a.name}={fa} {path_b.name}={fb}")

    la, lb = links(ta), links(tb)
    if la != lb:
        only_a = sorted(la - lb)
        only_b = sorted(lb - la)
        if only_a:
            problems.append(f"links only in {path_a.name}: {only_a}")
        if only_b:
            problems.append(f"links only in {path_b.name}: {only_b}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="check_readme_i18n",
                                 description="enforce README / README.zh-CN.md parity")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--a", default=EN)
    ap.add_argument("--b", default=ZH)
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    problems = check(root / a.a, root / a.b)
    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        print(f"{len(problems)} structural difference(s)")
    else:
        print(f"OK: {a.a} and {a.b} share one skeleton")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
