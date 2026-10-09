"""Guard: the claim registry stays honest.

A paper about measured numbers is only as trustworthy as its weakest
un-labelled number. So every claim that may appear in the manuscript is
registered in ``claims.json`` *before* it is written down, and this guard
enforces three rules that are easy to break under deadline pressure:

1. ``measured`` requires evidence that exists on disk. A claim with no test,
   fixture or script behind it cannot be marked measured.
2. comparative / rate claims must name their control group. "Auto-built
   environments outperform random ones" is meaningless without saying what
   the random arm is.
3. control-group-first, by language: if a statement uses an improvement word
   ("outperform", "exceed", "improves", ...) it must carry a control even if
   whoever wrote it filed it under ``structural``. This is the tripwire for a
   claim that quietly grew an assertion while staying un-controlled.

Usage:  python tools/check_claims.py [--claims claims.json] [--paper FILE]
Exit:   0 = clean, 1 = at least one violation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

STATUSES = ("measured", "planned", "unverifiable")
KINDS = ("structural", "rate", "comparative")
REQUIRED = ("id", "statement", "kind", "status", "metric", "control", "evidence")

# If a statement says something got better, it needs a control arm.
# ASCII entries are matched on word boundaries — substring matching turns
# "against" into "gain" and produces a false positive on the first real file.
# CJK entries have no boundaries to anchor to, so they stay substring matches.
IMPROVEMENT_WORDS = (
    "improv", "outperform", "exceed", "surpass", "higher", "better", "faster",
    "increase", "reduce", "lift", "beat", "gain", "superior",
    "提升", "超过", "优于", "高于", "更快", "降低", "改善",
)

CLAIM_REF = re.compile(r"\b[CP]\d+\b")


def _improvement_word(statement: str) -> str | None:
    low = statement.lower()
    for w in IMPROVEMENT_WORDS:
        if w.isascii():
            if re.search(r"\b" + re.escape(w) + r"\b", low):
                return w
        elif w in low:
            return w
    return None


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(data: dict, *, root: Path, paper: Path | None = None) -> list[str]:
    """Return a list of violations (empty == clean)."""
    problems: list[str] = []
    claims = data.get("claims")
    if not isinstance(claims, list) or not claims:
        return ["claims.json has no 'claims' array"]

    seen: set[str] = set()
    for i, c in enumerate(claims):
        cid = c.get("id", f"<#{i}>")
        for field in REQUIRED:
            if field not in c:
                problems.append(f"{cid}: missing field '{field}'")
        if cid in seen:
            problems.append(f"{cid}: duplicate claim id")
        seen.add(cid)

        kind = c.get("kind")
        status = c.get("status")
        if kind not in KINDS:
            problems.append(f"{cid}: kind {kind!r} not in {KINDS}")
        if status not in STATUSES:
            problems.append(f"{cid}: status {status!r} not in {STATUSES}")

        # rule 1: measured needs evidence that exists
        if status == "measured":
            ev = c.get("evidence") or []
            if not ev:
                problems.append(f"{cid}: status=measured but no evidence listed")
            for e in ev:
                if not (root / e).exists():
                    problems.append(f"{cid}: evidence not found on disk: {e}")

        # rule 2: comparative and rate claims need a control and a metric
        if kind in ("comparative", "rate"):
            if not (c.get("control") or "").strip():
                problems.append(f"{cid}: kind={kind} must name a control group")
            if not (c.get("metric") or "").strip():
                problems.append(f"{cid}: kind={kind} must name its metric")

        # rule 3: control-group-first, judged from the wording
        if not (c.get("control") or "").strip():
            hit = _improvement_word(c.get("statement") or "")
            if hit and kind == "structural":
                problems.append(
                    f"{cid}: statement uses {hit!r} but declares no control "
                    f"— re-file as comparative/rate or drop the assertion")

    # every claim referenced by the manuscript must exist here
    if paper is not None and paper.is_file():
        text = paper.read_text(encoding="utf-8")
        for ref in sorted(set(CLAIM_REF.findall(text))):
            if ref not in seen:
                problems.append(f"paper references {ref}, which is not in claims.json")

    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="check_claims",
                                 description="enforce the claims registry")
    ap.add_argument("--claims", default=str(ROOT / "claims.json"))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--paper", default=None,
                    help="scan a manuscript for claim ids that must be registered")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    claims_path = Path(a.claims)
    if not claims_path.is_file():
        print(f"FAIL: no claims file at {claims_path}")
        return 1
    data = load(claims_path)
    problems = check(data, root=Path(a.root),
                     paper=Path(a.paper) if a.paper else None)

    counts: dict[str, int] = {}
    for c in data.get("claims", []):
        counts[c.get("status", "?")] = counts.get(c.get("status", "?"), 0) + 1

    if a.json:
        print(json.dumps({"n_claims": len(data.get("claims", [])), "counts": counts,
                          "problems": problems}, indent=2))
    else:
        print(f"claims: {len(data.get('claims', []))} "
              f"({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})")
        if problems:
            for p in problems:
                print(f"  FAIL {p}")
            print(f"{len(problems)} violation(s)")
        else:
            print("OK")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
