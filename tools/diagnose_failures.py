"""Summarize why each oracle trial failed, from the trial's own artifacts.

The harbor console only prints "RuntimeError"; the actionable line lives in
``<trial>/verifier/test-stdout.txt`` (grader traceback) or
``<trial>/exception.txt`` (infrastructure failure). This walks
``out/task_validation/<task>/<ts>/<trial>/`` and prints one classified line per
failing task, so a batch of failures can be triaged without opening each by hand.

Classification:
  REWARD_MISSING  -- grader never wrote /logs/verifier/reward.json
  GRADER_ERROR    -- grader raised (pydantic/json/type/key errors etc.)
  BUILD_FAIL      -- image build did not complete
  SCORE_ZERO      -- grader ran clean but the oracle solution scored 0
  OTHER           -- anything not matched above
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VAL = REPO / "out" / "task_validation"


def _last_error(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.search(r"(Error|Exception|error:|Traceback|failed to solve|"
                     r"has no installation candidate|E: |score=0|Passed 0/)", ln):
            return ln.strip()[:200]
    return lines[-1].strip()[:200] if lines else ""


def classify(task_dir: Path) -> tuple[str, str]:
    trials = sorted(task_dir.rglob("exception.txt"),
                    key=lambda p: p.stat().st_mtime)
    # newest trial dir
    ts_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir()],
                     key=lambda d: d.stat().st_mtime)
    if not ts_dirs:
        return "OTHER", "no trial dir"
    latest = ts_dirs[-1]
    tsos = list(latest.rglob("test-stdout.txt"))
    exc_files = list(latest.rglob("exception.txt"))

    err = ""
    if exc_files:
        err = _last_error(exc_files[-1].read_text(encoding="utf-8",
                                                  errors="replace"))
    if tsos:
        body = tsos[-1].read_text(encoding="utf-8", errors="replace")
        gerr = _last_error(body)
        if gerr and ("Traceback" in body or "Error" in gerr):
            err = gerr or err
        low = body.lower()
        if "no reward file" in low or "rewardfilenotfound" in low:
            return "REWARD_MISSING", err
        if "traceback" in low and "sanitized reward.json" in low:
            return "GRADER_ERROR", err
        if "traceback" in low:
            return "GRADER_ERROR", err
        if re.search(r"Passed 0/\d+", body) or "TOTAL SCORE = 0.0" in body:
            return "SCORE_ZERO", err
    if err:
        if "failed to solve" in err or "did not complete" in err:
            return "BUILD_FAIL", err
        if "no reward file" in err.lower() or "RewardFileNotFound" in err:
            return "REWARD_MISSING", err
    return "OTHER", err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-failures", action="store_true")
    a = ap.parse_args()

    ok_tasks: set[str] = set()
    jl = VAL / "oracle_results.jsonl"
    if jl.is_file():
        import json
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") == "ok":
                ok_tasks.add(rec.get("task", ""))

    buckets: Counter = Counter()
    rows = []
    for d in sorted(VAL.iterdir()):
        if not d.is_dir():
            continue
        if d.name in ok_tasks:
            continue
        # not yet swept this run: no trial directory exists -- don't count
        # pending work as failure
        if not any(d.iterdir()):
            continue
        kind, err = classify(d)
        if kind == "OTHER" and err == "no trial dir":
            continue
        buckets[kind] += 1
        rows.append((kind, d.name, err))

    for kind, name, err in sorted(rows):
        print(f"[{kind:14}] {name}\n{'':16}{err}")
    print("\n=== buckets ===")
    for k, n in buckets.most_common():
        print(f"  {k:14} {n}")
    print(f"  {'TOTAL FAIL':14} {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
