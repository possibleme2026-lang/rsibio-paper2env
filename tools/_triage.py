"""Compact per-failure triage: newest trial's actionable error line.

Why: the sweep log only keeps a 90-char error snippet, and the *newest* trial
for a task may be older than the latest fix on disk. Distinguishing "task is
broken" from "this trial ran before the fix landed" is the whole game when
deciding what to repair vs. what to simply re-run.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VAL = REPO / "out" / "task_validation"
TASKS = REPO / "out" / "runs" / "tasks"

FAILS = [l.strip() for l in (sys.argv[1:] or []) if l.strip()]
if not FAILS:
    p = VAL / "oracle_results.jsonl"
    ok = set()
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                ok.add(json.loads(line)["task"])
            except Exception:
                pass
    FAILS = sorted(d.name for d in TASKS.iterdir()
                   if (d / "task.toml").is_file() and d.name not in ok)

PAT = re.compile(r"(Error|Exception|Traceback|error:|failed to solve|"
                 r"has no installation candidate|E: |grpcio|ModuleNotFound)")


def newest_trial(t: str) -> Path | None:
    d = VAL / t
    runs = sorted((p for p in d.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime) if d.is_dir() else []
    return runs[-1] if runs else None


def scan(run: Path) -> tuple[str, str]:
    """Return (source, last action-worthy line)."""
    cands = []
    for name in ("exception.txt", "agent/oracle.txt",
                 "verifier/test-stdout.txt", "agent/exit-code.txt"):
        f = run / name
        if f.is_file():
            cands.append((name, f))
    # trial dir is nested: <ts>/<task>__<id>/
    for trial in run.iterdir():
        if not trial.is_dir() or trial.name.startswith("."):
            continue
        for name in ("exception.txt", "agent/oracle.txt",
                     "verifier/test-stdout.txt", "agent/exit-code.txt"):
            f = trial / name
            if f.is_file():
                cands.append((trial.name + "/" + name, f))
        for f in trial.rglob("grader_error.txt"):
            cands.append((str(f.relative_to(trial)), f))
    best = ("(nothing)", "")
    for name, f in cands:
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        hits = [l for l in lines if PAT.search(l)]
        if hits:
            return name, hits[-1][:220]
        if lines:
            best = (name, lines[-1][:220])
    return best


print(f"{'task':<52} {'src':<26} detail")
print("-" * 130)
for t in FAILS:
    run = newest_trial(t)
    if run is None:
        print(f"{t:<52} {'-':<26} no trial dir")
        continue
    src, det = scan(run)
    print(f"{t:<52} {src[:26]:<26} {det}")
