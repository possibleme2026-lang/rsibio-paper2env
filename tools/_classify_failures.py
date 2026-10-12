"""Decide, per failing task: stale trial (fix landed after the run) or new defect.

The oracle sweep caches only ``status == ok`` results, so any failure is retried
on the next pass. That means a failure whose task files were *modified after*
the trial ran is already resolved and only needs a re-run; a failure whose task
files are untouched is a genuinely new defect that needs a new fixer. Getting
this classification right is the difference between writing 3 fixers and 20.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VAL = REPO / "out" / "task_validation"
TASKS = REPO / "out" / "runs" / "tasks"

fails = [l.strip() for l in (sys.argv[1:] or []) if l.strip()]
if not fails:
    ok = set()
    p = VAL / "oracle_results.jsonl"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                ok.add(json.loads(line)["task"])
            except Exception:
                pass
    fails = sorted(d.name for d in TASKS.iterdir()
                   if (d / "task.toml").is_file() and d.name not in ok)


def newest_trial(t: str) -> Path | None:
    d = VAL / t
    if not d.is_dir():
        return None
    runs = [p for p in d.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def trial_wall(run: Path) -> float:
    """Newest mtime of any file the trial produced."""
    newest = run.stat().st_mtime
    for p in run.rglob("*"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            pass
    return newest


def task_edited_after(t: str, wall: float) -> tuple[bool, list[str]]:
    """Did any source file of the task change after the trial ran?"""
    d = TASKS / t
    hits: list[str] = []
    for sub in ("solution", "tests", "environment"):
        base = d / sub
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            try:
                if p.stat().st_mtime > wall:
                    hits.append(str(p.relative_to(d)))
            except OSError:
                pass
    return bool(hits), hits


print(f"{'task':<52} {'verdict':<18} evidence")
print("-" * 120)
stale, new = [], []
for t in fails:
    run = newest_trial(t)
    if run is None:
        print(f"{t:<52} {'NO-TRIAL':<18} never ran")
        new.append(t)
        continue
    wall = trial_wall(run)
    changed, hits = task_edited_after(t, wall)
    if changed:
        stale.append(t)
        print(f"{t:<52} {'STALE-FIXED':<18} "
              f"{len(hits)} file(s) edited after run: {', '.join(hits[:3])}")
    else:
        new.append(t)
        print(f"{t:<52} {'NEW-DEFECT':<18} no task edit since run")
print("-" * 120)
print(f"stale (just re-run): {len(stale)}   new defect: {len(new)}")
print("\nNEW:", " ".join(new))
