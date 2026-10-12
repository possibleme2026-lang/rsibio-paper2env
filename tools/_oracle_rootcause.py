"""Aggregate the *root* failure of every oracle trial: what the oracle agent
itself printed before the verifier ran.

The verifier output is often a downstream symptom ("score 0.0"), while the
actionable cause is in ``agent/oracle.txt`` (the oracle's own solve.sh run)
or ``agent/exit-code.txt``. This prints, per failed task, the last few
non-warning lines of the agent log so the failure taxonomy is built from
evidence rather than from the reward number.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "out" / "task_validation"


def last_meaningful(path: Path, n: int = 6) -> list[str]:
    if not path.is_file():
        return []
    txt = path.read_text(encoding="utf-8", errors="replace")
    lines = [l.rstrip() for l in txt.splitlines()
             if l.strip() and "warn(" not in l and "Warning" not in l
             and not l.lstrip().startswith("warnings.warn")]
    return lines[-n:]


def main() -> None:
    rows = []
    for tdir in sorted(ROOT.iterdir()):
        if not tdir.is_dir():
            continue
        runs = sorted([d for d in tdir.iterdir() if d.is_dir()], reverse=True)
        if not runs:
            continue
        trials = [d for d in runs[0].iterdir() if d.is_dir()]
        if not trials:
            continue
        tr = trials[0]
        rj = tr / "result.json"
        if not rj.is_file():
            continue
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        rewards = (data.get("verifier_result") or {}).get("rewards") or {}
        score = rewards.get("score")
        if score is None:
            continue
        if isinstance(score, (int, float)) and float(score) >= 0.999:
            continue
        exc = (data.get("exception_info") or {}).get("exception_type")
        agent_lines = last_meaningful(tr / "agent" / "oracle.txt")
        ec = (tr / "agent" / "exit-code.txt")
        ecv = ec.read_text(encoding="utf-8", errors="replace").strip() if ec.is_file() else "?"
        ge = tr / "verifier" / "grader_error.txt"
        glast = ge.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1] if ge.is_file() else None
        rows.append((tdir.name, score, ecv, exc, glast, agent_lines))

    print(f"{len(rows)} tasks with score < 1.0 (from latest run)\n")
    for name, score, ecv, exc, glast, agent_lines in rows:
        print(f"===== {name}  score={score}  oracle_exit={ecv}  exc={exc}")
        if glast:
            print(f"  [grader_error] {glast}")
        for l in agent_lines:
            print(f"  | {l[:200]}")


if __name__ == "__main__":
    main()
