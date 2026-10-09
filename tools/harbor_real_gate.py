#!/usr/bin/env python3
"""Optional real-Harbor gate: cross-validate the golden fixture with harbor.

Behavior contract (this is the whole point of the tool):

- harbor importable  -> run TaskConfig.model_validate_toml + Task(dir)
                        + Task.is_valid_dir on the golden fixture.
                        PASS -> exit 0. FAIL -> exit 1 (a real regression).
- harbor NOT importable -> print "UNVERIFIED: <reason>" and exit 0.

Exit 0 on UNVERIFIED is deliberate: the *required* CI job is the strict-schema
gate (offline, always meaningful). This job is a bonus cross-check that must
never red-block CI just because the PyPI 'harbor' wheel is absent or is a
different project with the same name. A human reading CI sees UNVERIFIED and
knows the authoritative check did not run here.

Usage:  python tools/harbor_real_gate.py [task_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK = ROOT / "tests" / "fixtures" / "golden_task"


def main(argv: list[str]) -> int:
    task_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_TASK
    toml_path = task_dir / "task.toml"
    if not toml_path.is_file():
        print(f"UNVERIFIED: no task.toml at {task_dir}")
        return 0
    toml_text = toml_path.read_text(encoding="utf-8")

    try:
        from harbor.models.task.config import TaskConfig
        from harbor.models.task.task import Task
    except Exception as e:  # noqa: BLE001 - the whole point is to degrade
        print(f"UNVERIFIED: harbor not importable ({type(e).__name__}: {e}). "
              f"Set HARBOR_INSTALL_SPEC to the correct package/URL.")
        return 0

    try:
        TaskConfig.model_validate_toml(toml_text)
        Task(str(task_dir))
        if hasattr(Task, "is_valid_dir"):
            Task.is_valid_dir(str(task_dir))
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: harbor rejected {task_dir}: {type(e).__name__}: {e}")
        return 1

    print(f"PASS: harbor accepted {task_dir} (TaskConfig + Task)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
