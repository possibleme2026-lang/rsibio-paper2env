"""PaperState: per-paper resumable state machine.

Five stages, strictly ordered (paper section 3.2):

    curate -> construct -> reproduce -> review -> package

Invariants enforced here:
* A stage may only advance one step forward; skipping is a programming error.
* ``state.json`` is written atomically (tmp + os.replace) so a crash mid-write
  can never leave a half-written file that a resume would misread.
* A corrupt/unreadable state file is REFUSED (raises); the caller must pass
  ``force=True`` to start over. We never silently rebuild state, because that
  would let a partially-failed run look clean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..util.io import atomic_write_text, sha256_text

STAGES = ("curate", "construct", "reproduce", "review", "package")


class StateError(RuntimeError):
    pass


@dataclass
class Attempt:
    stage: str
    status: str                 # ok | failed | skipped
    ts: str = ""
    digest: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperState:
    paper_id: str
    stage: str = ""             # last COMPLETED stage ("" = none)
    attempts: dict[str, Attempt] = field(default_factory=dict)

    # -- queries ------------------------------------------------------------
    def completed(self) -> list[str]:
        return [s for s in STAGES if self.attempts.get(s) and self.attempts[s].status == "ok"]

    def next_stage(self) -> str | None:
        done = set(self.completed())
        for s in STAGES:
            if s not in done:
                return s
        return None

    def is_done(self) -> bool:
        return self.next_stage() is None

    def can_resume_into(self, stage: str) -> bool:
        """True if *stage* is runnable now (its predecessor is complete)."""
        if stage not in STAGES:
            raise StateError(f"unknown stage {stage!r}")
        idx = STAGES.index(stage)
        if idx == 0:
            return True
        prev = STAGES[idx - 1]
        return bool(self.attempts.get(prev) and self.attempts[prev].status in ("ok", "skipped"))

    # -- mutation -----------------------------------------------------------
    def advance(self, stage: str, *, digest: str = "", status: str = "ok",
                ts: str = "", error: str = "") -> None:
        if stage not in STAGES:
            raise StateError(f"unknown stage {stage!r}")
        if status == "ok" and not self.can_resume_into(stage):
            raise StateError(
                f"cannot advance to {stage!r}: predecessor not complete "
                f"(completed={self.completed()})"
            )
        self.attempts[stage] = Attempt(stage=stage, status=status, ts=ts,
                                       digest=digest, error=error)
        if status == "ok":
            self.stage = stage

    def to_dict(self) -> dict[str, Any]:
        return {"paper_id": self.paper_id, "stage": self.stage,
                "attempts": {k: v.to_dict() for k, v in self.attempts.items()}}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaperState":
        st = cls(paper_id=d["paper_id"], stage=d.get("stage", ""))
        for k, v in (d.get("attempts") or {}).items():
            st.attempts[k] = Attempt(**{kk: vv for kk, vv in v.items()
                                        if kk in Attempt.__dataclass_fields__})
        return st


# -- persistence ------------------------------------------------------------
def state_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "state.json"


def save_state(run_dir: Path | str, state: PaperState) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_text(state_path(run_dir),
                      json.dumps(state.to_dict(), ensure_ascii=False, indent=2))


def load_state(run_dir: Path | str, *, force: bool = False) -> PaperState:
    """Load state.json. Refuses a corrupt file unless force=True."""
    p = state_path(run_dir)
    if not p.is_file():
        pid = Path(run_dir).name
        return PaperState(paper_id=pid)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return PaperState.from_dict(d)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        if force:
            pid = Path(run_dir).name
            return PaperState(paper_id=pid)
        raise StateError(
            f"state.json at {p} is corrupt ({e}); pass force=True to reinit"
        ) from e


def artifact_digest(text: str) -> str:
    return sha256_text(text)[:16]
