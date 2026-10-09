"""Stage reports: the audit trail each pipeline stage must emit.

Design rule inherited from EnvForge: **any semantic the pipeline drops must
appear in a report**. A stage that "succeeded" while silently discarding
information is worse than a stage that failed, because downstream readers
trust the green checkmark. Hence every report carries ``warnings`` and
``lost_semantics`` lists that the verifier and the guards inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Report:
    """Common base: status + human-readable diagnostics."""

    stage: str
    status: str = "ok"                 # ok | warning | failed | unverified
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lost_semantics: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "warning") and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Report":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SelectionReport(Report):
    """Curator: which papers were screened in/out and why."""

    stage: str = "curate"
    screened: int = 0
    selected: int = 0
    selected_ids: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)  # {paper_id, reason}
    bucket_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class EmitReport(Report):
    """Construction/Packager: blueprint -> Harbor directory."""

    stage: str = "emit"
    task_dir: str = ""
    files_written: list[str] = field(default_factory=list)
    mapped: list[str] = field(default_factory=list)


@dataclass
class VerificationReport(Report):
    """Harbor four-gate verification result."""

    stage: str = "verify"
    task_dir: str = ""
    gates: dict[str, Any] = field(default_factory=dict)   # gate name -> {status, detail}

    @property
    def real_gate_status(self) -> str:
        g = self.gates.get("harbor_real", {})
        return str(g.get("status", "missing"))


@dataclass
class ReproductionReport(Report):
    """Reproduction: independent baseline + agreement with construction."""

    stage: str = "reproduce"
    baseline_metric: float | None = None
    paper_reported_metric: float | None = None
    agreement: float | None = None      # normalized |baseline - paper| distance
    second_channel_metric: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ReviewReport(Report):
    """Review: fidelity / shortcut-resistance / biosafety findings."""

    stage: str = "review"
    fidelity: dict[str, Any] = field(default_factory=dict)
    shortcut: dict[str, Any] = field(default_factory=dict)
    biosafety: dict[str, Any] = field(default_factory=dict)
    approved: bool = False
    findings: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StageResult:
    """One stage's execution outcome inside a run (state machine bookkeeping)."""

    stage: str
    status: str                         # ok | failed | skipped | pending
    artifact: str = ""                  # relative path to the stage's main output
    digest: str = ""                    # short digest of the artifact
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StageResult":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})
