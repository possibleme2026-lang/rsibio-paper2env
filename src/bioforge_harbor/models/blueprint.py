"""TaskBlueprint: the Construction stage's output, the emitter's input.

The blueprint is the *single* contract between the LLM-driven stages (which
invent content) and the deterministic emitter (which writes Harbor files).
Keeping it a validated dataclass is what lets the whole pipeline run offline:
tests feed a hand-written blueprint and assert byte-level Harbor output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

ORG_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$")

# Base images we allow a generated Dockerfile to start FROM. Keeps the
# Packager from inventing exotic bases and gives Review a cheap first check.
ALLOWED_BASE_PREFIXES = (
    "python:",
    "python",
    "continuumio/miniconda3",
    "condaorg/",
    "rocker/",
    "bioconductor/",
    "ubuntu:",
    "debian:",
    "node:",
    "ghcr.io",
    "quay.io",
    "docker.io",
)


@dataclass
class Provenance:
    """Where a task came from. Always attached; never optional."""

    paper_id: str
    title: str = ""
    doi: str = ""
    pmcid: str = ""
    url: str = ""
    abstract_digest: str = ""          # sha256 prefix of source abstract
    source_db: str = "europepmc"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Provenance":
        if not d.get("paper_id"):
            raise ValueError("Provenance requires paper_id")
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TaskBlueprint:
    """Everything needed to deterministically emit one Harbor task directory."""

    task_name: str                     # "bioforge/<slug>" (org/name form)
    version: str = "0.1.0"
    instruction_md: str = ""
    dockerfile: str = ""
    test_files: dict[str, str] = field(default_factory=dict)     # tests/test.sh, tests/grade.py ...
    solution_files: dict[str, str] = field(default_factory=dict) # solution/solve.sh ...
    env_files: dict[str, str] = field(default_factory=dict)      # environment/data/... inputs
    task_toml: dict[str, Any] = field(default_factory=dict)      # the parsed task.toml body
    provenance: Provenance | None = None
    category: str = "computational-biology"
    difficulty: str = "medium"
    keywords: list[str] = field(default_factory=list)
    metric: str = ""                   # "f1" | "auroc" | "rmse" | "accuracy" | ...
    notes: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty == valid."""
        errs: list[str] = []
        if not ORG_NAME_RE.match(self.task_name or ""):
            errs.append(
                f"task_name must be org/name (letters/digits/._-), got {self.task_name!r}"
            )
        if not (self.instruction_md or "").strip():
            errs.append("instruction_md is empty")
        if not (self.dockerfile or "").strip():
            errs.append("dockerfile is empty")
        elif not self._base_allowed():
            errs.append(
                "dockerfile FROM base not in allowlist "
                f"({', '.join(ALLOWED_BASE_PREFIXES)})"
            )
        if "tests/test.sh" not in self.test_files:
            errs.append("test_files must include tests/test.sh")
        if not self.test_files:
            errs.append("no test files")
        if self.provenance is None:
            errs.append("provenance is required")
        return errs

    def _base_allowed(self) -> bool:
        for line in self.dockerfile.splitlines():
            s = line.strip()
            if s.upper().startswith("FROM "):
                base = s.split(None, 2)[1]
                return any(base.startswith(p) or base == p for p in ALLOWED_BASE_PREFIXES)
        return False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskBlueprint":
        prov = d.get("provenance")
        if isinstance(prov, dict):
            prov = Provenance.from_dict(prov)
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["provenance"] = prov
        return cls(**kwargs)
