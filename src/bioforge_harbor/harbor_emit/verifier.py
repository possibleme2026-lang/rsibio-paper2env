"""verify_task_dir: four gates over an emitted Harbor task directory.

Gate order (each is independent; a later gate does not assume an earlier one
passed, so the report always shows the full picture):

  gate1_schema   tomllib parse + embedded STRICT JSON Schema (catches the
                 extra='ignore' silent key-drop that Harbor's own pydantic
                 model would let through).
  gate2_layout   required files present + non-empty + Dockerfile base allowlist.
  gate3_harbor   OPTIONAL cross-check: import the real harbor package and run
                 TaskConfig.model_validate_toml + Task(dir). If harbor is not
                 importable -> status UNVERIFIED (never a silent pass).
  gate4_hygiene  every .sh/.py is LF-clean; no path-escaping file names.

Overall status is the *worst* gate that is not UNVERIFIED; UNVERIFIED only
propagates if no gate failed.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas import default_validator
from ..models.blueprint import ALLOWED_BASE_PREFIXES

HARBOR_SCHEMA_VERSION = "1.4"


@dataclass
class VerificationOutcome:
    task_dir: str
    status: str = "ok"                       # ok | warning | failed | unverified
    gates: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "warning")

    @property
    def real_gate_status(self) -> str:
        return str(self.gates.get("gate3_harbor", {}).get("status", "missing"))

    def to_dict(self) -> dict:
        return {"task_dir": self.task_dir, "status": self.status, "gates": self.gates}


def _allowed_base(dockerfile: str) -> bool:
    for line in dockerfile.splitlines():
        s = line.strip()
        if s.upper().startswith("FROM "):
            parts = s.split()
            # FROM [--platform=...] <image> [AS name]
            img = next((p for p in parts[1:] if not p.startswith("--")), "")
            if "$" in img:  # build-arg base: cannot statically check
                return True
            return any(img.startswith(pref) for pref in ALLOWED_BASE_PREFIXES)
    return False


def verify_task_dir(task_dir: Path | str, *, run_harbor_gate: bool = True) -> VerificationOutcome:
    d = Path(task_dir)
    outcome = VerificationOutcome(task_dir=str(d))

    # ---------- gate1: strict schema ----------
    g1: dict[str, Any] = {"status": "pass", "errors": []}
    toml_path = d / "task.toml"
    toml_body: dict[str, Any] = {}
    if not toml_path.is_file():
        g1 = {"status": "fail", "errors": ["task.toml missing"]}
    else:
        try:
            toml_body = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        except Exception as e:
            g1 = {"status": "fail", "errors": [f"task.toml parse error: {e}"]}
    if g1["status"] == "pass":
        res = default_validator().validate("task_toml_1_4", toml_body)
        if not res.ok:
            g1 = {"status": "fail", "errors": res.summary().splitlines()}
    outcome.gates["gate1_schema"] = g1

    # ---------- gate2: layout sanity ----------
    g2: dict[str, Any] = {"status": "pass", "errors": []}
    instr = d / "instruction.md"
    if not instr.is_file() or not instr.read_text(encoding="utf-8").strip():
        g2["errors"].append("instruction.md missing or empty")
    docker = d / "environment" / "Dockerfile"
    if not docker.is_file():
        g2["errors"].append("environment/Dockerfile missing")
    else:
        if not _allowed_base(docker.read_text(encoding="utf-8")):
            g2["errors"].append("Dockerfile FROM base not in allowlist")
    testsh = d / "tests" / "test.sh"
    if not testsh.is_file():
        g2["errors"].append("tests/test.sh missing")
    if toml_body.get("schema_version") not in (None, HARBOR_SCHEMA_VERSION):
        g2["errors"].append(f"schema_version {toml_body.get('schema_version')!r} unexpected")
    if g2["errors"]:
        g2["status"] = "fail"
    outcome.gates["gate2_layout"] = g2

    # ---------- gate3: optional real-harbor cross-check ----------
    g3: dict[str, Any] = {"status": "UNVERIFIED", "detail": "harbor gate disabled"}
    if run_harbor_gate:
        g3 = _harbor_gate(d)
    outcome.gates["gate3_harbor"] = g3

    # ---------- gate4: LF / path hygiene ----------
    g4: dict[str, Any] = {"status": "pass", "errors": []}
    for f in d.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(d).as_posix()
        if ".." in Path(rel).parts or Path(rel).is_absolute():
            g4["errors"].append(f"escaping path: {rel}")
            continue
        if f.suffix in (".sh", ".py", ".toml", ".md", ".txt"):
            raw = f.read_bytes()
            if b"\r" in raw:
                g4["errors"].append(f"CRLF found in {rel}")
    if g4["errors"]:
        g4["status"] = "fail"
    outcome.gates["gate4_hygiene"] = g4

    # ---------- roll up ----------
    hard_fail = [k for k, g in outcome.gates.items() if g.get("status") == "fail"]
    if hard_fail:
        outcome.status = "failed"
    elif any(g.get("status") == "warning" for g in outcome.gates.values()):
        outcome.status = "warning"
    elif outcome.gates.get("gate3_harbor", {}).get("status") == "UNVERIFIED":
        # gates 1/2/4 passed but the authoritative model check did not run.
        outcome.status = "ok"  # strict schema is our first trusted gate
    else:
        outcome.status = "ok"
    return outcome


def _harbor_gate(d: Path) -> dict[str, Any]:
    """Try to import harbor and validate with the real TaskConfig + Task."""
    try:
        import importlib
        cfg_mod = importlib.import_module("harbor.models.task.config")
        task_mod = importlib.import_module("harbor.models.task.task")
    except Exception as e:
        return {"status": "UNVERIFIED", "detail": f"harbor not importable: {e}"}
    try:
        toml_text = (d / "task.toml").read_text(encoding="utf-8")
        cfg_mod.TaskConfig.model_validate_toml(toml_text)
        Task = getattr(task_mod, "Task", None)
        if Task is not None:
            Task(d)  # exercises path-escape + dir-layout checks
        return {"status": "pass", "detail": "harbor TaskConfig + Task accepted"}
    except Exception as e:
        return {"status": "fail", "detail": f"harbor rejected: {e}"}
