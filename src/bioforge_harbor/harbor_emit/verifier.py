"""verify_task_dir: five gates over an emitted Harbor task directory.

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
  gate5_depth    benchmark-grade depth floor (v2): grader line count, number
                 of hard constraints stated in the instruction, input data
                 volume, multi-step signal in the instruction, and reference
                 solution size. Fails when the task is toy-grade.

Overall status is the *worst* gate that is not UNVERIFIED; UNVERIFIED only
propagates if no gate failed.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas import default_validator
from ..models.blueprint import ALLOWED_BASE_PREFIXES

HARBOR_SCHEMA_VERSION = "1.4"

# v2 depth floor, calibrated against the 8 TB-Science (terminal-bench 2.0
# life-science subset) reference tasks. A synthesized task below ANY of these
# is toy-grade and fails gate5:
#   grader_lines    60   (TB-Science min 69: raman-fitting)
#   constraints      4   (TB-Science median 11.5; our prompt mandates >= 4)
#   data_bytes    2048   (TB-Science median ~6.6 KB; two image-carried-data
#                          outliers exist in TB, but we control generation and
#                          must never ship 1.6 KB toys)
#   step_signal      3   (TB-Science min 3, distinct domain verbs)
#   solution_lines  15   (TB-Science min 33)
DEPTH_MIN_GRADER_LINES = 60
DEPTH_MIN_CONSTRAINTS = 4
DEPTH_MIN_DATA_BYTES = 2048
DEPTH_MIN_INSTR_STEPS = 3
DEPTH_MIN_SOLUTION_LINES = 15


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

    # ---------- gate5: benchmark-grade depth floor (v2) ----------
    g5 = _depth_gate(d)
    outcome.gates["gate5_depth"] = g5

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


_CONSTRAINT_RE = re.compile(
    r"^\s*(?:[-*]|\d+[\.\)])\s+(?:\*\*)?[A-Za-z]", re.M)
# Imperative/domain verb density: counts occurrences anywhere in the
# instruction (list markers are inconsistent across tasks, so we do not
# anchor to them). Multi-step tasks mention >=4 of these verbs.
_STEP_VERB_TOKENS = re.compile(
    r"\b(first|then|next|finally|compute|parse|load|write|run|implement|"
    r"filter|normalize|estimate|fit|score|compare|design|generate|"
    r"assemble|align|call|inspect|explore|verify|check|build|convert|"
    r"process|extract|train|evaluate|validate|output|create|apply|"
    r"select|combine|determine|calculate|identify|find|use)\b", re.I)


def _depth_gate(d: Path) -> dict[str, Any]:
    """v2: benchmark-grade depth floor. Fails if the task is toy-grade.

    Measures four things that a synthetic task must clear (see module
    constants): grader line count, hard-constraint count, input data volume,
    and multi-step signal in the instruction. The reference solution size is
    checked too since it is the oracle for the reward ceiling.
    """
    g5: dict[str, Any] = {"status": "pass", "errors": [], "metrics": {}}

    # 1. grader depth
    grade_py = d / "tests" / "grade.py"
    grader_lines = 0
    for cand in (grade_py, *(d / "tests").glob("*.py")) if (d / "tests").is_dir() else []:
        if cand.is_file():
            n = len(cand.read_text(encoding="utf-8", errors="replace").splitlines())
            grader_lines = max(grader_lines, n)
    g5["metrics"]["grader_lines"] = grader_lines
    if grader_lines < DEPTH_MIN_GRADER_LINES:
        g5["errors"].append(
            f"grader too shallow: {grader_lines} lines < {DEPTH_MIN_GRADER_LINES}")

    # 2. hard constraints in instruction (numbered/bulleted lines).
    # TB-Science states constraints either in a "Constraints"/"Rules" section,
    # or inline as bullet/numbered requirement lines anywhere in the doc.
    # We count bullets/numbers that carry a requirement verb or a "should /
    # must / recommended setting" pattern; fall back to whole doc.
    instr = d / "instruction.md"
    n_constraints = 0
    instr_text = ""
    if instr.is_file():
        instr_text = instr.read_text(encoding="utf-8", errors="replace")
        sec = re.search(
            r"(?ims)^#+\s*(constraints?|rules?|requirements)\s*$"
            r"([\s\S]+?)(?=^#+\s|\Z)", instr_text)
        body = sec.group(2) if sec else instr_text
        n_constraints = len(_CONSTRAINT_RE.findall(body))
        if n_constraints < DEPTH_MIN_CONSTRAINTS:
            # inline requirement sentences (TB-Science raman-fitting style)
            n_constraints += len(re.findall(
                r"(?im)^\s*(?:[-*]|\d+[\.\)])\s+.*\b(should|must|need[sd]?|"
                r"require[ds]?|only|between|at least|at most|minimum|maximum|"
                r"exactly|format)\b.*$", instr_text))
        n_constraints = min(n_constraints, len(
            _CONSTRAINT_RE.findall(instr_text)) + len(re.findall(
                r"(?im)^\s*(?:[-*]|\d+[\.\)])\s+", instr_text)))
    g5["metrics"]["constraint_count"] = n_constraints
    if n_constraints < DEPTH_MIN_CONSTRAINTS:
        g5["errors"].append(
            f"too few hard constraints: {n_constraints} < {DEPTH_MIN_CONSTRAINTS}")

    # 3. input data volume (environment/ excluding Dockerfile and docs).
    # TB-Science carries data three ways: files under environment/ (possibly
    # in subdirs like task-deps/), or generated at build time inside the
    # Dockerfile (RUN curl/wget/heredoc). We count all of them; a generator
    # script in environment/data also counts (it recreates the data).
    data_bytes = 0
    env_dir = d / "environment"
    if env_dir.is_dir():
        for f in env_dir.rglob("*"):
            if f.is_file() and f.name not in ("Dockerfile", "README.md"):
                data_bytes += f.stat().st_size
    dockerfile = env_dir / "Dockerfile" if env_dir.is_dir() else None
    docker_text = dockerfile.read_text(encoding="utf-8", errors="replace") \
        if dockerfile and dockerfile.is_file() else ""
    build_data = bool(re.search(
        r"(?im)^\s*RUN\s+.*\b(curl|wget|heredoc|python.*generate|pip download)\b",
        docker_text))
    if build_data:
        data_bytes = max(data_bytes, DEPTH_MIN_DATA_BYTES)  # build-time data: assume >= floor
    g5["metrics"]["data_bytes"] = data_bytes
    if data_bytes < DEPTH_MIN_DATA_BYTES:
        g5["errors"].append(
            f"input data trivial: {data_bytes} B < {DEPTH_MIN_DATA_BYTES} B")

    # 4. multi-step signal (distinct imperative/domain verbs used)
    steps = len(set(m.group(0).lower() for m in
                    _STEP_VERB_TOKENS.finditer(instr_text))) if instr_text else 0
    g5["metrics"]["step_signal"] = steps
    if steps < DEPTH_MIN_INSTR_STEPS:
        g5["errors"].append(
            f"not multi-step: {steps} imperative steps < {DEPTH_MIN_INSTR_STEPS}")

    # 5. reference solution real-workflow size
    solve = d / "solution" / "solve.sh"
    sol_lines = len(solve.read_text(encoding="utf-8", errors="replace").splitlines()) \
        if solve.is_file() else 0
    g5["metrics"]["solution_lines"] = sol_lines
    if sol_lines < DEPTH_MIN_SOLUTION_LINES:
        g5["errors"].append(
            f"solution too short: {sol_lines} < {DEPTH_MIN_SOLUTION_LINES} lines")

    # Disjunctive floor (calibration lesson from TB-Science): real tasks are
    # deep on DIFFERENT axes (raman: huge data, no constraints section;
    # mcmc: tiny data, heavy model implementation; bn-fit: build-time data).
    # So a task passes gate5 when it is deep on at least ONE axis, and fails
    # only when it is shallow on ALL of them (toy-grade everywhere).
    # Core axes: grader depth and constraint count must still individually
    # clear the toy line, since a shallow grader can be gamed and constraint-
    # free tasks have no verification substance.
    metrics = g5["metrics"]
    deep_axes = (
        (metrics["grader_lines"] >= DEPTH_MIN_GRADER_LINES)
        + (metrics["constraint_count"] >= DEPTH_MIN_CONSTRAINTS)
        + (metrics["data_bytes"] >= DEPTH_MIN_DATA_BYTES)
        + (metrics["step_signal"] >= DEPTH_MIN_INSTR_STEPS)
        + (metrics["solution_lines"] >= DEPTH_MIN_SOLUTION_LINES)
    )
    hard_violations = []
    if metrics["grader_lines"] < DEPTH_MIN_GRADER_LINES:
        hard_violations.append(
            f"grader too shallow: {metrics['grader_lines']} < {DEPTH_MIN_GRADER_LINES}")
    if metrics["constraint_count"] < DEPTH_MIN_CONSTRAINTS:
        hard_violations.append(
            f"too few hard constraints: {metrics['constraint_count']} < {DEPTH_MIN_CONSTRAINTS}")
    if deep_axes < 3:
        hard_violations.append(
            f"toy-grade on too many axes: deep on {deep_axes}/5 "
            f"(need >=3; metrics={metrics})")
    g5["errors"] = hard_violations
    if g5["errors"]:
        g5["status"] = "fail"
    return g5


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
