"""emit_task: TaskBlueprint -> a Harbor task directory on disk.

The emitter is *deterministic*: same blueprint -> byte-identical files. All
model creativity is confined to the blueprint; this layer is a compiler. Two
invariants from EnvForge are enforced here:

1. Every file is written with LF endings (util.io.write_text_lf).
2. Any blueprint field we cannot faithfully place in the Harbor layout is
   recorded in ``EmitReport.lost_semantics`` — never silently dropped.
"""

from __future__ import annotations

from pathlib import Path

from ..models.blueprint import TaskBlueprint
from ..models.reports import EmitReport
from ..util.io import write_text_lf
from .tomlx import dump_toml

HARBOR_SCHEMA_VERSION = "1.4"

# Harbor's canonical task layout (from the official hello-world example):
#   task.toml  instruction.md  environment/Dockerfile  tests/  solution/
REQUIRED_TEST_ENTRY = "tests/test.sh"


def _default_task_toml(bp: TaskBlueprint) -> dict:
    """Build a minimal-but-complete task.toml body when the blueprint omits one."""
    return {
        "schema_version": HARBOR_SCHEMA_VERSION,
        "task": {
            "name": bp.task_name,
            "version": bp.version or "0.1.0",
            "description": (bp.instruction_md.strip().splitlines() or [""])[0][:200],
            "keywords": list(bp.keywords),
        },
        "metadata": {
            "category": bp.category,
            "difficulty": bp.difficulty,
            "metric": bp.metric,
            "generated_by": "bioforge-harbor",
        },
        "agent": {"timeout_sec": 600.0},
        "verifier": {"timeout_sec": 300.0},
        "environment": {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240, "gpus": 0},
    }


def _merge_provenance(toml_body: dict, bp: TaskBlueprint, report: EmitReport) -> None:
    prov = bp.provenance
    if prov is None:
        return
    meta = toml_body.setdefault("metadata", {})
    if not isinstance(meta, dict):
        report.warnings.append("metadata was not a table; provenance not merged")
        return
    meta.setdefault("source_paper_id", prov.paper_id)
    if prov.doi:
        meta.setdefault("source_doi", prov.doi)
    if prov.pmcid:
        meta.setdefault("source_pmcid", prov.pmcid)
    if prov.url:
        meta.setdefault("source_url", prov.url)
    if prov.abstract_digest:
        meta.setdefault("source_abstract_digest", prov.abstract_digest)


def emit_task(bp: TaskBlueprint, out_dir: Path | str) -> EmitReport:
    """Write the Harbor task for *bp* into *out_dir*. Returns an EmitReport.

    Does NOT run the verifier; callers should follow with verify_task_dir().
    """
    out = Path(out_dir)
    report = EmitReport(stage="emit", task_dir=str(out))

    errs = bp.validate()
    if errs:
        report.status = "failed"
        report.errors = errs
        return report

    # 1) task.toml
    toml_body = dict(bp.task_toml) if bp.task_toml else _default_task_toml(bp)
    # Force the org/name + schema_version to agree with the blueprint so a
    # stale blueprint.task_toml can't emit a mismatched name.
    toml_body.setdefault("schema_version", HARBOR_SCHEMA_VERSION)
    if toml_body["schema_version"] != HARBOR_SCHEMA_VERSION:
        report.warnings.append(
            f"schema_version {toml_body['schema_version']!r} != {HARBOR_SCHEMA_VERSION!r}"
        )
    task_sec = toml_body.setdefault("task", {})
    if task_sec.get("name") and task_sec["name"] != bp.task_name:
        # Conflict: the blueprint field is the validated authority
        # (TaskBlueprint.validate checked its org/name form); record the
        # override so nothing is silently rewritten.
        report.lost_semantics.append(
            f"task_toml.task.name {task_sec['name']!r} overridden by validated "
            f"blueprint.task_name {bp.task_name!r}"
        )
    task_sec["name"] = bp.task_name
    _merge_provenance(toml_body, bp, report)
    toml_text = dump_toml(toml_body)
    write_text_lf(out / "task.toml", toml_text)
    report.files_written.append("task.toml")
    report.mapped.append("task.toml <- task_toml (+provenance into [metadata])")

    # 2) instruction.md
    write_text_lf(out / "instruction.md", bp.instruction_md.rstrip("\n") + "\n")
    report.files_written.append("instruction.md")

    # 3) environment/Dockerfile
    write_text_lf(out / "environment" / "Dockerfile", bp.dockerfile.rstrip("\n") + "\n")
    report.files_written.append("environment/Dockerfile")

    # 4) env_files (inputs / data) under environment/
    for rel, content in sorted(bp.env_files.items()):
        rel_norm = _safe_rel(rel)
        if rel_norm is None:
            report.lost_semantics.append(f"env_files path rejected (escape/absolute): {rel!r}")
            continue
        # accept "data/x" or "environment/data/x"; normalize under environment/
        sub = rel_norm[len("environment/"):] if rel_norm.startswith("environment/") else rel_norm
        write_text_lf(out / "environment" / sub, content)
        report.files_written.append(f"environment/{sub}")

    # 5) tests/
    for rel, content in sorted(bp.test_files.items()):
        rel_norm = _safe_rel(rel)
        if rel_norm is None:
            report.lost_semantics.append(f"test path rejected: {rel!r}")
            continue
        # blueprint may give "tests/test.sh" or "test.sh"; normalize under tests/
        sub = rel_norm[len("tests/"):] if rel_norm.startswith("tests/") else rel_norm
        write_text_lf(out / "tests" / sub, content)
        report.files_written.append(f"tests/{sub}")

    # 6) solution/
    for rel, content in sorted(bp.solution_files.items()):
        rel_norm = _safe_rel(rel)
        if rel_norm is None:
            report.lost_semantics.append(f"solution path rejected: {rel!r}")
            continue
        sub = rel_norm[len("solution/"):] if rel_norm.startswith("solution/") else rel_norm
        write_text_lf(out / "solution" / sub, content)
        report.files_written.append(f"solution/{sub}")

    # 7) notes -> a visible REVIEW.md so a human reviewer sees the caveats
    if bp.notes:
        body = "# Review notes\n\n" + "\n".join(f"- {n}" for n in bp.notes) + "\n"
        write_text_lf(out / "REVIEW.md", body)
        report.files_written.append("REVIEW.md")

    if report.lost_semantics:
        report.status = "warning"
    return report


def _safe_rel(rel: str) -> str | None:
    """Normalize a blueprint-supplied relative path; reject escapes/absolute."""
    r = (rel or "").replace("\\", "/").strip("/")
    if not r:
        return None
    parts = r.split("/")
    if any(p in ("", "..", ".") for p in parts):
        return None
    if ":" in parts[0]:  # windows drive letter
        return None
    return r
