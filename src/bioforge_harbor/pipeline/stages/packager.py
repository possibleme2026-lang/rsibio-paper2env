"""Packager stage (paper section 3.2: "Packager Agent").

Promotes the verified construction output into the final tasks/ layout and
writes a machine-readable manifest (the environment set Stage 1 measures).
This is where a validated environment becomes a Harbor *task directory* that
``harbor run`` can consume, with a stable name and full provenance.

The packager refuses to ship unless the construct output passed verification
and the review did not hard-fail — the last line of "no fake green".
"""

from __future__ import annotations

import shutil
from typing import Any

from ...models.reports import StageResult
from ...harbor_emit import verify_task_dir
from ..context import RunContext, digest
from . import register


@register("package")
def package(ctx: RunContext, record: dict[str, Any]) -> StageResult:
    src = ctx.stage_dir("construct") / "harbor_task"
    if not (src / "task.toml").is_file():
        return StageResult(stage="package", status="failed",
                           error="no constructed harbor_task to package")

    # re-verify at the last gate before shipping
    ver = verify_task_dir(src)
    if not ver.ok:
        return StageResult(stage="package", status="failed",
                           error=f"pre-package verify failed: {ver.status}")

    try:
        rev = ctx.read_json("review", "report.json")
    except FileNotFoundError:
        rev = {}
    if rev.get("approved") is False and rev.get("biosafety", {}).get("status") == "block":
        return StageResult(stage="package", status="failed",
                           error="review blocked (biosafety)")

    bp = ctx.read_json("construct", "blueprint.json")
    task_name = bp.get("task_name", f"bioforge/{ctx.paper_id}")
    slug = task_name.split("/")[-1]

    # destination: tasks/<slug> at repo root (the published environment set)
    tasks_root = ctx.run_dir.parent.parent / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    dest = tasks_root / slug
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    manifest = {
        "task_name": task_name,
        "dir": str(dest.relative_to(tasks_root.parent)),
        "paper_id": ctx.paper_id,
        "provenance": bp.get("provenance"),
        "metric": bp.get("metric"),
        "verify_status": ver.status,
        "review_approved": bool(rev.get("approved")),
    }
    ctx.write_json("package", "manifest.json", manifest)
    # append to a set-level manifest (one line per env) for Stage-1 tooling
    _append_set_manifest(tasks_root / "manifest.jsonl", manifest)

    return StageResult(stage="package", status="ok",
                       artifact=manifest["dir"], digest=digest(manifest))


def _append_set_manifest(path, manifest: dict[str, Any]) -> None:
    import json
    line = json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n"
    existing = ""
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
    # de-dup by task_name (last write wins) but keep LF
    rows = [json.loads(l) for l in existing.splitlines() if l.strip()]
    rows = [r for r in rows if r.get("task_name") != manifest.get("task_name")]
    rows.append(manifest)
    rows.sort(key=lambda r: r.get("task_name", ""))
    from ...util.io import write_text_lf
    write_text_lf(path, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                for r in rows))
