"""Construction stage (paper section 3.2: "Construction Agent").

Turns one screened paper into a TaskBlueprint via the agent backend, then runs
the deterministic emitter + verifier. The LLM invents content; the blueprint
schema and the four gates are what keep it honest. If the emitted task fails
verification, the stage reports ``failed`` with the gate detail — it does NOT
silently ship a broken environment.

Offline note: with MockBackend/DryRunBackend this whole path runs with no
model and no network, which is what makes the pipeline testable end-to-end.
"""

from __future__ import annotations

from typing import Any

from ...models.blueprint import TaskBlueprint
from ...models.reports import StageResult
from ...schemas import default_validator
from ...harbor_emit import emit_task, verify_task_dir
from ..context import RunContext, digest
from . import register


def _paper_block(record: dict[str, Any]) -> str:
    return (
        f"Title: {record.get('title','')}\n"
        f"DOI: {record.get('doi','')}\n"
        f"PMCID: {record.get('pmcid','')}\n"
        f"Year: {record.get('pub_year','')}\n"
        f"Abstract: {record.get('abstract','')}\n"
    )


@register("construct")
def construct(ctx: RunContext, record: dict[str, Any]) -> StageResult:
    prompt = ctx.load_prompt("constructor.build") + "\n\n" + _paper_block(record)
    try:
        raw = ctx.backend.ask_json("constructor.build", prompt, schema_name="blueprint_1")
    except Exception as e:  # noqa: BLE001
        return StageResult(stage="construct", status="failed", error=f"backend: {e}")

    # schema-check the model's blueprint before trusting it
    vres = default_validator().validate("blueprint_1", raw)
    if not vres.ok:
        return StageResult(stage="construct", status="failed",
                           error="blueprint schema: " + vres.summary())
    try:
        bp = TaskBlueprint.from_dict(raw)
    except Exception as e:  # noqa: BLE001
        return StageResult(stage="construct", status="failed", error=f"blueprint: {e}")

    berrs = bp.validate()
    if berrs:
        return StageResult(stage="construct", status="failed",
                           error="blueprint.validate: " + "; ".join(berrs))

    ctx.write_json("construct", "blueprint.json", bp.to_dict())

    task_dir = ctx.stage_dir("construct") / "harbor_task"
    rep = emit_task(bp, task_dir)
    ctx.write_json("construct", "emit-report.json", rep.to_dict())
    if rep.status == "failed":
        return StageResult(stage="construct", status="failed",
                           error="emit: " + "; ".join(rep.errors))

    ver = verify_task_dir(task_dir)
    ctx.write_json("construct", "verify-report.json", ver.to_dict())
    if not ver.ok:
        return StageResult(stage="construct", status="failed",
                           error=f"verify: {ver.status} "
                                 f"{[k for k,g in ver.gates.items() if g.get('status')=='fail']}")

    return StageResult(stage="construct", status="ok",
                       artifact="construct/blueprint.json", digest=digest(bp.to_dict()))
