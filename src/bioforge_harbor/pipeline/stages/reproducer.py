"""Reproduction stage (paper section 3.2: "Reproduction Agent").

Independently re-derives a baseline score for the constructed task, through a
*second channel* that does not see the construction blueprint's own claimed
numbers. This is the RSI-Forge discipline: the reproduction agent
reimplements the method to establish a baseline; agreement between the two
channels is the signal, not either channel alone.

Offline: MockBackend returns a canned baseline; the *comparison arithmetic*
here is real and tested. Live: the backend (pi) would actually run the task.
"""

from __future__ import annotations

from typing import Any

from ...models.reports import StageResult
from ..context import RunContext, digest
from . import register


def _agreement(a: float, b: float, tol: float) -> float:
    """Normalized agreement in [0,1]: 1.0 == identical, 0 == >=tol apart."""
    if tol <= 0:
        return 1.0 if a == b else 0.0
    return max(0.0, 1.0 - abs(a - b) / tol)


@register("reproduce")
def reproduce(ctx: RunContext, record: dict[str, Any]) -> StageResult:
    try:
        bp = ctx.read_json("construct", "blueprint.json")
    except FileNotFoundError:
        return StageResult(stage="reproduce", status="failed",
                           error="no construct/blueprint.json")

    prompt = ctx.load_prompt("reproducer.baseline") + "\n\n" + (
        f"Task: {bp.get('task_name')}\nMetric: {bp.get('metric')}\n"
        f"Instruction excerpt:\n{str(bp.get('instruction_md',''))[:1200]}\n"
    )
    try:
        resp = ctx.backend.ask_json("reproducer.baseline", prompt)
    except Exception as e:  # noqa: BLE001
        return StageResult(stage="reproduce", status="failed", error=f"backend: {e}")

    baseline = _as_float(resp.get("baseline_metric"))
    paper = _as_float(resp.get("paper_reported_metric"))
    second = _as_float(resp.get("second_channel_metric"))
    tol = float(resp.get("tolerance", 0.1))

    notes: list[str] = []
    if baseline is None:
        return StageResult(stage="reproduce", status="failed",
                           error="backend gave no baseline_metric")
    if paper is not None:
        agree = _agreement(baseline, paper, tol)
        notes.append(f"baseline vs paper-reported agreement={agree:.3f}")
    else:
        agree = None
        notes.append("paper-reported metric unavailable; agreement not computed")

    out = {
        "baseline_metric": baseline,
        "paper_reported_metric": paper,
        "second_channel_metric": second,
        "agreement": agree,
        "tolerance": tol,
        "notes": notes,
    }
    ctx.write_json("reproduce", "report.json", out)
    status = "ok" if (agree is None or agree >= 0.5) else "warning"
    return StageResult(stage="reproduce", status=status,
                       artifact="reproduce/report.json", digest=digest(out))


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
