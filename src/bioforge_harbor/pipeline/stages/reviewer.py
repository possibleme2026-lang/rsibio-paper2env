"""Review stage (paper section 3.2: "Review Agent").

Three independent checks, each with its own prompt so a finding is attributable
to a specific rubric rather than one blended opinion:

  fidelity   — does the task actually test the paper's method?
  shortcut   — can a trivial strategy (copy the gold file, print a constant)
               fool the evaluator?  (the paper's central failure mode)
  biosafety  — computational-analysis-only; reject dual-use red flags.

The stage FAILS CLOSED on biosafety: any biosafety concern blocks packaging.
For fidelity/shortcut it degrades to ``warning`` (a human expert reviews the
subset), never silently to ``ok``.
"""

from __future__ import annotations

from typing import Any

from ...models.reports import StageResult
from ..context import RunContext, digest
from . import register

# Dual-use terms that must never appear as the *task objective*. Kept as an
# explicit denylist so the rule is auditable and testable, not vibes.
BIOSAFETY_DENY = [
    "gain of function", "gain-of-function", "weaponiz", "pathogen enhancement",
    "aerosol transmission", "variola", "ebolavirus synthesis", "anthrax",
]


def _scan_biosafety(text: str) -> list[str]:
    hay = (text or "").lower()
    return [term for term in BIOSAFETY_DENY if term in hay]


@register("review")
def review(ctx: RunContext, record: dict[str, Any]) -> StageResult:
    try:
        bp = ctx.read_json("construct", "blueprint.json")
    except FileNotFoundError:
        return StageResult(stage="review", status="failed",
                           error="no construct/blueprint.json")

    task_text = (str(bp.get("instruction_md", "")) + "\n"
                 + "\n".join(str(v) for v in (bp.get("test_files") or {}).values()))

    # deterministic biosafety pre-screen (never trust the LLM alone here)
    hits = _scan_biosafety(task_text)
    if hits:
        out = {"approved": False, "biosafety": {"status": "block", "matched": hits},
               "findings": [{"kind": "biosafety", "detail": f"denylist term: {h}"} for h in hits]}
        ctx.write_json("review", "report.json", out)
        return StageResult(stage="review", status="failed",
                           error="biosafety block: " + ", ".join(hits),
                           digest=digest(out))

    findings: list[dict[str, Any]] = []
    for kind, prompt_name in (("fidelity", "reviewer.fidelity"),
                              ("shortcut", "reviewer.shortcut"),
                              ("biosafety", "reviewer.biosafety")):
        prompt = ctx.load_prompt(prompt_name) + "\n\nTASK:\n" + task_text[:2000]
        try:
            resp = ctx.backend.ask_json(f"reviewer.{kind}", prompt)
        except Exception as e:  # noqa: BLE001
            resp = {"concern": f"backend error: {e}"}
        # Fail-closed interpretation: an ambiguous or erroring review is a
        # finding, never a silent pass.
        if "pass" in resp:
            flag = not bool(resp["pass"])
        elif "concern" in resp:
            flag = True
        elif "ok" in resp:
            flag = not bool(resp["ok"])
        else:
            flag = True
        if flag:
            findings.append({"kind": kind,
                             "detail": str(resp.get("concern") or resp.get("reason")
                                           or "no explicit pass")})

    approved = not findings
    out = {
        "approved": approved,
        "biosafety": {"status": "ok", "matched": []},
        "findings": findings,
    }
    ctx.write_json("review", "report.json", out)
    status = "ok" if approved else "warning"
    return StageResult(stage="review", status=status,
                       artifact="review/report.json", digest=digest(out))
