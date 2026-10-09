"""Runner: orchestrates the five stages over a set of papers.

Responsibilities:
  * build a RunContext per paper,
  * drive curate -> construct -> reproduce -> review -> package,
  * persist PaperState after every stage so a crash resumes cleanly,
  * honor --resume (skip completed) / --force (rerun) / --stages (subset).

The curate stage is corpus-level (one pass over all papers); the other four are
per-paper. This asymmetry mirrors the paper: the Curator screens a corpus, the
other agents each handle one paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..backends.base import AgentBackend
from ..models.records import PaperRecord
from ..models.reports import SelectionReport
from .context import RunContext
from .state import (
    STAGES, PaperState, load_state, save_state, StateError,
)
from .stages import get_stage
from .stages import curator as _curate_mod


@dataclass
class RunSummary:
    papers: int = 0
    packaged: int = 0
    failed: int = 0
    skipped: int = 0
    per_paper: dict[str, str] = field(default_factory=dict)   # paper_id -> final status

    def to_dict(self) -> dict[str, Any]:
        return {"papers": self.papers, "packaged": self.packaged,
                "failed": self.failed, "skipped": self.skipped,
                "per_paper": self.per_paper}


def curate_corpus(records: list[PaperRecord], *, select_n: int = 200,
                  year_max: int = 2023) -> SelectionReport:
    cfg = _curate_mod.CuratorConfig(select_n=select_n, year_max=year_max)
    return _curate_mod.curate(records, cfg)


def run_pipeline(
    records: list[PaperRecord],
    backend: AgentBackend,
    *,
    runs_dir: Path | str,
    stages: tuple[str, ...] = STAGES,
    resume: bool = True,
    force: bool = False,
    selected_ids: set[str] | None = None,
) -> RunSummary:
    """Run the per-paper stages. Returns a RunSummary.

    ``selected_ids`` (from a prior curate pass) restricts which papers proceed;
    if None, every record is processed.
    """
    runs_dir = Path(runs_dir)
    summary = RunSummary()

    todo = [r for r in records if (selected_ids is None or r.paper_id in selected_ids)]
    summary.papers = len(todo)

    for rec in todo:
        run_dir = runs_dir / rec.paper_id
        try:
            state = load_state(run_dir, force=force)
        except StateError:
            # corrupt state and no --force: refuse rather than clobber
            summary.failed += 1
            summary.per_paper[rec.paper_id] = "state_corrupt"
            continue

        ctx = RunContext(paper_id=rec.paper_id, run_dir=run_dir, backend=backend)
        record_dict = rec.to_dict()

        # Curate is corpus-level; for a selected paper, mark it done once so
        # the per-paper state machine can legitimately advance to construct.
        if "curate" in stages and not (
            state.attempts.get("curate") and state.attempts["curate"].status in ("ok", "skipped")
        ):
            state.advance("curate", status="ok", digest="corpus")
            save_state(run_dir, state)

        final = "ok"
        for stage in stages:
            if stage == "curate":
                continue  # corpus-level, handled by curate_corpus()
            if resume and not force:
                if state.attempts.get(stage) and state.attempts[stage].status in ("ok", "skipped"):
                    continue
                if not state.can_resume_into(stage):
                    final = f"blocked_before_{stage}"
                    break
            try:
                fn = get_stage(stage)
            except KeyError as e:
                final = f"no_stage:{e}"
                break
            result = fn(ctx, record_dict)
            state.advance(stage, digest=result.digest,
                          status=("ok" if result.status == "ok" else
                                  "skipped" if result.status == "warning" else "failed"),
                          error=result.error)
            save_state(run_dir, state)
            if result.status == "failed":
                final = f"failed_at_{stage}"
                break
        else:
            final = "packaged" if "package" in stages else "ok"

        summary.per_paper[rec.paper_id] = final
        if final == "packaged":
            summary.packaged += 1
        elif final.startswith("failed") or final.startswith("no_stage"):
            summary.failed += 1
        elif final.startswith("blocked"):
            summary.skipped += 1
    return summary
