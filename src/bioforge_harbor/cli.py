"""Top-level CLI: ``bioforge <command>``.

Two command families:

  pipeline   curate / construct / reproduce / review / package / run
             (the five paper-3.2 stages; run drives them end-to-end)

  validation validate-envs / run-weightfree / compare-harness /
             split-holdout / check-contamination
             (the four paper-4 stages)

Every command has an offline path (mock backend / manifest executor / dry-run)
so the whole surface is exercised in CI without a model, Docker, or network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .models.records import PaperRecord
from .pipeline.state import STAGES


def _load_records(path: Path) -> list[PaperRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("records", data.get("resultList", {}).get("result", []))
    # tolerate raw Europe PMC hits
    out = []
    for d in data:
        if "paper_id" in d:
            out.append(PaperRecord.from_dict(d))
        else:
            out.append(PaperRecord.from_europepmc(d))
    return out


def _backend_from_args(args):
    from .backends import make_backend
    kind = getattr(args, "backend", "mock")
    if kind == "mock":
        from .backends import MockBackend
        resp_dir = getattr(args, "mock_responses", None)
        if resp_dir:
            return MockBackend.from_dir(resp_dir)
        return MockBackend({}, strict=False)
    return make_backend(kind, provider=getattr(args, "provider", None),
                        model=getattr(args, "model", None))


# --------------------------------------------------------------------------
# pipeline commands
# --------------------------------------------------------------------------
def cmd_curate(args) -> int:
    from .pipeline.runner import curate_corpus
    recs = _load_records(Path(args.corpus))
    rep = curate_corpus(recs, select_n=args.select_n, year_max=args.year_max)
    out = {
        "screened": rep.screened, "selected": rep.selected,
        "selected_ids": rep.selected_ids, "bucket_counts": rep.bucket_counts,
        "rejected": rep.rejected,
    }
    text = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def cmd_run(args) -> int:
    from .pipeline.runner import curate_corpus, run_pipeline
    recs = _load_records(Path(args.corpus))
    backend = _backend_from_args(args)
    selected = None
    if args.select_n is not None:
        report = curate_corpus(recs, select_n=args.select_n, year_max=args.year_max)
        selected = set(report.selected_ids)
    stages = tuple(args.stages.split(",")) if args.stages else STAGES
    summary = run_pipeline(
        recs, backend, runs_dir=args.runs_dir, stages=stages,
        resume=not args.force, force=args.force, selected_ids=selected,
    )
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
    return 0 if summary.failed == 0 else 1


def cmd_validate_envs(args) -> int:
    from .validation import stage1_envquality
    return stage1_envquality.main(_rest(args)) or 0


def cmd_weightfree(args) -> int:
    from .validation import stage2_weightfree
    return stage2_weightfree.main(_rest(args)) or 0


def cmd_compare_harness(args) -> int:
    from .validation import stage3_harness
    return stage3_harness.main(_rest(args)) or 0


def cmd_stage4(args) -> int:
    from .validation import stage4_heldout
    return stage4_heldout.main(_rest(args)) or 0


def _rest(args) -> list[str]:
    # sub-argparse commands receive their raw argv tail via a passthrough
    return getattr(args, "passthrough", [])


def _passthrough_handlers() -> dict[str, Any]:
    """Commands whose argv must reach a sub-module verbatim.

    Derived from the parser itself — every sub-parser tagged with
    ``passthrough_cmd`` — so a command can never be added to the parser and
    forgotten here (or vice versa). The reason this dispatch exists at all:
    ``argparse.REMAINDER`` does not capture a *leading* ``--flag``, so
    ``bioforge validate-envs --out x`` used to die with
    "unrecognized arguments: --out" (exit 2) while the same parser happily
    swallowed ``bioforge stage4 check-contamination --lock L``. A selectively
    broken command is worse than an obviously broken one.
    """
    out: dict[str, Any] = {}
    for action in build_parser()._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, sub in action.choices.items():
            fn = sub._defaults.get("passthrough_cmd")
            if fn is not None:
                out[name] = fn
    return out


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="bioforge",
                                 description="BioForge-Harbor pipeline + validation protocol")
    ap.add_argument("--version", action="version", version=f"bioforge-harbor {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("curate", help="screen a corpus (paper 3.2 Curator)")
    p.add_argument("--corpus", required=True)
    p.add_argument("--select-n", type=int, default=200)
    p.add_argument("--year-max", type=int, default=2023)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_curate)

    r = sub.add_parser("run", help="run the full per-paper pipeline")
    r.add_argument("--corpus", required=True)
    r.add_argument("--runs-dir", default="runs")
    r.add_argument("--backend", default="mock", choices=["mock", "dryrun", "pi"])
    r.add_argument("--mock-responses", default=None)
    r.add_argument("--provider", default=None)
    r.add_argument("--model", default=None)
    r.add_argument("--stages", default=None, help="comma list subset of " + ",".join(STAGES))
    r.add_argument("--select-n", type=int, default=None)
    r.add_argument("--year-max", type=int, default=2023)
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_run)

    # The four validation commands take their own argv verbatim (see
    # _passthrough_handlers); the REMAINDER positional below is only what a
    # direct parse_args() call would produce, the CLI path bypasses it.
    v = sub.add_parser("validate-envs", help="Stage 1 (paper 4.1)")
    v.add_argument("passthrough", nargs=argparse.REMAINDER)
    v.set_defaults(fn=cmd_validate_envs, passthrough_cmd=cmd_validate_envs)

    w = sub.add_parser("run-weightfree", help="Stage 2 (paper 4.2)")
    w.add_argument("passthrough", nargs=argparse.REMAINDER)
    w.set_defaults(fn=cmd_weightfree, passthrough_cmd=cmd_weightfree)

    c = sub.add_parser("compare-harness", help="Stage 3 (paper 4.3)")
    c.add_argument("passthrough", nargs=argparse.REMAINDER)
    c.set_defaults(fn=cmd_compare_harness, passthrough_cmd=cmd_compare_harness)

    s = sub.add_parser("stage4", help="Stage 4 split/contamination (paper 4.4)")
    s.add_argument("passthrough", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_stage4, passthrough_cmd=cmd_stage4)

    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    handlers = _passthrough_handlers()
    if argv and argv[0] in handlers:
        return handlers[argv[0]](argparse.Namespace(passthrough=argv[1:]))
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
