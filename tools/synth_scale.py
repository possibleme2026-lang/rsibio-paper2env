"""Scale synthesis driver: run the five-stage pipeline over a whole corpus.

Differences from ``synth_parallel.py`` (which was a 6-paper proof of concept):

  * **Resumable per paper.** Each paper's state lives in
    ``out/runs/<paper_id>/state.json``; a re-run skips stages already ``ok``.
    A crashed 2-hour run costs one paper, not the whole batch.
  * **Bounded retries per paper.** A transient gateway timeout on the
    construct call is retried (with a fresh backend) up to ``--retries``
    times before the paper is recorded as ``error``.
  * **Per-paper worker thread + own backend.** TierflowBackend keeps per-call
    state (``self.calls``); the backend docstring says connection state must
    not be shared, so each worker constructs its own.
  * **Append-only progress ledger** (``out/synth_scale_progress.jsonl``) so a
    killed run still reports which papers completed and which did not.
  * **Honest outcome accounting.** A paper that fails verification is
    ``failed``, not silently retried into a pass. The final report separates
    ``packaged | failed | error | blocked``.

Output layout (matches run_pipeline's contract):
  out/runs/<paper_id>/<stage>/...  per-stage artifacts + state.json
  out/runs/tasks/<slug>/           packaged Harbor task dirs
  out/runs/tasks/manifest.jsonl    one line per packaged environment

Usage:
  TIERFLOW_API_KEY=... .rlvenv/Scripts/python.exe tools/synth_scale.py \
      --corpus out/corpus_wave3.json --select-n 100 --workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bioforge_harbor.backends.tierflow_backend import TierflowBackend  # noqa: E402
from bioforge_harbor.models.records import PaperRecord  # noqa: E402
from bioforge_harbor.pipeline.runner import curate_corpus, run_pipeline  # noqa: E402
from bioforge_harbor.pipeline.state import STAGES  # noqa: E402

LEDGER_LOCK = threading.Lock()


def _append_ledger(path: Path, entry: dict) -> None:
    with LEDGER_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_corpus(path: Path) -> list[PaperRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    recs = data["records"] if isinstance(data, dict) else data
    return [PaperRecord.from_dict(d) for d in recs]


def _paper_already_done(runs_dir: Path, pid: str) -> bool:
    """True when state.json shows all five stages ok/skipped."""
    st = runs_dir / pid / "state.json"
    if not st.is_file():
        return False
    try:
        data = json.loads(st.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    attempts = data.get("attempts") or {}
    return all(
        (attempts.get(s) or {}).get("status") in ("ok", "skipped") for s in STAGES
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs-dir", default="out/runs")
    ap.add_argument("--select-n", type=int, default=100)
    ap.add_argument("--year-max", type=int, default=2027)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--timeout-sec", type=float, default=600.0)
    ap.add_argument("--only-packaged", action="store_true",
                    help="run only papers whose construct stage already succeeded")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    ledger = REPO / "out" / "synth_scale_progress.jsonl"

    recs = _load_corpus(Path(args.corpus))
    report = curate_corpus(recs, select_n=args.select_n, year_max=args.year_max)
    sel = set(report.selected_ids)
    todo = [r for r in recs if r.paper_id in sel]
    print(f"[curate] screened={report.screened} selected={report.selected} "
          f"buckets={report.bucket_counts}", flush=True)
    if report.rejected:
        from collections import Counter
        print("[curate] reject reasons:",
              Counter(x["reason"] for x in report.rejected).most_common(6), flush=True)

    # Skip papers already fully done (crash-resume at the batch level).
    pending = [r for r in todo if not _paper_already_done(runs_dir, r.paper_id)]
    done = len(todo) - len(pending)
    print(f"[resume] {done} already complete, {len(pending)} to run", flush=True)

    def work(rec: PaperRecord) -> tuple[str, str]:
        last = ""
        for attempt in range(1, args.retries + 2):
            backend = TierflowBackend(timeout_sec=args.timeout_sec)
            try:
                summary = run_pipeline(
                    [rec], backend,
                    runs_dir=runs_dir, stages=STAGES,
                    resume=True, force=False,
                    selected_ids={rec.paper_id},
                )
                status = summary.per_paper.get(rec.paper_id, "?")
                if status == "packaged":
                    return rec.paper_id, status
                last = status
                # A genuine stage failure (verify/emit) will not be fixed by a
                # retry; only retry transient `error:` statuses.
                if not status.startswith("error"):
                    return rec.paper_id, status
            except Exception as e:  # noqa: BLE001
                last = f"error: {type(e).__name__}: {e}"
                (runs_dir / rec.paper_id).mkdir(parents=True, exist_ok=True)
                (runs_dir / rec.paper_id / "LAST_ERROR.txt").write_text(
                    traceback.format_exc(), encoding="utf-8")
            time.sleep(3 * attempt)
        return rec.paper_id, last

    results: dict[str, str] = {}
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r.paper_id for r in pending}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            pid, status = fut.result()
            results[pid] = status
            _append_ledger(ledger, {"paper_id": pid, "status": status,
                                    "ts": time.time()})
            el = time.time() - t0
            print(f"[{i}/{len(pending)}] {pid} -> {status}  "
                  f"({el/60:.1f}min elapsed)", flush=True)

    packaged = sum(1 for s in results.values() if s == "packaged")
    summary = {
        "corpus": args.corpus,
        "selected": len(todo),
        "newly_run": len(pending),
        "already_done": done,
        "packaged_now": packaged,
        "packaged_total": sum(
            1 for r in todo if _paper_already_done(runs_dir, r.paper_id)),
        "status_counts": {k: sum(1 for s in results.values() if s == k)
                          for k in set(results.values())},
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "per_paper": results,
    }
    out = REPO / "out" / "synth_scale_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[done] packaged_now={packaged} packaged_total={summary['packaged_total']}"
          f"/{len(todo)} elapsed={summary['elapsed_min']}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
