"""Parallel multi-agent synthesis driver.

The stock runner processes papers serially; a live construct call is ~5-7
minutes, so 6 papers serial would be an hour of wall time. This driver runs
N papers concurrently — each worker thread gets its OWN TierflowBackend
(connection state must not be shared across threads; see backend docstring).

Output layout matches run_pipeline's contract:
  out/runs/<paper_id>/...   per-stage artifacts + state.json
  out/runs/tasks/...        packaged Harbor task dirs + manifest.jsonl

Usage:
  PYTHONPATH=src TIERFLOW_API_KEY=... python tools/synth_parallel.py \
      --corpus out/corpus_biorxiv_2026-09.json --select-n 6 --workers 3
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioforge_harbor.backends.tierflow_backend import TierflowBackend  # noqa: E402
from bioforge_harbor.models.records import PaperRecord  # noqa: E402
from bioforge_harbor.pipeline.runner import curate_corpus, run_pipeline  # noqa: E402
from bioforge_harbor.pipeline.state import STAGES  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs-dir", default="out/runs")
    ap.add_argument("--select-n", type=int, default=6)
    ap.add_argument("--year-max", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout-sec", type=float, default=420.0)
    args = ap.parse_args(argv)

    recs = [PaperRecord.from_dict(d)
            for d in json.load(open(args.corpus, encoding="utf-8"))["records"]]
    report = curate_corpus(recs, select_n=args.select_n, year_max=args.year_max)
    sel = set(report.selected_ids)
    todo = [r for r in recs if r.paper_id in sel]
    print(f"curated: {report.screened} screened, {report.selected} selected", flush=True)

    def work(rec: PaperRecord) -> tuple[str, str]:
        backend = TierflowBackend(timeout_sec=args.timeout_sec)
        try:
            summary = run_pipeline(
                [rec], backend,
                runs_dir=Path(args.runs_dir),
                stages=STAGES,
                resume=True, force=True,
                selected_ids={rec.paper_id},
            )
            return rec.paper_id, summary.per_paper.get(rec.paper_id, "?")
        except Exception as e:  # noqa: BLE001
            return rec.paper_id, f"error: {e}"

    results: dict[str, str] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r.paper_id for r in todo}
        for fut in cf.as_completed(futs):
            pid, status = fut.result()
            results[pid] = status
            print(f"[done] {pid} -> {status}", flush=True)

    Path("out/synth_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    packaged = sum(1 for s in results.values() if s == "packaged")
    print(f"packaged {packaged}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
