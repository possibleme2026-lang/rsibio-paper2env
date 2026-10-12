"""Oracle validation sweep over the synthesized task set.

Why this exists
---------------
A synthesized task is only usable as an eval item if its *own* reference
solution passes its verifier. Without that check, "the agent scored 0" is
ambiguous: the harness may have failed, or the task may simply be unsolvable /
the verifier mis-specified. The oracle agent runs ``solution/solve.sh`` inside
the task's container and then the verifier, so it isolates task validity from
agent capability -- the authoritative "is this task well-formed" gate.

Runs are independent of and much cheaper than harness trials, so this sweep is
meant to finish before the harness comparison.

Usage
-----
    .rlvenv/Scripts/python.exe tools/validate_tasks.py \
        --task-dir out/runs/tasks --limit 0 --concurrency 4

Output: out/task_validation/{summary.json, <task>/oracle.log}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out" / "task_validation"


def _harbor_bin() -> str:
    local = REPO / ".rlvenv" / "Scripts" / "harbor.exe"
    if local.is_file():
        return str(local)
    return shutil.which("harbor") or "harbor"


def _parse(jobs_dir: Path) -> tuple[float, str]:
    """Return (reward, error) for the newest *trial* under jobs_dir.

    harbor writes two kinds of ``result.json``: a job-level file
    (``<jobs-dir>/<run-ts>/result.json``, which has no ``verifier_result``) and
    one per trial (``<jobs-dir>/<run-ts>/<trial>/result.json``). Picking by
    mtime alone silently selects the job-level file and reports a passing
    trial as "verifier did not run" -- a measurement artifact, not a task
    defect. So we explicitly require a file that carries ``verifier_result``.
    """
    def _load(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    all_results = sorted(jobs_dir.rglob("result.json"), key=lambda p: p.stat().st_mtime)
    if not all_results:
        return 0.0, "no result.json"

    with_rewards = []
    for path in all_results:
        data = _load(path)
        if not data:
            continue
        rewards = (data.get("verifier_result") or {}).get("rewards")
        if rewards:
            with_rewards.append((path, data, rewards))

    if with_rewards:
        _path, _data, rewards = with_rewards[-1]
        val = rewards.get("reward", next(iter(rewards.values())))
        return float(val), ""

    # No trial produced rewards: surface the newest exception for diagnosis.
    # The console summary only says "RuntimeError", so prefer the trial's own
    # exception.txt -- that is where the actionable line (apt failure,
    # generator traceback, pydantic ValidationError) actually lives.
    exc_files = sorted(jobs_dir.rglob("exception.txt"),
                       key=lambda p: p.stat().st_mtime)
    if exc_files:
        text = exc_files[-1].read_text(encoding="utf-8", errors="replace")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in reversed(lines):
            if re.search(r"(Error|Exception|failed to solve|has no installation"
                         r" candidate|E: )", ln):
                return 0.0, ln[:300]
        if lines:
            return 0.0, lines[-1][:300]

    newest = _load(all_results[-1]) or {}
    exc = (newest.get("exception_info") or {}).get("exception_message") or ""
    return 0.0, (exc or "no rewards (verifier did not run)")


async def run_one(task_dir: Path, *, timeout_sec: int, setup_mult: float,
                  timeout_mult: float, proxy: str, sem: asyncio.Semaphore) -> dict:
    slug = task_dir.name
    jobs_dir = OUT / slug
    jobs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [_harbor_bin(), "run", "--path", str(task_dir),
           "--jobs-dir", str(jobs_dir), "--n-concurrent", "1",
           "--agent", "oracle", "--yes",
           "--agent-setup-timeout-multiplier", str(setup_mult),
           "--timeout-multiplier", str(timeout_mult)]

    env = dict(os.environ)
    if proxy:
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[k] = proxy
        env["NO_PROXY"] = env["no_proxy"] = (
            "localhost,127.0.0.1,host.docker.internal,http.docker.internal,tierflow.cn")

    async with sem:
        t0 = time.time()
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, env=env)
        timed_out = False
        try:
            out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            timed_out = True
            proc.kill()
            out_b, _ = await proc.communicate()

    tail = (out_b or b"").decode(errors="replace")
    (jobs_dir / "oracle.log").write_text(tail, encoding="utf-8")
    reward, err = _parse(jobs_dir)
    if timed_out:
        err = err or "timeout"
    status = "ok" if reward > 0 else ("timeout" if timed_out else "fail")
    print(f"[oracle] {slug} -> reward={reward} {status} "
          f"({round(time.time() - t0, 1)}s)" + (f"  {err[:90]}" if err else ""),
          flush=True)
    return {"task": slug, "reward": reward, "status": status, "error": err[:300],
            "timed_out": timed_out, "elapsed_sec": round(time.time() - t0, 1)}


async def main_async(a) -> int:
    task_root = Path(a.task_dir)
    tasks = sorted(d for d in task_root.iterdir() if (d / "task.toml").is_file())
    if a.task_list:
        wanted = {x.strip() for x in Path(a.task_list).read_text(
            encoding="utf-8").splitlines() if x.strip()}
        tasks = [t for t in tasks if t.name in wanted]
    if a.limit:
        tasks = tasks[:a.limit]
    if not tasks:
        print("no tasks found", file=sys.stderr)
        return 2

    print(f"[validate] {len(tasks)} tasks, concurrency={a.concurrency}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # -- resume support ----------------------------------------------------
    # The oracle sweep is the longest phase (one Docker container per task)
    # and previously only wrote summary.json at the very end -- losing the
    # sweep to a crash at task 90 of 100 was unacceptable. "Completed" means
    # status == ok: a failed oracle run may pass on retry, so caching a
    # failure as final would hide real defects from the eval set.
    partial_path = OUT / "oracle_results.jsonl"
    cached: dict[str, dict] = {}
    if partial_path.is_file() and not a.force_rerun:
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok":
                cached[rec.get("task", "")] = rec
        if cached:
            print(f"[validate] resuming: {len(cached)} oracle-verified tasks",
                  flush=True)

    def _append_partial(rec: dict) -> None:
        with partial_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    async def _one_wrapped(t: Path):
        hit = cached.get(t.name)
        if hit is not None:
            print(f"[validate] skip (cached) {t.name}", flush=True)
            return hit
        r = await run_one(t, timeout_sec=a.timeout_sec, setup_mult=a.setup_mult,
                          timeout_mult=a.timeout_mult, proxy=a.proxy, sem=sem)
        if r["status"] == "ok":
            _append_partial(r)
        return r

    sem = asyncio.Semaphore(max(1, a.concurrency))
    results = await asyncio.gather(*[_one_wrapped(t) for t in tasks])

    # Stub-solution exclusion: a reference solution that writes only 'END'
    # (kukulu-cdr-reconstruction) can never produce a gold trace for any
    # harness. Exclude such tasks from the pass list with the evidence kept.
    STUB_EXCLUDE = {
        "kukulu-cdr-reconstruction":
            "reference solution writes only 'END' (placeholder, per its own "
            "comment 'For brevity, write a placeholder output'); grader then "
            "crashes on the empty structure -> no harness can ever score it",
    }
    for r in results:
        if r["task"] in STUB_EXCLUDE and r["status"] == "ok":
            # even a (surprising) pass is unusable: the oracle output is a
            # placeholder, so the eval would reward garbage
            r["status"] = "excluded_stub"

    ok = [r for r in results if r["status"] == "ok"]
    bad = [r for r in results if r["status"] not in ("ok", "excluded_stub")]
    stub = [r for r in results if r["status"] == "excluded_stub"]
    for r in stub:
        r["excluded_reason"] = STUB_EXCLUDE.get(r["task"], "stub solution")
    summary = {
        "total": len(results),
        "oracle_pass": len(ok),
        "oracle_fail": len(bad),
        "excluded_stub": [r["task"] for r in stub],
        "pass_rate": round(len(ok) / len(results), 4) if results else 0.0,
        "failed_tasks": [r["task"] for r in bad],
        "results": results,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # a plain slug list is what the eval phase consumes (plus stub exclusions
    # recorded separately so the reason survives)
    (OUT / "oracle_pass.txt").write_text(
        "\n".join(sorted(r["task"] for r in ok)) + "\n", encoding="utf-8")
    (OUT / "oracle_excluded.txt").write_text(
        "\n".join(f"{r['task']}\t{r.get('excluded_reason','')}" for r in stub)
        + ("\n" if stub else ""), encoding="utf-8")
    print(f"\n=== oracle validation ===\n"
          f"pass {len(ok)}/{len(results)} ({summary['pass_rate']:.1%}), "
          f"excluded_stub={len(stub)}", flush=True)
    if bad:
        print("failing: " + ", ".join(r["task"] for r in bad[:20]), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default="out/runs/tasks")
    ap.add_argument("--task-list", default="",
                    help="file with one task slug per line (subset to run)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout-sec", type=int, default=1800)
    ap.add_argument("--setup-mult", type=float, default=4.0)
    ap.add_argument("--timeout-mult", type=float, default=3.0)
    ap.add_argument("--proxy", default="http://127.0.0.1:7897")
    ap.add_argument("--force-rerun", action="store_true",
                    help="ignore cached oracle_results.jsonl and rerun every task")
    a = ap.parse_args()
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    raise SystemExit(main())
