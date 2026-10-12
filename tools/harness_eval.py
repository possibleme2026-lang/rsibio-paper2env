"""Three-harness comparative eval on the synthesized bio task set.

Runs the SAME task directories under three different agent harnesses and
collects verifier rewards, so "spread across models" (Stage 3 of the paper's
protocol) can be separated from "spread across harnesses".

Harnesses
---------
  codex   : harbor's built-in Codex agent (OpenAI Codex CLI in-container),
            pointed at the tierflow gateway via OPENAI_BASE_URL/API_KEY.
  pi      : harbor's built-in Pi agent (pi-coding-agent), pointed at tierflow
            via a models.json written into the sandbox. Implemented by the
            ``tierflow_pi_agent:TierflowPi`` subclass shipped in tools/.
  dsh     : DeepSeek Harness CLI (``dsh``), installed from npm in-container and
            driven headless via its ``dsh exec`` one-shot mode. Implemented by
            ``tierflow_dsh_agent:TierflowDsh``.

All three talk to the same model id ``tierflow_pro`` so a difference in score
is attributable to the harness (tool surface, prompting, context management),
not to the underlying model.

Container networking
--------------------
On Windows the host proxy is 127.0.0.1:7897 but a container must address the
host as host.docker.internal:7897 (VERIFIED: container -> tierflow works only
through that name). We therefore override HTTP(S)_PROXY for the harbor
subprocess tree AND inject the same values into the container via --agent-env,
exactly as pi_grpo_train.run_one_rollout does.

Usage
-----
  TIERFLOW_API_KEY=... .rlvenv/Scripts/python.exe tools/harness_eval.py \
      --task-dir out/runs/tasks --harnesses codex,pi,dsh \
      --limit 100 --rollout-concurrency 3

Output: out/harness_eval/{summary.json, <harness>/<task>/harbor_stdout.log}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

log = logging.getLogger("harness_eval")

TIERFLOW_BASE_URL = "https://tierflow.cn/v1"
TIERFLOW_MODEL = "tierflow_pro"
DEFAULT_SANDBOX_PROXY = "http://host.docker.internal:7897"

OUT = REPO / "out" / "harness_eval"


def _harbor_bin() -> str:
    local = REPO / ".rlvenv" / "Scripts" / "harbor.exe"
    if local.is_file():
        return str(local)
    return shutil.which("harbor") or "harbor"


def _find_result_json(result_dir: Path) -> Path | None:
    def _has_rewards(p: Path) -> bool:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return isinstance(data, dict) and bool(
            (data.get("verifier_result") or {}).get("rewards"))

    candidates = [result_dir / "result.json", *sorted(result_dir.rglob("result.json"))]
    for hit in candidates:
        if hit.is_file() and _has_rewards(hit):
            return hit
    for hit in candidates:
        if hit.is_file():
            return hit
    return None


def _parse_reward(result_dir: Path, since: float | None = None) -> tuple[float, dict]:
    """Return (reward, rewards) for the newest *trial* under result_dir.

    harbor emits both a job-level ``result.json`` (no ``verifier_result``) and
    one per trial. Three traps this guards against:

    1. Picking the job-level file and reporting a passing trial as 0.0 -- so a
       file must actually carry rewards to be considered.
    2. Picking the *oldest* trial when several runs piled up under the same
       task dir: candidates are sorted newest-first and the newest scoring
       trial wins.
    3. Reading a stale trial from a previous run (leftovers the safe-delete
       guard would not let us remove): ``since`` (a wall-clock start time)
       restricts candidates to files this run just produced.
    """
    candidates = sorted(result_dir.rglob("result.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if since is not None and path.stat().st_mtime < since - 1.0:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rewards = (data.get("verifier_result") or {}).get("rewards") or {}
        if not rewards:
            continue
        if "reward" in rewards:
            return float(rewards["reward"]), rewards
        return float(next(iter(rewards.values()))), rewards
    return 0.0, {}


def _build_cmd(harness: str, task_dir: Path, jobs_dir: Path, *, model: str,
               setup_mult: float, proxy: str, api_key: str) -> list[str]:
    """Assemble the harbor argv for one (harness, task) trial."""
    cmd = [_harbor_bin(), "run", "--path", str(task_dir),
           "--jobs-dir", str(jobs_dir), "--n-concurrent", "1",
           "--agent-setup-timeout-multiplier", str(setup_mult)]

    if harness == "codex":
        # Stock harbor Codex cannot reach a third-party gateway: it ignores
        # OPENAI_BASE_URL for provider selection (dialing
        # wss://api.openai.com) and rejects wire_api="chat". TierflowCodex
        # writes a config.toml provider table with wire_api="responses".
        cmd += [
            "--agent-import-path", "tierflow_codex_agent:TierflowCodex",
            "--model", model,
            "--agent-env", "CODEX_FORCE_API_KEY=1",
            "--agent-env", f"OPENAI_API_KEY={api_key}",
            "--agent-env", f"OPENAI_BASE_URL={TIERFLOW_BASE_URL}",
            "--agent-kwarg", "reasoning_effort=low",
        ]
    elif harness == "pi":
        cmd += [
            "--agent-import-path", "tierflow_pi_agent:TierflowPi",
            "--model", f"openai/{model}",
            "--agent-env", f"OPENAI_API_KEY={api_key}",
            "--agent-env", f"PI_TIERFLOW_BASE_URL={TIERFLOW_BASE_URL}",
            "--agent-env", f"PI_TIERFLOW_MODEL={model}",
            "--agent-env", f"PI_TIERFLOW_API_KEY={api_key}",
        ]
    elif harness == "dsh":
        cmd += [
            "--agent-import-path", "tierflow_dsh_agent:TierflowDsh",
            "--model", model,
            "--agent-env", f"DEEPSEEK_API_KEY={api_key}",
            "--agent-env", f"DEEPSEEK_BASE_URL={TIERFLOW_BASE_URL}",
            "--agent-env", f"DSH_MODEL={model}",
        ]
    else:
        raise ValueError(f"unknown harness {harness}")

    if proxy:
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            cmd += ["--agent-env", f"{k}={proxy}"]
        cmd += ["--agent-env",
                f"NO_PROXY=localhost,127.0.0.1,host.docker.internal,http.docker.internal,tierflow.cn"]
    return cmd


async def run_trial(harness: str, task_dir: Path, *, model: str, timeout_sec: int,
                    setup_mult: float, out_root: Path, proxy: str,
                    api_key: str) -> dict:
    slug = task_dir.name
    jobs_dir = out_root / harness / slug
    jobs_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(harness, task_dir, jobs_dir, model=model,
                     setup_mult=setup_mult, proxy=proxy, api_key=api_key)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "tools"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    if proxy:
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[k] = proxy
        env["NO_PROXY"] = env["no_proxy"] = (
            "localhost,127.0.0.1,host.docker.internal,http.docker.internal,tierflow.cn")

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
    (jobs_dir / "harbor_stdout.log").write_text(tail, encoding="utf-8")

    reward, rewards = _parse_reward(jobs_dir, since=t0)
    err = ""
    if reward == 0.0:
        m = re.search(r"Trial .*? failed: ([^\n]+)", tail)
        err = m.group(1) if m else (tail.strip().splitlines()[-1] if tail.strip() else "no output")
    return {
        "harness": harness, "task": slug, "reward": reward, "rewards": rewards,
        "timed_out": timed_out, "exit": proc.returncode, "error": err[:300],
        "elapsed_sec": round(time.time() - t0, 1), "jobs_dir": str(jobs_dir),
    }


async def main_async(a) -> int:
    task_root = Path(a.task_dir)
    tasks = sorted([d for d in task_root.iterdir() if (d / "task.toml").is_file()])
    if a.task_list:
        wanted = {x.strip() for x in Path(a.task_list).read_text(
            encoding="utf-8").splitlines() if x.strip()}
        tasks = [t for t in tasks if t.name in wanted]
    if a.limit:
        tasks = tasks[:a.limit]
    harnesses = [h.strip() for h in a.harnesses.split(",") if h.strip()]
    api_key = a.api_key or os.environ.get("TIERFLOW_API_KEY", "")
    if not api_key:
        print("ERROR: no tierflow api key (--api-key or TIERFLOW_API_KEY)", flush=True)
        return 2

    print(f"[eval] {len(tasks)} tasks x {len(harnesses)} harnesses "
          f"({', '.join(harnesses)})", flush=True)
    sem = asyncio.Semaphore(max(1, a.rollout_concurrency))
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    # -- resume support ---------------------------------------------------
    # A 315-trial sweep takes hours; losing it to a crash at trial 300 is
    # unacceptable. Partial results stream to results.jsonl after every
    # trial, and previously completed (harness, task) pairs are skipped on
    # restart unless --force-rerun is given.
    #
    # "Completed" means the verifier actually produced a reward. A record with
    # an empty ``rewards`` dict did not run the verifier (image build or setup
    # failure) -- caching that as a 0.0 would deflate the harness score with an
    # infrastructure artifact, so those are re-attempted on resume.
    partial_path = OUT / "results.jsonl"
    cached: dict[tuple[str, str], dict] = {}
    if partial_path.is_file() and not a.force_rerun:
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("rewards") and not rec.get("timed_out"):
                cached[(rec.get("harness", ""), rec.get("task", ""))] = rec
        if cached:
            print(f"[eval] resuming: {len(cached)} scored trials already recorded",
                  flush=True)
        results.extend(cached.values())

    def _append_partial(rec: dict) -> None:
        with partial_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    async def one(h: str, t: Path) -> tuple[dict, bool]:
        """Return (record, is_cached)."""
        hit = cached.get((h, t.name))
        if hit is not None:
            print(f"[eval] skip (cached) {h}/{t.name} -> {hit['reward']}",
                  flush=True)
            return hit, True
        async with sem:
            log.info("[%s] %s", h, t.name)
            try:
                r = await run_trial(h, t, model=a.model, timeout_sec=a.timeout_sec,
                                    setup_mult=a.setup_mult, out_root=OUT,
                                    proxy=a.sandbox_proxy, api_key=api_key)
            except Exception as e:  # noqa: BLE001
                r = {"harness": h, "task": t.name, "reward": 0.0, "rewards": {},
                     "error": f"{type(e).__name__}: {e}", "timed_out": False,
                     "exit": -1, "elapsed_sec": 0.0, "jobs_dir": ""}
            print(f"[{h}] {t.name} -> reward={r['reward']} ({r['elapsed_sec']}s)"
                  + (f" ERR {r.get('error','')[:80]}" if r["reward"] == 0.0 else ""),
                  flush=True)
            return r, False

    jobs = [one(h, t) for h in harnesses for t in tasks]
    n_total = len(jobs)
    n_done = 0
    for coro in asyncio.as_completed(jobs):
        r, is_cached = await coro
        n_done += 1
        if is_cached:
            continue
        # Persist only trials whose verifier produced a reward: a timeout or a
        # setup failure may succeed on retry, so re-running is the safe default.
        if r.get("rewards") and not r.get("timed_out"):
            _append_partial(r)
        results.append(r)
        print(f"[eval] progress {n_done}/{n_total}", flush=True)

    summary = _summarize(results, harnesses, tasks)
    (OUT / "summary.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    _print_summary(summary)
    return 0


def _summarize(results: list[dict], harnesses: list[str], tasks: list[Path]) -> dict:
    def _stats(h: str) -> dict:
        rs = [r for r in results if r["harness"] == h]
        rew = [r["reward"] for r in rs]
        n = len(rew) or 1
        nz = [x for x in rew if x > 0]
        return {
            "trials": len(rs),
            "mean_reward": round(sum(rew) / n, 4),
            "nonzero": len(nz),
            "nonzero_rate": round(len(nz) / n, 4),
            "max_reward": round(max(rew), 4) if rew else 0.0,
            "solved_ge_0.5": sum(1 for x in rew if x >= 0.5),
            "errors": sum(1 for r in rs if r.get("error")),
            "timeouts": sum(1 for r in rs if r.get("timed_out")),
        }

    by_h = {h: _stats(h) for h in harnesses}
    # per-task winner (for a paired comparison, only on tasks where >=2 harnesses ran)
    per_task: dict[str, dict[str, float]] = {}
    for t in tasks:
        row = {r["harness"]: r["reward"] for r in results if r["task"] == t.name}
        if row:
            per_task[t.name] = row
    paired = {k: v for k, v in per_task.items() if len(v) == len(harnesses)}
    wins = {h: 0 for h in harnesses}
    ties = 0
    for v in paired.values():
        best = max(v.values())
        winners = [h for h, x in v.items() if x == best]
        if len(winners) > 1:
            ties += 1
        else:
            wins[winners[0]] += 1
    return {
        "harnesses": by_h,
        "paired_tasks": len(paired),
        "head_to_head_wins": wins,
        "ties": ties,
        "per_task": per_task,
    }


def _print_summary(s: dict) -> None:
    print("\n=== harness comparison ===", flush=True)
    print(f"{'harness':10} {'trials':>6} {'mean':>8} {'nonzero':>8} "
          f"{'>=0.5':>6} {'max':>6} {'err':>4}")
    for h, st in s["harnesses"].items():
        print(f"{h:10} {st['trials']:>6} {st['mean_reward']:>8} "
              f"{st['nonzero']:>8} {st['solved_ge_0.5']:>6} "
              f"{st['max_reward']:>6} {st['errors']:>4}")
    print(f"paired tasks: {s['paired_tasks']}  wins: {s['head_to_head_wins']}  "
          f"ties: {s['ties']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default="out/runs/tasks")
    ap.add_argument("--task-list", default="",
                    help="file with one task slug per line (subset to run)")
    ap.add_argument("--harnesses", default="codex,pi,dsh")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=TIERFLOW_MODEL)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--sandbox-proxy", default=DEFAULT_SANDBOX_PROXY)
    ap.add_argument("--rollout-concurrency", type=int, default=3)
    ap.add_argument("--timeout-sec", type=int, default=2400)
    ap.add_argument("--setup-mult", type=float, default=2.0)
    ap.add_argument("--force-rerun", action="store_true",
                    help="ignore cached results.jsonl and rerun every trial")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s")
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    raise SystemExit(main())
