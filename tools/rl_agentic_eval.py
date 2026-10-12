"""Two-arm agentic eval: base policy vs RL checkpoint, same rollout contract.

Reuses pi_grpo_train.run_one_rollout VERBATIM (same pi agent, same bridge,
same sandbox proxy injection) so the eval measures the policy, not the
harness. The only difference between arms is which weights the bridge
serves:
  arm base: bridge WITHOUT --model-path  -> openai/gpt-oss-20b base
  arm rl  : bridge WITH --model-path <tinker://ckpt>

Usage:
  TINKER_API_KEY=... .rlvenv/Scripts/python.exe tools/rl_agentic_eval.py \
      --rl-model tinker://<session>/sampler_weights/pi-grpo-full-12tasks-v2-step20 \
      --group-size 2 --rollout-concurrency 3

Output: out/rl_runs/agentic_eval/{summary.json, base/*, rl/*}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from pi_grpo_train import (  # noqa: E402  (same rollout contract as training)
    RolloutSpec, run_one_rollout, _resolve_harbor_bin,
)

log = logging.getLogger("agentic_eval")
OUT = REPO / "out" / "rl_runs" / "agentic_eval"


class RecordStoreStub:
    """Eval does not need trajectory assembly; kept for schema parity."""

    def __init__(self, path: Path):
        self.path = Path(path)


async def run_arm(
    arm: str,
    model_path: str | None,
    task_dirs: list[Path],
    group_size: int,
    *,
    bridge_port: int,
    max_turns: int,
    rollout_timeout: int,
    rollout_concurrency: int,
    harbor_bin: str,
    sandbox_proxy: str,
    records_dir: Path,
    out_dir: Path,
) -> list[dict]:
    """Start a bridge for this arm, roll out every task x group, stop bridge."""
    import tinker as tinker_sdk  # noqa: F401  (parity with trainer imports)

    # start the bridge as its own process (same as the trainer does)
    cmd = [
        sys.executable, str(REPO / "tools" / "tinker_bridge.py"),
        "--port", str(bridge_port),
        "--records-dir", str(records_dir),
        "--max-tokens", "8192",
    ]
    if model_path:
        cmd += ["--model-path", model_path]
    bridge_log = records_dir / "bridge_server.log"
    bridge_log.parent.mkdir(parents=True, exist_ok=True)
    fh = bridge_log.open("ab")
    log.info("[%s] bridge: %s", arm, " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=fh, stderr=asyncio.subprocess.STDOUT)
    await asyncio.sleep(4)  # uvicorn warm-up

    try:
        sem = asyncio.Semaphore(max(1, rollout_concurrency))

        async def _bounded(spec: RolloutSpec) -> dict:
            async with sem:
                return await run_one_rollout(
                    spec, bridge_port=bridge_port, model_id="openai/gpt-oss-20b",
                    max_turns=max_turns, timeout_sec=rollout_timeout,
                    harbor_bin=harbor_bin, sandbox_proxy=sandbox_proxy,
                    setup_timeout_multiplier=2.0, dry_run=False,
                )

        specs = [
            RolloutSpec(task_dir=d, task_name=d.name, group_idx=g,
                        rollout_key=uuid.uuid4().hex)
            for d in task_dirs for g in range(group_size)
        ]
        rows = await asyncio.gather(*(_bounded(s) for s in specs))
    finally:
        try:
            proc.terminate()
            await asyncio.wait_for(proc.communicate(), timeout=10)
        except (ProcessLookupError, asyncio.TimeoutError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    # persist per-rollout rows
    (out_dir / arm).mkdir(parents=True, exist_ok=True)
    (out_dir / arm / "rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return list(rows)


def summarize(rows: list[dict]) -> dict:
    scored = [r["reward"] for r in rows if r["reward"] is not None]
    per_task: dict[str, list[float]] = {}
    for r in rows:
        per_task.setdefault(r["task"], []).append(r["reward"])
    return {
        "n_rollouts": len(rows),
        "n_scored": len(scored),
        "mean_reward": sum(scored) / len(scored) if scored else None,
        "pass_rate_1": (sum(1 for x in scored if x >= 1.0) / len(scored)) if scored else None,
        "per_task_mean": {
            t: (sum(v) / len(v)) for t, v in sorted(per_task.items())},
        "n_failures": sum(1 for r in rows if r.get("error")),
    }


async def amain(a: argparse.Namespace) -> None:
    task_dirs = sorted(d for d in Path(a.task_dir).iterdir() if d.is_dir())
    if a.include:
        task_dirs = [d for d in task_dirs if d.name in set(a.include)]
    log.info("eval: %d tasks x group=%d per arm", len(task_dirs), a.group_size)

    harbor_bin = _resolve_harbor_bin(a.harbor_bin)
    arms: dict[str, str | None] = {"base": None}
    if a.rl_model:
        arms["rl"] = a.rl_model

    all_rows: dict[str, list[dict]] = {}
    # arms use different ports so they could in principle run after another
    port = a.bridge_port
    for arm, mp in arms.items():
        log.info("=== arm %s: %s ===", arm, mp or "base gpt-oss-20b")
        all_rows[arm] = await run_arm(
            arm, mp, task_dirs, a.group_size,
            bridge_port=port, max_turns=a.max_turns,
            rollout_timeout=a.rollout_timeout,
            rollout_concurrency=a.rollout_concurrency,
            harbor_bin=harbor_bin, sandbox_proxy=a.sandbox_proxy,
            records_dir=OUT / "bridge" / arm, out_dir=OUT,
        )
        port += 1

    summary = {arm: summarize(rows) for arm, rows in all_rows.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps({"summary": summary,
                    "meta": {"rl_model": a.rl_model,
                             "task_dir": str(a.task_dir),
                             "group_size": a.group_size}},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("summary -> %s", OUT / "summary.json")
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rl-model", default=None,
                   help="tinker:// checkpoint path; omit to eval base arm only")
    p.add_argument("--task-dir", type=Path,
                   default=REPO / "out" / "runs" / "tasks")
    p.add_argument("--include", nargs="*", default=None)
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--rollout-timeout", type=int, default=1800)
    p.add_argument("--rollout-concurrency", type=int, default=3)
    p.add_argument("--bridge-port", type=int, default=8093)
    p.add_argument("--harbor-bin", default="harbor")
    p.add_argument("--sandbox-proxy", default="http://host.docker.internal:7897")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
