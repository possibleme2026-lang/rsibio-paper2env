"""Stage 2 runner: weight-free self-improvement (paper 4.2).

Protocol (RSI-Forge design): N models x K sequential attempts per environment,
inheriting code and notes across rounds, with NO weight updates. The scheduler
here is real: it emits the per-(model, env, round) execution plan, and either
(a) dry-runs it, (b) replays a results manifest, or (c) shells out to a user
harness command per attempt.

The harness command contract (option c): the runner invokes

    <harness_cmd> --model M --env DIR --round R --prev LOGDIR --out RESULT.json

and RESULT.json must contain {"reward": <float>}. How the harness runs a model
with inherited notes is the user's business — the framework only schedules,
collects, and computes the paper's three Stage-2 rates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .metrics import Attempt, stage2_summary
from ..util.subproc import run_to_file


def build_plan(models: list[str], envs: list[str], rounds: int) -> list[dict[str, Any]]:
    """Deterministic attempt order: env-major, then round, then model."""
    plan = []
    for env in sorted(envs):
        for r in range(1, int(rounds) + 1):
            for m in sorted(models):
                plan.append({"env": env, "round": r, "model": m})
    return plan


def replay_manifest(path: Path | str) -> list[Attempt]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(Attempt(model=d["model"], env=d["env"], round=int(d["round"]),
                           reward=float(d["reward"])))
    return out


def execute(harness_cmd: list[str], plan: list[dict[str, Any]], *,
            envs_dir: Path, logs_dir: Path, timeout_sec: float = 3600.0) -> list[Attempt]:
    """Run the user harness per planned attempt. Live mode only."""
    attempts: list[Attempt] = []
    logs_dir.mkdir(parents=True, exist_ok=True)
    for item in plan:
        result_path = logs_dir / f"{item['env']}__{item['model']}__r{item['round']}.json"
        argv = list(harness_cmd) + [
            "--model", item["model"],
            "--env", str(Path(envs_dir) / item["env"]),
            "--round", str(item["round"]),
            "--prev", str(logs_dir),
            "--out", str(result_path),
        ]
        res = run_to_file(argv, timeout_sec=timeout_sec)
        if not res.ok or not result_path.is_file():
            reward = 0.0   # failed rollout = zero reward, recorded not hidden
        else:
            reward = float(json.loads(result_path.read_text(encoding="utf-8")).get("reward", 0.0))
        attempts.append(Attempt(model=item["model"], env=item["env"],
                                round=item["round"], reward=reward))
    return attempts


def run(config: dict[str, Any], *, out: Path | str, dry_run: bool = False,
        replay: Path | str | None = None) -> dict[str, Any]:
    models = config["models"]
    envs = config["envs"]
    rounds = int(config.get("rounds", 3))
    baseline = {k: float(v) for k, v in (config.get("baseline") or {}).items()}
    plan = build_plan(models, envs, rounds)

    if dry_run:
        report = {"stage": 2, "dry_run": True, "planned_attempts": len(plan),
                  "plan_sample": plan[:6]}
    elif replay:
        attempts = replay_manifest(replay)
        report = stage2_summary(attempts, baseline)
    else:
        harness = config.get("harness_cmd")
        if not harness:
            raise ValueError("config needs harness_cmd (list) or --replay MANIFEST")
        attempts = execute(harness, plan, envs_dir=Path(config["envs_dir"]),
                           logs_dir=Path(config.get("logs_dir", "runs/stage2-logs")))
        report = stage2_summary(attempts, baseline)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="bioforge run-weightfree",
                                 description="Stage 2: weight-free self-improvement (paper 4.2)")
    ap.add_argument("--config", required=True, help="JSON: models, envs, rounds, baseline, harness_cmd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the execution plan, run nothing")
    ap.add_argument("--replay", default=None, help="JSONL of attempt results to score offline")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    rep = run(cfg, out=a.out, dry_run=a.dry_run, replay=a.replay)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
