"""pi + GRPO: train the policy INSIDE the pi agent scaffold via harbor + a
tinker sampling bridge.

ARCHITECTURE (per training step)
--------------------------------
  1. save_weights_for_sampler  -> tinker:// checkpoint of the current policy
  2. restart the bridge pointed at the new checkpoint
  3. for each task x group_size: harbor run --agent pi with the model routed
     to the bridge (PI_BRIDGE_KEY=<rollout-uuid>); pi works in the Docker
     sandbox (bash/read/edit/grep), the bridge records token ids + logprobs
  4. parse verifier reward from harbor's result.json per rollout
  5. rebuild cookbook Trajectory objects from the bridge records
  6. GRPO: compute_advantages (group-centered) + assemble_training_data
  7. forward_backward("importance_sampling") + optim_step
  8. every --save-every steps: keep a named checkpoint for eval

The policy sampled by the bridge IS the trained model (same renderer), so
the recorded (prompt_ids, completion_ids, logprobs) are valid on-policy
GRPO datums - pi only decides *what to ask*, never *how to sample*.

USAGE
-----
  TINKER_API_KEY=... .rlvenv/Scripts/python.exe tools/pi_grpo_train.py \
      --task-dir out/runs/tasks --steps 20 --group-size 4 --save-every 5

  # full offline rehearsal, no tinker / no docker / no pi:
  .rlvenv/Scripts/python.exe tools/pi_grpo_train.py --dry-run --steps 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from tinker_bridge import RecordStore  # noqa: E402

log = logging.getLogger("pi_grpo")

BRIDGE_HOST = "host.docker.internal"


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

@dataclass
class RolloutSpec:
    task_dir: Path
    task_name: str
    group_idx: int
    rollout_key: str          # = PI_BRIDGE_KEY, unique per rollout
    result_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp()))


def _resolve_harbor_bin(explicit: str | None) -> str:
    """The venv Scripts dir is not on PATH; resolve harbor robustly.
    A bare name (the argparse default "harbor") still needs resolution,
    because CreateProcess does not search the venv Scripts dir."""
    if explicit and (os.path.sep in explicit or Path(explicit).is_file()):
        return explicit  # user passed a real path
    hit = shutil.which(explicit or "harbor")
    if hit:
        return hit
    local = REPO / ".rlvenv" / "Scripts" / "harbor.exe"
    if local.is_file():
        return str(local)
    return explicit or "harbor"  # let CreateProcess produce the real error


async def run_one_rollout(
    spec: RolloutSpec,
    *,
    bridge_port: int,
    model_id: str,
    max_turns: int,
    timeout_sec: int,
    harbor_bin: str,
    sandbox_proxy: str = "",
    setup_timeout_multiplier: float = 2.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One harbor trial with pi as the agent and the bridge as the model."""
    out_dir = spec.result_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --agent-import-path: our BridgePi subclass writes models.json in the
    # sandbox (model-level baseUrl -> bridge) and runs pi against it, so the
    # stock harbor 0.3.0 pi provider whitelist is bypassed entirely.
    # PI_BRIDGE_KEY doubles as (a) pi's apiKey via env-name resolution and
    # (b) the bridge's record key = the rollout uuid.
    # harbor 0.3.0 CLI: local task dir goes to --path (NOT --task, which is
    # a registry ref 'org/name'); --jobs-dir (not --output-dir); results land
    # at <jobs-dir>/<job>/<trial>/result.json (found recursively).
    cmd = [
        harbor_bin, "run",
        "--path", str(spec.task_dir),
        "--agent-import-path", "rl_pi_agent:BridgePi",
        "--agent-kwarg", f"bridge_port={bridge_port}",
        "--agent-kwarg", "max_tokens=8192",
        "--model", "gpt-oss-rl",
        "--jobs-dir", str(out_dir),
        "--agent-env", f"PI_BRIDGE_KEY={spec.rollout_key}",
        "--agent-env", f"PI_RL_BRIDGE_PORT={bridge_port}",
    ]
    if sandbox_proxy:
        # Injected into the container for the agent-install apt-get and the
        # verifier's pip run. host.docker.internal is the container-side name
        # for the host (VERIFIED reachable from a container).
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            cmd += ["--agent-env", f"{k}={sandbox_proxy}"]
        cmd += ["--agent-env",
                "NO_PROXY=tinker.thinkingmachines.ai,localhost,127.0.0.1,"
                "host.docker.internal,http.docker.internal"]
    cmd += [
        # VERIFIED TIMINGS (isolated container, via the LAN proxy):
        # apt 107s + nvm/node 55s + npm pi 147s = ~310s for the agent setup
        # chain alone; harbor's default setup budget is 360s and ANY concurrent
        # load on the proxy pushes the chain past it -> AgentSetupTimeoutError.
        # Double the setup budget; the task-level timeout stays untouched.
        "--agent-setup-timeout-multiplier", str(setup_timeout_multiplier),
        # harbor defaults to -n 4 concurrent trials PER job; each of our
        # rollout jobs has exactly 1 trial, so the real concurrency knob is
        # --rollout-concurrency below (semaphore over harbor subprocesses).
        "--n-concurrent", "1",
    ]
    if dry_run:
        # offline rehearsal: no harbor/docker/pi at all. Fake a result.json
        # + a fake bridge record so the whole downstream assembly runs.
        await _write_dry_run_artifacts(spec, out_dir, model_id)
        return {"key": spec.rollout_key, "task": spec.task_name,
                "group_idx": spec.group_idx, "reward": _dry_run_reward(spec),
                "exit": 0, "result_dir": str(out_dir)}

    log.info("[%s] harbor run pi (%s/g%d)", spec.rollout_key[:8],
             spec.task_name, spec.group_idx)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "tools"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # The proxy must be addressed differently from INSIDE a container than
    # from the host. On Windows the host resolves "host.docker.internal" to a
    # LAN IP where the proxy does NOT listen, while 127.0.0.1 is the host
    # loopback (unreachable from the container). So we override the proxy ONLY
    # for the harbor subprocess tree, which does its networking in containers:
    #  - docker compose build: BuildKit rewrites the client's HTTP_PROXY /
    #    HTTPS_PROXY into implicit build args (VERIFIED: a no-proxy `pip
    #    install pandas` times out, the proxied one builds and imports pandas).
    #  - `docker compose exec` (agent install / verifier): forwarded to the
    #    container process by harbor via --agent-env (see --ae below).
    # The host-side tinker calls never go through this env, so the real tinker
    # traffic still uses NO_PROXY=... + 127.0.0.1 as before.
    if sandbox_proxy:
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[k] = sandbox_proxy
        # both spellings: containers reach the host bridge and Docker's
        # internal proxy by NAME; adding the host loopback name too is
        # harmless (it simply will not resolve inside the container).
        env["NO_PROXY"] = "tinker.thinkingmachines.ai,localhost,127.0.0.1,host.docker.internal,http.docker.internal"
        env["no_proxy"] = env["NO_PROXY"]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    timed_out = False
    try:
        out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        out_b, _ = await proc.communicate()
        log.warning("[%s] harbor run timed out after %ds", spec.rollout_key[:8], timeout_sec)

    # PERSIST harbor stdout/stderr: without this, a rollout that dies before
    # writing a job dir (docker build / agent setup failure) is completely
    # invisible and silently scores reward 0 - exactly how a whole training
    # run burned an hour on container-network failures.
    tail = (out_b or b"").decode(errors="replace")
    (out_dir / "harbor_stdout.log").write_text(tail, encoding="utf-8")

    reward = _parse_reward(out_dir)
    err = ""
    if reward == 0.0:
        m = re.search(r"Trial .*? failed: ([^\n]+)", tail)
        err = (m.group(1) if m else tail.strip().splitlines()[-1] if tail.strip() else "no output")
    return {"key": spec.rollout_key, "task": spec.task_name,
            "group_idx": spec.group_idx, "reward": reward,
            "exit": proc.returncode, "timed_out": timed_out,
            "error": err[:300], "result_dir": str(out_dir)}


def _find_result_json(result_dir: Path) -> Path | None:
    """harbor 0.3.0 writes <jobs-dir>/<job>/<trial>/result.json; find the
    trial-level one. Both job and trial dirs contain a result.json and only
    the trial one carries verifier rewards, so select by content."""

    def _has_rewards(p: Path) -> bool:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if not isinstance(data, dict):
            return False  # whole-file null / list / scalar
        return bool((data.get("verifier_result") or {}).get("rewards"))

    candidates = [result_dir / "result.json", *sorted(result_dir.rglob("result.json"))]
    for hit in candidates:
        if hit.is_file() and _has_rewards(hit):
            return hit  # trial-level result with actual rewards wins
    for hit in candidates:
        if hit.is_file():
            return hit  # fall back to the first parseable one
    return None


def _parse_reward(result_dir: Path) -> float:
    rj = _find_result_json(result_dir)
    if rj is not None:
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
            # verifier_result can be explicitly null (job-level result with
            # no verifier phase) - .get(k, {}) returns None for an existing
            # null key, so coerce with `or {}`.
            rewards = (data.get("verifier_result") or {}).get("rewards") or {}
            # VerifierResult.rewards is a flat dict[str, float]; prefer the
            # canonical "reward" key, else the first entry.
            if "reward" in rewards:
                return float(rewards["reward"])
            if rewards:
                return float(next(iter(rewards.values())))
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            pass
    return 0.0


async def _write_dry_run_artifacts(
    spec: RolloutSpec, out_dir: Path, model_id: str
) -> None:
    (out_dir / "result.json").write_text(json.dumps({
        "verifier_result": {"rewards": {"reward": _dry_run_reward(spec)}},
    }), encoding="utf-8")


def _dry_run_reward(spec: RolloutSpec) -> float:
    # deterministic pseudo-rewards: varies across group => non-zero advantage
    h = hash(f"{spec.task_name}:{spec.group_idx}") % 100
    return round(0.2 + 0.01 * h, 4)


# ---------------------------------------------------------------------------
# trajectory assembly from bridge records
# ---------------------------------------------------------------------------

def records_to_trajectory(records: list[dict[str, Any]], reward: float):
    """Bridge sampling records -> cookbook Trajectory (one Transition per call).

    Records whose logprobs are missing/None are DROPPED (with a warning):
    importance_sampling loss needs per-token logprobs, and one bad sampling
    call must not poison the whole training step. Import of cookbook types is
    deferred so --dry-run works on the default interpreter (no torch there).
    """
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.types import Trajectory, Transition
    import tinker as tinker_sdk

    usable = [r for r in records if r.get("logprobs") is not None]
    if len(usable) < len(records):
        log.warning("dropped %d/%d sampling records without logprobs",
                    len(records) - len(usable), len(records))
    transitions = []
    for i, rec in enumerate(usable):
        prompt_ids = rec["prompt_ids"]
        completion_ids = rec["completion_ids"]
        logprobs = rec["logprobs"]
        transitions.append(Transition(
            ob=tinker_sdk.ModelInput.from_ints(prompt_ids),
            ac=TokensWithLogprobs(
                tokens=completion_ids,
                maybe_logprobs=logprobs,
                stop_reason=("stop" if rec.get("finish_reason") != "length" else "length"),
            ),
            reward=(reward if i == len(usable) - 1 else 0.0),
            episode_done=(i == len(usable) - 1),
        ))
    if not transitions:
        return None
    return Trajectory(
        transitions=transitions,
        final_ob=tinker_sdk.ModelInput.from_ints([]),
        stop_reason="max_turns",
    )


# ---------------------------------------------------------------------------
# the training loop
# ---------------------------------------------------------------------------

class PiGRPOTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.records = RecordStore(Path(args.records_dir))
        self._bridge_process: asyncio.subprocess.Process | None = None
        self._bridge_log_fh = None
        self._sampling_model_path: str | None = None
        self.training_client: Any = None
        self.service_client: Any = None
        self.saved_checkpoints: list[str] = []
        # every step's sampler weight path, so a mid-run kill still leaves a
        # usable tinker:// checkpoint trail (each step calls
        # save_weights_for_sampler; losing the PATHS was the 2026-10-11 loss).
        self.step_sampler_paths: list[dict[str, Any]] = []
        self._history_cache: list[dict[str, Any]] = []

    # -- crash-resilient history --------------------------------------------
    def _flush_history(self, history: list[dict[str, Any]]) -> None:
        """Write history.json after EVERY step.

        Previously history was only written once, after the loop, so a kill
        or a machine sleep mid-run lost the entire curve. Written to a temp
        file then os.replace'd so a concurrent reader never sees a torn file.
        """
        if history is not None:
            self._history_cache = history
        out = Path(self.args.log_path)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_name": self.args.run_name,
            "steps_done": len(history),
            "steps": history,
            "checkpoints": self.saved_checkpoints,
            "step_samplers": self.step_sampler_paths,
        }
        tmp = out / "history.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, out / "history.json")

    # -- bridge lifecycle ---------------------------------------------------
    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process | None) -> None:
        if process is None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            pass  # bridge already exited (e.g. crashed on startup)
        try:
            await asyncio.wait_for(process.communicate(), timeout=10)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.communicate()

    async def _restart_bridge(self) -> int:
        await self._terminate(self._bridge_process)
        self._bridge_process = None

        port = self.args.bridge_port
        cmd = [
            sys.executable, str(REPO / "tools" / "tinker_bridge.py"),
            "--model", self.args.model,
            "--port", str(port),
            "--records-dir", self.args.records_dir,
            "--max-tokens", str(self.args.max_tokens),
        ]
        if self._sampling_model_path:
            cmd += ["--model-path", self._sampling_model_path]
        log.info("bridge: %s", " ".join(cmd))
        # stdout must NOT be a PIPE nobody reads: the bridge logs 2-3 lines
        # per sampling call and would block once the OS pipe buffer fills
        # mid-run. Log to a file instead (also survives for postmortem).
        bridge_log = Path(self.args.records_dir) / "bridge_server.log"
        bridge_log.parent.mkdir(parents=True, exist_ok=True)
        log_fh = bridge_log.open("ab")
        self._bridge_log_fh = log_fh  # keep ref so it is not GC-closed
        self._bridge_process = await asyncio.create_subprocess_exec(
            *cmd, stdout=log_fh, stderr=asyncio.subprocess.STDOUT
        )
        await asyncio.sleep(3)  # uvicorn warm-up
        return port

    async def stop_bridge(self) -> None:
        await self._terminate(self._bridge_process)
        self._bridge_process = None

    # -- one training step ---------------------------------------------------
    async def train_step(self, step: int, task_batch: list[Path]) -> dict[str, Any]:
        a = self.args
        # 1) checkpoint the current policy for sampling (skip in dry-run)
        if not a.dry_run:
            # tinker SDK double-await: the *_async call returns a future handle,
            # awaiting it yields an AwaitableConcurrentFuture whose
            # result_async() resolves to the actual response (resp.path).
            fut = await self.training_client.save_weights_for_sampler_async(
                name=f"pi-grpo-step{step}")
            resp = await fut.result_async()
            self._sampling_model_path = resp.path
            self.step_sampler_paths.append({"step": step, "path": resp.path})
            log.info("step %d: saved sampler weights -> %s", step, self._sampling_model_path)

        # 2) bridge serves the *current* policy. In dry-run there is no bridge
        # (and no harbor/pi/tinker at all) - skip the subprocess entirely.
        if a.dry_run:
            port = self.args.bridge_port
        else:
            port = await self._restart_bridge()

        # 3) roll out every task x group_size, bounded by a semaphore:
        # each rollout is its own harbor job with 1 trial x (2 cpus, 4 GB),
        # so the container count == the number of in-flight harbor
        # subprocesses. 16 unbounded jobs = 64 GB of requested memory on a
        # 15.7 GB host -> OOM/swap storms; cap it explicitly.
        specs = [
            RolloutSpec(
                task_dir=d, task_name=d.name, group_idx=g,
                rollout_key=uuid.uuid4().hex,
            )
            for d in task_batch for g in range(a.group_size)
        ]
        sem = asyncio.Semaphore(max(1, getattr(a, "rollout_concurrency", 3) or 3))

        async def _bounded(spec: RolloutSpec) -> dict[str, Any]:
            async with sem:
                return await run_one_rollout(
                    spec, bridge_port=port, model_id=a.model, max_turns=a.max_turns,
                    timeout_sec=a.rollout_timeout,
                    harbor_bin=_resolve_harbor_bin(a.harbor_bin),
                    sandbox_proxy=getattr(a, "sandbox_proxy", "") or "",
                    setup_timeout_multiplier=getattr(a, "setup_timeout_multiplier", 2.0),
                    dry_run=a.dry_run,
                )

        results = await asyncio.gather(*(_bounded(s) for s in specs))

        # 4) assemble trajectories: group = same task
        from tinker_cookbook.rl.types import TrajectoryGroup
        groups: list[TrajectoryGroup] = []
        rewards_seen: list[float] = []
        failures = [r for r in results if r.get("reward") == 0.0]
        if failures:
            log.warning("step %d: %d/%d rollouts scored 0 (e.g. %s)",
                        step, len(failures), len(results),
                        "; ".join(sorted({r.get("error", "")[:120] for r in failures if r.get("error")})[:3])
                        or "no error text captured")
        by_task: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            by_task.setdefault(r["task"], []).append(r)
        for task_name, rows in by_task.items():
            trajectories = []
            for r in rows:
                recs = self.records.read(r["key"]) if not a.dry_run else _fake_records(r)
                traj = records_to_trajectory(recs, r["reward"])
                if traj is not None:
                    trajectories.append(traj)
                rewards_seen.append(r["reward"])
            if trajectories:
                groups.append(TrajectoryGroup(
                    trajectories_G=trajectories,
                    final_rewards_G=[t.transitions[-1].reward for t in trajectories],
                    metrics_G=[{} for _ in trajectories],
                ))

        if not groups:
            log.warning("step %d: no usable trajectories", step)
            return {"step": step, "mean_reward": None, "n_groups": 0}

        if a.dry_run:
            log.info("step %d [dry-run] rewards=%s", step, rewards_seen)
            return {"step": step, "mean_reward": sum(rewards_seen) / len(rewards_seen),
                    "n_groups": len(groups)}

        # 5) GRPO: group-centered advantages -> datums -> update
        import torch  # noqa: F401  (cookbook imports need it)
        from tinker_cookbook.rl.data_processing import (
            compute_advantages, assemble_training_data, remove_constant_reward_groups,
        )

        groups = remove_constant_reward_groups(groups)
        advantages_P = compute_advantages(groups)
        data_D, meta_D = assemble_training_data(groups, advantages_P)
        # tinker service schema mismatch: cookbook 0.5.7 emits a 'mask' key
        # (parse-error masking) that the current server-side
        # importance_sampling loss does not accept. Our bridge-sampled
        # trajectories never set the parse-error metric, so mask is
        # uniformly 1.0 and dropping it is lossless.
        for datum in data_D:
            datum.loss_fn_inputs.pop("mask", None)
        log.info("step %d: %d groups -> %d datums", step, len(groups), len(data_D))

        fwd_bwd_future = await self.training_client.forward_backward_async(
            data=data_D, loss_fn="importance_sampling")
        import tinker as _tinker
        optim_future = await self.training_client.optim_step_async(
            _tinker.AdamParams(learning_rate=a.lr, beta1=0.9, beta2=0.95, eps=1e-8))
        await fwd_bwd_future.result_async()
        await optim_future.result_async()

        # periodic named checkpoint for the eval arm
        if (step + 1) % a.save_every == 0:
            ck_fut = await self.training_client.save_weights_for_sampler_async(
                name=f"{a.run_name}-step{step + 1}")
            ck = await ck_fut.result_async()
            self.saved_checkpoints.append(ck.path)
            log.info("step %d: eval checkpoint -> %s", step, ck.path)

        return {"step": step, "mean_reward": sum(rewards_seen) / len(rewards_seen),
                "n_groups": len(groups)}

    # -- entry ---------------------------------------------------------------
    async def run(self) -> None:
        a = self.args
        task_dirs = sorted(d for d in Path(a.task_dir).iterdir() if d.is_dir())
        if a.include:
            task_dirs = [d for d in task_dirs if d.name in set(a.include)]
        if not task_dirs:
            log.error("no tasks in %s", a.task_dir)
            sys.exit(1)
        log.info("pi-GRPO: %d tasks x group=%d, steps=%d%s",
                 len(task_dirs), a.group_size, a.steps,
                 " [DRY RUN]" if a.dry_run else "")

        if not a.dry_run:
            import tinker as tinker_sdk
            self.service_client = tinker_sdk.ServiceClient()
            self.training_client = await (
                self.service_client.create_lora_training_client_async(
                    base_model=a.model, rank=a.lora_rank)
            )

        batch_size = a.tasks_per_batch
        history = []
        for step in range(a.steps):
            batch = [task_dirs[(step * batch_size + i) % len(task_dirs)]
                     for i in range(min(batch_size, len(task_dirs)))]
            summary = await self.train_step(step, batch)
            history.append(summary)
            # persist after EVERY step: a kill / sleep mid-run must not lose
            # the curve (2026-10-11: 20h sleep + user kill wiped it).
            self._flush_history(history)
            log.info("step summary: %s", summary)

        await self.stop_bridge()
        self._flush_history(history)
        log.info("done. history + checkpoints -> %s", Path(a.log_path))


def _fake_records(result_row: dict[str, Any]) -> list[dict[str, Any]]:
    """Dry-run bridge records: two synthetic sampling calls per rollout."""
    h = hash(result_row["key"])
    return [
        {"prompt_ids": [100 + (h % 50) + i for i in range(12)],
         "completion_ids": [200 + (h % 30) + i for i in range(8)],
         "logprobs": [-0.05 * i - 0.01 * (h % 7) for i in range(8)],
         "finish_reason": "tool_calls"},
        {"prompt_ids": [300 + (h % 40) + i for i in range(14)],
         "completion_ids": [400 + (h % 25) + i for i in range(6)],
         "logprobs": [-0.04 * i - 0.01 * (h % 5) for i in range(6)],
         "finish_reason": "stop"},
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-dir", type=Path, default=REPO / "out" / "runs" / "tasks")
    p.add_argument("-m", "--model", default="openai/gpt-oss-20b")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--tasks-per-batch", type=int, default=4)
    p.add_argument("--rollout-timeout", type=int, default=1800)
    p.add_argument("--harbor-n-concurrent", type=int, default=4,
                   help="(legacy, unused) harbor per-job -n; every rollout job "
                        "has 1 trial, so use --rollout-concurrency instead.")
    p.add_argument("--rollout-concurrency", type=int, default=3,
                   help="max in-flight harbor rollout subprocesses. Each holds "
                        "one container at 2 cpus / 4 GB; 3 fits the 15.7 GB "
                        "host with headroom for the bridge + verifier.")
    p.add_argument("--setup-timeout-multiplier", type=float, default=2.0,
                   help="agent setup timeout multiplier; the pi install chain "
                        "alone measures ~310s through the proxy vs harbor's "
                        "default 360s budget, which any concurrency busts.")
    p.add_argument("--sandbox-proxy", default="http://host.docker.internal:7897",
                   help="HTTP proxy REACHABLE FROM INSIDE the sandbox, injected "
                        "into docker build + agent install + verifier. Empty "
                        "string disables proxy injection.")
    p.add_argument("--bridge-port", type=int, default=8092)
    p.add_argument("--harbor-bin", default="harbor")
    p.add_argument("--records-dir", default=str(REPO / "out" / "rl_runs" / "pi_bridge"))
    p.add_argument("--log-path", default=str(REPO / "out" / "rl_runs" / "pi_grpo"))
    p.add_argument("--run-name", default="pi-grpo")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--include", nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="offline rehearsal: fake harbor/pi/tinker, real assembly code")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    # ALSO persist the trainer log to disk. It used to go only to stdout, and
    # the launch wraps it in `| tail -60`; on kill/sleep the whole trainer log
    # (saved sampler weights, eval checkpoints, step summaries) was lost.
    log_dir = Path(args.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    _fh = (log_dir / "trainer.log").open("a", encoding="utf-8")
    logging.getLogger().addHandler(logging.StreamHandler(_fh))
    trainer = PiGRPOTrainer(args)
    try:
        asyncio.run(trainer.run())
    except KeyboardInterrupt:
        log.info("interrupted; flushing history")
        asyncio.run(trainer.stop_bridge())
    finally:
        try:
            trainer._flush_history(getattr(trainer, "_history_cache", []))
        except Exception:  # never mask the real exit
            pass
        try:
            _fh.flush()
            _fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
