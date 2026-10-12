"""Multi-turn agentic RL training on rsibio synthesized environments.

This is the REAL Stage-2 experiment the paper claims: the policy is an AGENT
(bash tool, multi-turn, sandboxed Linux env), trained with GRPO on the tasks
synthesized by our Stage-1 pipeline. Aligned with harbor-cookbook/harbor_rl.

Key differences from tools/rl_tinker_train.py (single-turn smoke):
  - env = full harbor task (container + data + verifier), not a text prompt
  - policy loop: agent <-> bash tool, max N turns
  - reward = harbor verifier grade (task.toml check()), the same four-gate
    discipline that offline verify enforces, executed for real
  - trajectories (multi-turn messages) become training datums

Usage (after Docker engine is up; .rlvenv has tinker 0.33.1 + harbor rl-4d0):
  TINKER_API_KEY=... .rlvenv/Scripts/python.exe tools/rl_agentic_train.py \
      --task-dir out/runs/tasks --steps 20 --group-size 4 --max-turns 12
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]

import tinker  # noqa: E402
from tinker import types  # noqa: E402

from harbor.environments.factory import SandboxFactory
from harbor.models.environment_type import SandboxType
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.rl import RLEnvironment, ToolInput
from harbor.rl.tools import BashTool

from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl import Env, EnvGroupBuilder, RLDataset
from tinker_cookbook.rl import train
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use import build_agent_tool_env, simple_tool_result, tool
from tinker_cookbook.tool_use.types import ToolResult

log = logging.getLogger(__name__)

BASE_MODEL = "openai/gpt-oss-20b"
SYSTEM_PROMPT = (
    "You are a resourceful life-science research engineer working in a sandboxed "
    "Linux environment. You have a bash tool. Explore the task files first "
    "(ls, cat), understand the required output format, then solve the task "
    "step by step. Verify your work before finishing."
)


def make_bash_tool(env: RLEnvironment, env_name: str):
    @tool
    async def bash(command: str) -> ToolResult:
        """Execute a bash command in the sandbox."""
        outputs = await env.step([ToolInput(name="bash", arguments={"command": command})])
        text = "\n".join(c.text for c in outputs[0].content)
        log.info("[%s] bash: %s -> %d chars (err=%s)",
                 env_name, command[:120], len(text), outputs[0].isError)
        return simple_tool_result(text)

    return bash


def make_reward_fn(env: RLEnvironment, env_name: str):
    async def reward_fn(messages: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
        result = await env.grade()
        reward = result.rewards.get("reward", 0.0)
        log.info("[%s] grade -> %s", env_name, result.rewards)
        return reward, result.rewards

    return reward_fn


class TaskGroupBuilder(EnvGroupBuilder):
    """One harbor task -> group_size parallel rollouts (GRPO group)."""

    def __init__(
        self,
        task_path: Path,
        model_name: str,
        group_size: int,
        max_turns: int,
        sandbox_type: str,
        renderer_name: str,
    ):
        self.task_path = task_path
        self.model_name = model_name
        self.group_size = group_size
        self.max_turns = max_turns
        self.sandbox_type = sandbox_type
        self.renderer_name = renderer_name
        self._envs: list[RLEnvironment] = []

    async def make_envs(self) -> Sequence[Env]:
        task = Task(self.task_path)
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        envs: list[Env] = []
        for i in range(self.group_size):
            sid = uuid.uuid4().hex[:12]
            trial_dir = Path(tempfile.mkdtemp(prefix=f"rl-{task.name}-{i}-"))
            sandbox = SandboxFactory.create_sandbox(
                type=SandboxType(self.sandbox_type),
                environment_dir=task.paths.environment_dir,
                environment_name=f"{task.name}-g{i}",
                session_id=sid,
                trial_paths=TrialPaths(trial_dir=trial_dir),
                task_env_config=task.config.environment,
            )
            rl_env = RLEnvironment(sandbox=sandbox, task=task)
            await rl_env.start(tools=[BashTool()])
            self._envs.append(rl_env)
            env_name = f"{task.name}/g{i}"
            bash = make_bash_tool(rl_env, env_name)
            messages = renderer.create_conversation_prefix_with_tools(
                tools=[bash.to_spec()], system_prompt=SYSTEM_PROMPT
            ) + [{"role": "user", "content": rl_env.get_prompt()}]
            envs.append(
                build_agent_tool_env(
                    renderer=renderer,
                    tools=[bash],
                    initial_messages=messages,
                    reward_fn=make_reward_fn(rl_env, env_name),
                    max_turns=self.max_turns,
                )
            )
        return envs

    async def cleanup(self) -> None:
        await asyncio.gather(*(e.stop() for e in self._envs), return_exceptions=True)
        self._envs.clear()

    def logging_tags(self) -> list[str]:
        return [self.task_path.name]


class TaskDataset(RLDataset):
    """Groups = one per task dir; batch = group_size builders per step."""

    def __init__(self, builders: list[TaskGroupBuilder], batch_size: int):
        self._builders = builders
        self._batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self._batch_size
        return self._builders[start : start + self._batch_size]

    def __len__(self) -> int:
        return -(-len(self._builders) // self._batch_size)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-dir", type=Path, default=REPO / "out" / "runs" / "tasks")
    p.add_argument("-m", "--model", default=BASE_MODEL)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("-s", "--sandbox", default="docker")
    p.add_argument("--include", nargs="*", default=None,
                   help="only train on these task names (subset selection)")
    p.add_argument("--log-path", default=str(REPO / "out" / "rl_runs" / "agentic"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    # collect task dirs (exclude non-directories like manifest.jsonl)
    task_dirs = sorted(d for d in args.task_dir.iterdir() if d.is_dir())
    if args.include:
        task_dirs = [d for d in task_dirs if d.name in set(args.include)]
    if not task_dirs:
        log.error("No tasks found in %s", args.task_dir)
        sys.exit(1)
    log.info("Training on %d tasks: %s", len(task_dirs), [d.name for d in task_dirs])

    async def build_datasets():
        renderer_name = get_recommended_renderer_name(args.model)
        builders = [
            TaskGroupBuilder(
                d, args.model, args.group_size, args.max_turns,
                args.sandbox, renderer_name,
            )
            for d in task_dirs
        ]
        return TaskDataset(builders, args.batch_size), None

    config = train.Config(
        model_name=args.model,
        renderer_name=get_recommended_renderer_name(args.model),
        dataset_builder=build_datasets,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        max_steps=args.steps,
        log_path=args.log_path,
    )
    log.info("Starting agentic RL: model=%s tasks=%d steps=%d group=%d turns<=%d sandbox=%s",
             args.model, len(task_dirs), args.steps, args.group_size, args.max_turns, args.sandbox)
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
