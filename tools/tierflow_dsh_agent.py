"""Harbor agent adapter: DeepSeek Harness (``dsh``) one-shot headless mode.

Why this exists
---------------
DeepSeek Harness is not one of harbor 0.3.0's built-in agents, so it has no
installer and no invocation contract. Its one-shot surface is
``dsh --profile headless "<task>"``: the agent works the task with its full
tool set, streams reasoning to stderr, prints the final assistant message on
stdout, and exits (0 completed, 1 aborted).

Wire protocol (this is the part that is easy to get wrong)
----------------------------------------------------------
``dsh`` does **not** speak the OpenAI Chat Completions API. Its default route
``deepseek-official`` uses the **DeepSeek Messages** transport
(``@deepseek-ai/dsh-llm-deepseek``), which posts to ``{baseURL}/v1/messages``
with ``x-api-key`` + ``anthropic-version`` and sends reasoning via
``output_config.effort``. An OpenAI-compatible gateway must therefore expose a
Messages-compatible surface (https://tierflow.cn/v1 does; VERIFIED 2026-10-12
that ``POST /v1/messages`` returns a valid message body).

The ``max_tokens`` trap
-----------------------
That transport's default per-request output cap is **256,000**, which the
gateway rejects::

    {"error":{"type":"invalid_request",
              "message":"Range of max_tokens should be [1, 131072]"}}

VERIFIED by capturing dsh's real request body against a local echo server: the
wire payload carried ``"max_tokens":256000``. The cap is a *config field* of the
``llm-deepseek`` row, so the adapter pins it with a profile patch. dsh applies a
``--patch`` overlay **after** the bundle layers, and a patch **replaces the
targeted row's whole ``config``** rather than merging into it — so the patch
below restates every field it needs rather than only the changed one.

With ``maxTokens: 131072`` the run completes end to end (VERIFIED: exit 0,
stdout ``DSH-PATCH-OK``, reasoning on stderr).

Model routing
-------------
dsh resolves the model from the ``agent-default-model`` plugin, which the same
patch pins to the gateway's model id.

Environment (supply via ``--agent-env``):
  DEEPSEEK_API_KEY    required — the gateway key (also read by dsh's apiKeyEnv)
  DEEPSEEK_BASE_URL   default https://tierflow.cn/v1
  DSH_MODEL           default tierflow_pro (the gateway's model id)
  DSH_MAX_TOKENS      default 131072 (must stay <= the gateway's ceiling)
  DSH_HOME            default $HOME/.dsh-eval (kept off the agent's default home)
"""

from __future__ import annotations

import shlex
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseSandbox
from harbor.models.agent.context import AgentContext

DEFAULT_BASE_URL = "https://tierflow.cn/v1"
DEFAULT_MODEL = "tierflow_pro"
DEFAULT_MAX_TOKENS = 131072

# Applied with `--patch`, i.e. after the bundle layers, per dsh's documented
# precedence. Each entry replaces the target row's *entire* config, so every
# field the row needs is restated here.
PATCH_TEMPLATE = """\
# rsibio-paper2env: evaluator overlay — route dsh at an OpenAI-compatible
# gateway whose Messages endpoint caps max_tokens below dsh's 256k default.
- id: llm-deepseek
  config:
    apiKeyEnv: DEEPSEEK_API_KEY
    baseURL: {base_url}
    reasoningEffort: {effort}
    maxTokens: {max_tokens}
    defaultContextWindow: {context_window}
- id: agent-default-model
  config:
    provider: deepseek-official
    model: {model}
"""


class TierflowDsh(BaseInstalledAgent):
    """DeepSeek Harness driven in one-shot headless mode."""

    _OUTPUT_FILENAME = "dsh.txt"

    @staticmethod
    def name() -> str:
        return "dsh"

    def get_version_command(self) -> str | None:
        return "dsh --version"

    def parse_version(self, stdout: str) -> str:
        return stdout.strip().splitlines()[-1].strip() if stdout.strip() else ""

    def __init__(self, *args: Any, base_url: str | None = None,
                 max_tokens: int | None = None, effort: str = "high",
                 context_window: int = 131072, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.base_url = (base_url or self._get_env("DEEPSEEK_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.dsh_model = self._get_env("DSH_MODEL") or DEFAULT_MODEL
        self.max_tokens = int(max_tokens or self._get_env("DSH_MAX_TOKENS")
                              or DEFAULT_MAX_TOKENS)
        self.effort = self._get_env("DSH_REASONING_EFFORT") or effort
        self.context_window = int(context_window)
        self.dsh_home = self._get_env("DSH_HOME") or "$HOME/.dsh-eval"

    async def install(self, environment: BaseSandbox) -> None:
        """Install a Node runtime dsh can run on, then dsh itself.

        The task base image is ``python:3.11-slim`` (Debian), so apt is
        present but its ``nodejs`` package may predate dsh's requirement.
        Node 22 is installed from NodeSource when the distro node is too old,
        which keeps this working on both slim and full base images.
        """
        await self.exec_as_root(
            environment,
            command=(
                "set -e; "
                "if command -v apt-get &>/dev/null; then "
                "  apt-get update && apt-get install -y curl ca-certificates "
                "    git ripgrep procps gnupg; "
                "elif command -v apk &>/dev/null; then "
                "  apk add --no-cache curl ca-certificates git ripgrep procps; "
                "else echo 'no known package manager' >&2; exit 1; fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        version_spec = f"@{self._version}" if self._version else "@latest"
        await self.exec_as_root(
            environment,
            command=(
                "set -e; "
                # Prefer the distro node when it is already new enough;
                # otherwise pull Node 22 from NodeSource.
                "need_node=0; "
                "if command -v node &>/dev/null; then "
                "  major=$(node -p 'process.versions.node.split(\".\")[0]' 2>/dev/null || echo 0); "
                "  [ \"$major\" -ge 20 ] 2>/dev/null || need_node=1; "
                "else need_node=1; fi; "
                "if [ \"$need_node\" = 1 ]; then "
                "  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && "
                "  apt-get install -y nodejs; "
                "fi; "
                "node -v && npm -v && "
                f"npm install -g @deepseek-ai/dsh{version_spec} && "
                "dsh --version"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseSandbox,
        context: AgentContext,
    ) -> None:
        escaped = shlex.quote(instruction)
        patch = PATCH_TEMPLATE.format(
            base_url=self.base_url,
            model=self.dsh_model,
            max_tokens=self.max_tokens,
            effort=self.effort,
            context_window=self.context_window,
        )
        # The patch travels as an env var so no shell quoting is needed for the
        # YAML body; it is materialised inside the sandbox before the run.
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f'export DSH_HOME="{self.dsh_home}"; '
                'mkdir -p "$DSH_HOME"; '
                'printf "%s\\n" "$DSH_PATCH" > "$DSH_HOME/eval-patch.yml"; '
                f'export DEEPSEEK_API_KEY="{self._get_env("DEEPSEEK_API_KEY") or ""}"; '
                f'export DEEPSEEK_BASE_URL="{self.base_url}"; '
                f"dsh --profile headless --patch \"$DSH_HOME/eval-patch.yml\" {escaped} "
                f"2>&1 </dev/null | stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}"
            ),
            env={"DSH_PATCH": patch},
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        """No structured event stream in headless mode; nothing to parse."""
        return
