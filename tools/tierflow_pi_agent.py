"""Harbor agent adapter: pi-coding-agent wired to the tierflow gateway.

Why this exists
---------------
harbor 0.3.0's built-in ``Pi`` agent (``harbor.agents.installed.pi.Pi``)
hard-codes a provider whitelist (openai/anthropic/google/...) and derives its
base URL from that provider's own environment. There is no supported way to
point stock Pi at an OpenAI-*compatible* third-party gateway like
https://tierflow.cn/v1, because pi's ``models.json`` custom-provider mechanism
is never used by the stock agent.

This subclass does exactly that, mirroring the technique already proven in
``rl_pi_agent.BridgePi`` (which points pi at a local sampling bridge):

  1. reuse the stock install (nvm + node 22 + ``@mariozechner/pi-coding-agent``)
  2. write ``$PI_TF_DIR/models.json`` declaring our gateway as a model-level
     ``baseUrl`` override under the built-in ``openai`` provider *name*, with
     the apiKey resolved from an environment variable — so the real key never
     lands in a file that outlives the run
  3. run pi headless, ``--print --mode json --no-session``

Import path for harbor::

    harbor run ... \\
        --agent-import-path tierflow_pi_agent:TierflowPi \\
        --model openai/tierflow_pro

Environment (supply via ``--agent-env``):
  PI_TIERFLOW_BASE_URL   default https://tierflow.cn/v1
  PI_TIERFLOW_MODEL      default tierflow_pro
  PI_TIERFLOW_API_KEY    required — the gateway key
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseSandbox
from harbor.models.agent.context import AgentContext

DEFAULT_BASE_URL = "https://tierflow.cn/v1"
DEFAULT_MODEL = "tierflow_pro"

MODELS_JSON = """{{
  "providers": {{
    "openai": {{
      "apiKey": "PI_TIERFLOW_API_KEY",
      "models": [
        {{
          "id": "{model}",
          "baseUrl": "{base_url}",
          "api": "openai-completions",
          "contextWindow": 131072,
          "maxTokens": 16384,
          "compat": {{"requiresToolResultName": true}}
        }}
      ]
    }}
  }}
}}
"""


class TierflowPi(Pi):
    """pi-coding-agent whose model is served by the tierflow gateway."""

    def __init__(self, *args: Any, base_url: str | None = None,
                 max_tokens: int = 16384, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.base_url = (base_url or self._get_env("PI_TIERFLOW_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.tf_model = self._get_env("PI_TIERFLOW_MODEL") or DEFAULT_MODEL
        self.max_tokens = int(max_tokens)

    async def install(self, environment: BaseSandbox) -> None:
        """Stock install + explicit ``nvm use``.

        EMPIRICAL (2026-10-10, python:3.11-slim + ubuntu:24.04): after
        ``nvm install 22`` the npm binary is NOT on PATH in this exec chain
        (exit 127 at ``npm -v``) until ``nvm use 22`` runs. The stock
        harbor 0.3.0 install chain fails there, so this override adds it.
        """
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        version_spec = f"@{self._version}" if self._version else "@latest"
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && "
                'export NVM_DIR="$HOME/.nvm" && '
                '\\. "$NVM_DIR/nvm.sh" || true && '
                "command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } && "
                "nvm install 22 && nvm use 22 && npm -v && "
                f"npm install -g @mariozechner/pi-coding-agent{version_spec} && "
                "pi --version"
            ),
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseSandbox,
        context: AgentContext,
    ) -> None:
        escaped = shlex.quote(instruction)
        tf_dir = "$HOME/.pi-tf"
        models_json = MODELS_JSON.format(model=self.tf_model,
                                         base_url=self.base_url)
        # The apiKey value in models.json is an env-var NAME; pi resolves it
        # from the process environment, which harbor's --agent-env seeded.
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {tf_dir} && "
                f"cat > {tf_dir}/models.json <<'EOF'\n"
                f"{models_json}"
                f"EOF\n"
                f". ~/.nvm/nvm.sh; "
                f"PI_CODING_AGENT_DIR={tf_dir} pi --print --mode json --no-session "
                f"--provider openai --model {self.tf_model} "
                f"{escaped} "
                f"2>&1 </dev/null | stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}"
            ),
            env={"PI_TIERFLOW_API_KEY": self._get_env("PI_TIERFLOW_API_KEY") or ""},
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Parse pi's JSON event stream for token accounting (best effort)."""
        output_file = self.logs_dir / self._OUTPUT_FILENAME
        if not output_file.exists():
            return
        total_in = total_out = total_read = total_write = 0
        cost = 0.0
        for line in output_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "message_end":
                continue
            msg = event.get("message") or {}
            if msg.get("role") != "assistant":
                continue
            usage = msg.get("usage") or {}
            total_in += usage.get("input", 0) or 0
            total_out += usage.get("output", 0) or 0
            total_read += usage.get("cacheRead", 0) or 0
            total_write += usage.get("cacheWrite", 0) or 0
            cost += (usage.get("cost") or {}).get("total", 0.0) or 0.0
        context.n_input_tokens = total_in + total_read
        context.n_output_tokens = total_out
        context.n_cache_tokens = total_read
        context.cost_usd = cost if cost > 0 else None
