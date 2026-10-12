"""Harbor agent plugin: pi wired to the tinker sampling bridge (pi-GRPO).

harbor 0.3.0's built-in ``harbor.agents.installed.pi.Pi`` hard-codes a
provider whitelist (openai/anthropic/...) and passes ``--provider`` +
``--model``; it has no notion of models.json custom providers. This module
subclasses Pi to point it at OUR bridge instead:

  1. ``install()``      - unchanged (nvm + node 22 + pi from npm)
  2. ``run()``          - writes a models.json into $PI_RL_DIR inside the
     sandbox defining a custom model under the *built-in* ``openai``
     provider with a model-level ``baseUrl`` pointing at the bridge
     (http://host.docker.internal:<port>/v1). pi 0.73.1 resolves the apiKey
     via ``resolveConfigValue``: a value that names an environment variable
     (``PI_BRIDGE_KEY``) is looked up in the env first, so the Bearer token
     the bridge records = the rollout uuid, set via harbor's ``--agent-env``.

The model id must collide with a real OpenAI model ONLY in name; we pick
``gpt-oss-rl`` so pi never routes to api.openai.com (model-level baseUrl
override wins over the provider default).

Import path for harbor::

    harbor run ... --agent-import-path rl_pi_agent:BridgePi \\
        --model gpt-oss-rl

(module is found via PYTHONPATH - pi_grpo_train.py exports it).
"""

from __future__ import annotations

import shlex
from typing import Any

from harbor.agents.installed.base import with_prompt_template
from harbor.models.agent.context import AgentContext
from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseSandbox

# models.json written into the sandbox before pi runs. Note:
#  - provider "openai" is BUILT-IN, so baseUrl/apiKey/api are optional per
#    pi's validateConfig; the model-level baseUrl override routes to the
#    bridge and the apiKey is resolved from the env (PI_BRIDGE_KEY).
#  - requiresToolResultName: bridge forwards tool results with ``name``.
MODELS_JSON = """{{
  "providers": {{
    "openai": {{
      "apiKey": "PI_BRIDGE_KEY",
      "models": [
        {{
          "id": "gpt-oss-rl",
          "baseUrl": "http://host.docker.internal:{bridge_port}/v1",
          "api": "openai-completions",
          "contextWindow": 131072,
          "maxTokens": {max_tokens},
          "compat": {{"requiresToolResultName": true}}
        }}
      ]
    }}
  }}
}}
"""


class BridgePi(Pi):
    """Pi agent whose model is the tinker-bridged policy checkpoint."""

    def __init__(self, *args: Any, bridge_port: int = 8092,
                 max_tokens: int = 8192, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.bridge_port = int(bridge_port)
        self.max_tokens = int(max_tokens)

    async def install(self, environment: BaseSandbox) -> None:
        """Stock Pi.install + explicit ``nvm use``.

        EMPIRICAL (2026-10-10, python:3.11-slim + ubuntu:24.04): after
        ``nvm install 22`` the npm binary is NOT on PATH in this exec chain
        (exit 127 at ``npm -v``) until ``nvm use 22`` runs. The stock
        harbor 0.3.0 install command fails here, so we override it.
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
        rl_dir = "$HOME/.pi-rl"

        # 1) models.json: model-level baseUrl override -> bridge; apiKey from env
        models_json = MODELS_JSON.format(
            bridge_port=self.bridge_port, max_tokens=self.max_tokens)
        # VERIFIED IN A LIVE CONTAINER (2026-10-10): nvm-sourced PATH must be
        # re-exported inside the same bash invocation or `pi` is not found;
        # pi 0.73.1 resolves apiKey "PI_BRIDGE_KEY" as an env-var NAME (so the
        # Bearer token = rollout uuid = the bridge's record key).
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {rl_dir} && "
                f"cat > {rl_dir}/models.json <<'EOF'\n"
                f"{models_json}"
                f"EOF\n"
                f". ~/.nvm/nvm.sh; "
                f"PI_CODING_AGENT_DIR={rl_dir} pi --print --mode json --no-session "
                f"--provider openai --model gpt-oss-rl "
                f"{escaped} "
                f"2>&1 </dev/null | stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}"
            ),
            env={},
        )
