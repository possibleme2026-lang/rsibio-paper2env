"""Harbor agent adapter: OpenAI Codex CLI wired to the tierflow gateway.

Why this exists
---------------
harbor 0.3.0's built-in ``Codex`` agent only exports ``OPENAI_BASE_URL`` /
``OPENAI_API_KEY`` (see ``harbor/agents/installed/codex.py:run``). That is not
sufficient for a third-party gateway, for two independent reasons discovered
empirically (2026-10-12, codex-cli 0.162.1 in python:3.11-slim):

1. **The transport is not chosen by OPENAI_BASE_URL.** Codex CLI resolves its
   provider from ``$CODEX_HOME/config.toml``; with no provider table it used its
   built-in OpenAI defaults and opened ``wss://api.openai.com/v1/responses``,
   which is unreachable here::

       ERROR codex_api::endpoint::responses_websocket: failed to connect to
       websocket: IO error: tls handshake eof, url: wss://api.openai.com/v1/responses

   The run then sat in "Reconnecting... waiting for network" until the agent
   timeout, so every trial scored 0 for a harness-plumbing reason.

2. **``wire_api = "chat"`` no longer exists.** The obvious fix -- declare a
   provider with the Chat Completions wire format -- is rejected outright::

       Error loading config.toml: `wire_api = "chat"` is no longer supported.
       How to fix: set `wire_api = "responses"` in your provider config.

   So the gateway must expose an OpenAI *Responses* endpoint. tierflow does
   (VERIFIED: ``POST https://tierflow.cn/v1/responses`` returns a
   ``object: "response"`` body routed to a real model), as do most modern
   OpenAI-compatible routers.

This subclass therefore writes a ``config.toml`` declaring a named provider
with ``wire_api = "responses"`` and points ``model_provider`` at it, then runs
the stock one-shot ``codex exec`` command. Everything else (install, auth.json,
trajectory parsing) is inherited unchanged.

Import path for harbor::

    harbor run ... \\
        --agent-import-path tierflow_codex_agent:TierflowCodex \\
        --model tierflow_pro

Environment (supply via ``--agent-env``):
  OPENAI_API_KEY             required — the gateway key
  OPENAI_BASE_URL            default https://tierflow.cn/v1
  CODEX_FORCE_API_KEY=1      honoured by the stock agent; keep it set so no
                             ~/.codex/auth.json from an image leaks in
  TIERFLOW_CODEX_PROVIDER_ID default tierflow (the config.toml table name)
  TIERFLOW_CODEX_WIRE_API    default responses
"""

from __future__ import annotations

import shlex
from typing import Any

from harbor.agents.installed.codex import Codex
from harbor.agents.installed.base import with_prompt_template
from harbor.environments.base import BaseSandbox
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

DEFAULT_BASE_URL = "https://tierflow.cn/v1"
DEFAULT_PROVIDER_ID = "tierflow"
DEFAULT_WIRE_API = "responses"

CONFIG_TOML = """\
model = "{model}"
model_provider = "{provider_id}"
model_reasoning_effort = "low"

[model_providers.{provider_id}]
name = "{provider_name}"
base_url = "{base_url}"
env_key = "OPENAI_API_KEY"
wire_api = "{wire_api}"
"""


class TierflowCodex(Codex):
    """Codex CLI whose provider table points at an OpenAI-compatible gateway."""

    @staticmethod
    def name() -> str:
        return "tierflow-codex"

    def __init__(self, *args: Any, base_url: str | None = None,
                 provider_id: str | None = None, wire_api: str | None = None,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.base_url = (base_url or self._get_env("OPENAI_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.provider_id = (provider_id
                            or self._get_env("TIERFLOW_CODEX_PROVIDER_ID")
                            or DEFAULT_PROVIDER_ID)
        self.wire_api = (wire_api
                         or self._get_env("TIERFLOW_CODEX_WIRE_API")
                         or DEFAULT_WIRE_API)

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseSandbox,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError("Model name is required")
        model = self.model_name.split("/")[-1]
        escaped = shlex.quote(instruction)
        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""
        agent_dir = EnvironmentPaths.agent_dir.as_posix()

        config_toml = CONFIG_TOML.format(
            model=model,
            provider_id=self.provider_id,
            provider_name="Tierflow Gateway",
            base_url=self.base_url,
            wire_api=self.wire_api,
        )

        # CODEX_HOME is the agent dir, same as the stock agent, so the CLI
        # finds auth.json and (now) our provider table in one place.
        env: dict[str, str] = {
            "CODEX_HOME": agent_dir,
            "OPENAI_API_KEY": self._get_env("OPENAI_API_KEY") or "",
            "OPENAI_BASE_URL": self.base_url,
        }

        setup_command = (
            f'mkdir -p "$CODEX_HOME" /tmp/codex-secrets\n'
            'cat >"$CODEX_HOME/config.toml" <<\'CODEXCONF\'\n'
            f"{config_toml}"
            "CODEXCONF\n"
            # The stock call chain expects auth.json; provide it via a symlink
            # so the key never persists in CODEX_HOME after the run.
            'cat >/tmp/codex-secrets/auth.json <<EOF\n'
            '{\n  "OPENAI_API_KEY": "${OPENAI_API_KEY}"\n}\n'
            "EOF\n"
            'ln -sf /tmp/codex-secrets/auth.json "$CODEX_HOME/auth.json"\n'
        )
        await self.exec_as_agent(environment, command=setup_command, env=env)

        try:
            await self.exec_as_agent(
                environment,
                command=(
                    "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                    "codex exec "
                    "--dangerously-bypass-approvals-and-sandbox "
                    "--skip-git-repo-check "
                    f"--model {model} "
                    "--json "
                    "--enable unified_exec "
                    f"{cli_flags_arg}"
                    "-- "
                    f"{escaped} "
                    f"2>&1 </dev/null | tee {agent_dir}/{self._OUTPUT_FILENAME}"
                ),
                env=env,
            )
        finally:
            try:
                await self.exec_as_agent(
                    environment,
                    command=('rm -rf /tmp/codex-secrets "$CODEX_HOME/auth.json" '
                             '"$CODEX_HOME/tmp"'),
                    env={"CODEX_HOME": agent_dir},
                )
            except Exception:  # noqa: BLE001 - cleanup is best effort
                pass
