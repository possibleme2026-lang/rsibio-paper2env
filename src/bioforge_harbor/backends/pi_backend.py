"""RealPiBackend: drive the local ``pi`` coding agent headlessly.

Hard-won constraints encoded here (each one caused a silent failure before):

1. **Never invoke the ``pi`` shim on Windows.** It is a ``.CMD`` wrapper and
   CMD eats newlines in multi-line arguments, so structured prompts arrive
   truncated. We resolve ``node`` + ``cli.js`` and exec directly.
2. **stdin must be DEVNULL.** ``pi -p`` blocks forever waiting on stdin
   otherwise — and the symptom (hang, no output, no exit code) is identical to
   a missing API key, which wastes hours on the wrong diagnosis.
3. **Output goes to files, not pipes.** pi spawns helper processes; with
   ``capture_output=True`` a grandchild holding the pipe write end can keep
   ``communicate()`` from ever seeing EOF (see util/subproc.py docstring).
4. **--provider and --model must both be explicit.** A stray
   ``DEEPSEEK_API_KEY`` in the environment makes pi's default model
   resolution override the provider you meant to use -> 401 from a provider
   you never selected.
5. **Proxy lives in ~/.pi/agent/auth.json**, under the credential's ``env``
   object. Node's fetch ignores HTTP_PROXY/HTTPS_PROXY from the process env.

This class is fully wired but *not executed* by the framework's own test
suite — offline tests use MockBackend. Set ``BIOFORGE_PI=1`` (or pass
``--backend pi`` on the CLI) to route real calls through it.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .base import BackendError
from .jsonx import extract_json
from ..util.subproc import run_to_file


class PiUnavailable(BackendError):
    pass


def _locate_cli() -> tuple[str, str]:
    """Resolve (node_executable, cli.js) for the pi coding agent."""
    node = os.environ.get("PI_NODE") or shutil.which("node") or ""
    cli = os.environ.get("PI_CLI_JS") or ""
    if cli and not Path(cli).is_file():
        raise PiUnavailable(f"PI_CLI_JS set but not a file: {cli}")
    if node and cli:
        return node, cli

    # Derive from the `pi` shim location: <root>/pi.cmd (or bin/pi) sits next
    # to node_modules/@mariozechner/pi-coding-agent/dist/cli.js.
    found = shutil.which("pi") or ""
    if found:
        absf = Path(found).resolve()
        for base in (absf.parent, absf.parent.parent):
            for pkg in ("@mariozechner", "@earendil-works"):
                cand = base / "node_modules" / pkg / "pi-coding-agent" / "dist" / "cli.js"
                if cand.is_file():
                    cli = str(cand)
                    break
            if cli:
                break
    if not node:
        # managed node first, then system node
        for cand in (
            Path(os.path.expanduser("~")) / ".workbuddy/binaries/node/versions/22.22.2-6/node.exe",
            shutil.which("node.exe") or "",
        ):
            if cand and Path(str(cand)).is_file():
                node = str(cand)
                break
    if not (node and cli):
        raise PiUnavailable(
            "cannot locate pi cli.js / node. Set PI_CLI_JS and PI_NODE, or install "
            "@mariozechner/pi-coding-agent globally."
        )
    return node, cli


class RealPiBackend:
    name = "pi"

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        thinking: str | None = None,
        timeout_sec: float = 240.0,
        extra_flags: list[str] | None = None,
    ):
        self.provider = provider or os.environ.get("PI_PROVIDER", "deepseek")
        self.model = model or os.environ.get("PI_MODEL", "deepseek-v4-flash")
        self.thinking = thinking or os.environ.get("PI_THINKING", "low")
        self.timeout_sec = float(timeout_sec)
        self.extra_flags = list(extra_flags or [])
        self._node_cli: tuple[str, str] | None = None
        self.calls: list[str] = []

    def _command(self, prompt: str) -> list[str]:
        if self._node_cli is None:
            self._node_cli = _locate_cli()
        node, cli = self._node_cli
        return [
            node, cli,
            "-p",
            "--no-session",
            "--no-extensions",
            "--no-skills",
            "--no-context-files",
            "--no-prompt-templates",
            "--no-tools",
            "--provider", self.provider,
            "--model", self.model,
            "--thinking", self.thinking,
            *self.extra_flags,
            prompt,
        ]

    def ask_text(self, tag: str, prompt: str, *, timeout_sec: float | None = None) -> str:
        self.calls.append(tag)
        res = run_to_file(
            self._command(prompt),
            timeout_sec=timeout_sec or self.timeout_sec,
        )
        if res.timed_out:
            raise PiUnavailable(f"pi timed out for tag {tag!r}")
        if res.returncode != 0 or not res.stdout.strip():
            raise PiUnavailable(
                f"pi failed for tag {tag!r}: exit={res.returncode} "
                f"stderr={res.stderr.strip()[:400]!r}"
            )
        return res.stdout.strip()

    def ask_json(
        self,
        tag: str,
        prompt: str,
        *,
        schema_name: str | None = None,
        timeout_sec: float | None = None,
    ) -> dict:
        raw = self.ask_text(tag, prompt, timeout_sec=timeout_sec)
        try:
            obj = extract_json(raw)
        except ValueError as e:
            raise PiUnavailable(f"pi output for tag {tag!r} is not JSON: {e}") from e
        if not isinstance(obj, dict):
            raise PiUnavailable(f"pi output for tag {tag!r} is not a JSON object")
        return obj

    def close(self) -> None:
        pass
