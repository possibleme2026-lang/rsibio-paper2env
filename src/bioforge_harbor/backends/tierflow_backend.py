"""TierflowBackend: OpenAI-compatible HTTP backend (https://tierflow.cn/v1).

Design points, each one earned the hard way on this machine:

1. **Proxy comes from the environment, honored explicitly.** The Windows
   sandbox often has HTTPS_PROXY set; urllib honors it only when the opener is
   built with a ProxyHandler. We build one explicitly so behavior is the same
   with and without the sandbox.
2. **Streaming is mandatory here, not an optimization.** Measured on 2026-10-10:
   the gateway accepts small non-stream requests but a ~5 KB prompt with
   ``stream:false`` never returns a first byte (times out through proxy AND
   direct). With ``stream:true`` the first SSE chunk arrives in <1s. So we
   always request SSE and reassemble ``choices[0].delta.content``.
3. **A fresh opener per attempt.** After a socket timeout the connection pool
   state is unreliable; reusing the opener produced spurious
   "Invalid URL (GET /v1/chat/completions)" 404s from the retry path.
4. **JSON mode is requested but never trusted.** tierflow models may return
   reasoning_content and wrap JSON in prose; extraction goes through the same
   tolerant scanner the pi backend uses.
5. **Retries with backoff on 429/5xx/timeouts.** Shared gateways throttle; a
   single transient failure must not fail a whole pipeline run.
6. **Requests are logged with the stage tag** so every call is auditable
   post-run (same contract as RealPiBackend).

This class is NOT executed by the framework's own test suite — offline tests
use MockBackend. Pass ``--backend tierflow`` to route real calls through it.
Credentials come from TIERFLOW_API_KEY / TIERFLOW_BASE_URL / TIERFLOW_MODEL or
explicit constructor args.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import BackendError
from .jsonx import extract_json

DEFAULT_BASE_URL = "https://tierflow.cn/v1"
DEFAULT_MODEL = "tierflow_pro"
USER_AGENT = "rsibio-paper2env/0.1 (research; contact: RisingSI)"


class TierflowUnavailable(BackendError):
    pass


def _build_opener():
    proxies = urllib.request.getproxies()
    if proxies:
        return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
    return urllib.request.build_opener()


def _consume_sse(resp) -> tuple[str, str]:
    """Read an SSE stream, return (content, finish_reason).

    OpenAI-style chunks: ``data: {json}\\n\\n``, terminated by ``data: [DONE]``.
    We accumulate only assistant text (``delta.content``); ``reasoning_content``
    deltas are ignored — the blueprint JSON lives in the answer, not the CoT.
    """
    parts: list[str] = []
    finish = ""
    while True:
        line = resp.readline()
        if not line:  # EOF
            break
        s = line.decode("utf-8", errors="replace").strip()
        if not s or not s.startswith("data:"):
            continue
        payload = s[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            piece = delta.get("content")
            if piece:
                parts.append(piece)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return "".join(parts), finish


class TierflowBackend:
    name = "tierflow"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_sec: float = 300.0,
        max_retries: int = 3,
        backoff_sec: float = 2.0,
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.environ.get("TIERFLOW_API_KEY") or ""
        self.base_url = (base_url or os.environ.get("TIERFLOW_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("TIERFLOW_MODEL") or DEFAULT_MODEL
        self.timeout_sec = float(timeout_sec)
        self.max_retries = int(max_retries)
        self.backoff_sec = float(backoff_sec)
        self.temperature = float(temperature)
        self.calls: list[str] = []
        if not self.api_key:
            raise TierflowUnavailable(
                "no API key: set TIERFLOW_API_KEY or pass api_key="
            )

    # -- HTTP --------------------------------------------------------------
    def _post_chat(self, messages: list[dict]) -> dict:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            # Fresh opener per attempt: after a timeout the pooled connection
            # is poisoned and reusing it produced bogus 404 responses.
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": USER_AGENT,
                },
                method="POST",
            )
            try:
                with _build_opener().open(req, timeout=self.timeout_sec) as resp:
                    content, finish = _consume_sse(resp)
                if not content.strip():
                    raise TierflowUnavailable(
                        f"empty streamed content (finish={finish!r})")
                # Re-wrap in the non-stream shape so callers/tests see one API.
                return {
                    "choices": [{"message": {"content": content},
                                 "finish_reason": finish}],
                }
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:300]
                except Exception:  # noqa: BLE001
                    pass
                last_err = TierflowUnavailable(f"HTTP {e.code}: {detail}")
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                    TierflowUnavailable) as e:
                last_err = e if isinstance(e, TierflowUnavailable) else \
                    TierflowUnavailable(f"network/parse error: {e}")
            time.sleep(self.backoff_sec * (attempt + 1))
        raise last_err or TierflowUnavailable("request failed")

    # -- AgentBackend protocol ----------------------------------------------
    def ask_text(self, tag: str, prompt: str, *, timeout_sec: float | None = None) -> str:
        self.calls.append(tag)
        data = self._post_chat([{"role": "user", "content": prompt}])
        choices = data.get("choices") or []
        if not choices:
            raise TierflowUnavailable(f"empty choices for tag {tag!r}")
        content = (choices[0].get("message") or {}).get("content") or ""
        if not content.strip():
            raise TierflowUnavailable(f"empty content for tag {tag!r}")
        return content.strip()

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
            raise TierflowUnavailable(
                f"tierflow output for tag {tag!r} is not JSON: {e}"
            ) from e
        if not isinstance(obj, dict):
            raise TierflowUnavailable(f"tierflow output for tag {tag!r} is not a JSON object")
        return obj

    def close(self) -> None:
        pass
